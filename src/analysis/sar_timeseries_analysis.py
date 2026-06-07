import io
import requests
import re
import ee
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageSequence

from utils.collection_utils import _get_aoi, get_collection_s1
from utils.config import (
    QUANTIFICATION_SCALE, REGIONS_OF_INTEREST, WEST_COAST_AGGREGATE, SPIT_REGIONS,
    OUTPUT_PLOTS, OUTPUT_ANIMATIONS, VIS_CHANGE_MAP, VIS_BINARY_WATER_MASK, VIS_SAR_VV,
)
from utils.tidal_utils import filter_bin
from analysis.sar_core import get_otsu_mask


# --------------------------------------------------------
#  Region-of-interest configuration
# --------------------------------------------------------

def build_region_fc() -> ee.FeatureCollection:
    """Assemble the analysis FeatureCollection from config.

    Each feature carries properties: name (str), is_spit (bool).
    Ends with the western coast aggregate (falls back to island AOI if not defined).
    """
    features = []

    for name, geojson in REGIONS_OF_INTEREST.items():
        if geojson is None:
            print(f"{name}: geometry not set -> skipped")
            continue
        geom = ee.Geometry(geojson["features"][0]["geometry"])
        features.append(ee.Feature(geom, {"name": name, "is_spit": name in SPIT_REGIONS}))

    if WEST_COAST_AGGREGATE is not None:
        agg_geom = ee.Geometry(WEST_COAST_AGGREGATE["features"][0]["geometry"])
    else:
        print("WEST_COAST_AGGREGATE not set - using island AOI as aggregate")
        agg_geom = _get_aoi()
    features.append(ee.Feature(agg_geom, {"name": "island_aggregate", "is_spit": False}))

    return ee.FeatureCollection(features)


def _reduce_area_km2(image: ee.Image, region_fc: ee.FeatureCollection,
                     scale: int) -> ee.FeatureCollection:
    """Sum pixel area (km²) per band per region in one reduceRegions call."""
    pixel_area = ee.Image.pixelArea().divide(1e6)
    return (image.toFloat().multiply(pixel_area)
            .reduceRegions(
                collection = region_fc,
                reducer    = ee.Reducer.sum(),
                scale      = scale,
                tileScale  = 4,
            ))


# --------------------------------------------------------
#  GIF generation
# --------------------------------------------------------

def generate_sar_timeseries_gif(col: ee.ImageCollection, mask: bool,
                                fps: int = 2, width: int = 600):
    """Render a GIF of the SAR timeseries (raw VV or water mask)."""
    aoi   = _get_aoi()
    times = col.aggregate_array("system:time_start").getInfo()
    dates = [pd.to_datetime(t, unit="ms", utc=True).strftime("%Y-%m-%d") for t in times]

    def prep_for_gif(img):
        if mask:
            return img.visualize(**VIS_BINARY_WATER_MASK)
        vv_db = img.select("VV").log10().multiply(10)
        return vv_db.visualize(**VIS_SAR_VV)

    col_prepared = col.map(prep_for_gif)

    print("Rendering and downloading raw GIF...")
    gif_url = col_prepared.getVideoThumbURL({
        "dimensions":    width,
        "region":        aoi,
        "framesPerSecond": fps,
        "crs":           "EPSG:3857",
    })
    response = requests.get(gif_url)
    response.raise_for_status()

    raw_gif = Image.open(io.BytesIO(response.content))
    frames  = []

    for i, frame in enumerate(ImageSequence.Iterator(raw_gif)):
        frame      = frame.convert("RGBA")
        draw       = ImageDraw.Draw(frame)
        date_text  = dates[i] if i < len(dates) else "Unknown"
        x, y       = 15, 15
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            draw.text((x + dx, y + dy), date_text, fill="black")
        draw.text((x, y), date_text, fill="white")
        frames.append(frame)

    mode_tag    = "mask" if mask else "vv"
    output_path = f"{OUTPUT_ANIMATIONS}timeseries_sar_{mode_tag}.gif"
    print(f"Saving GIF ({len(frames)} frames) to {output_path}...")
    frames[0].save(
        output_path,
        save_all      = True,
        append_images = frames[1:],
        loop          = 0,
        duration      = int(1000 / fps),
    )
    print("Finished")


# --------------------------------------------------------
#  Entry point 1 — land/water area timeseries
# --------------------------------------------------------

def quantify_timeseries(col: ee.ImageCollection = None,
                        scale: int = QUANTIFICATION_SCALE,
                        export: bool = False) -> pd.DataFrame | None:
    """Return long-format land/water area per region per date.

    Maps get_otsu_mask over the tidally-binned collection.
    """
    if col is None:
        col = get_collection_s1()
        col = filter_bin(col, "near_msl")

    col_size  = col.size().getInfo()
    print(f"\nQuantify_timeseries: {col_size} images")

    region_fc = build_region_fc()

    def process_one(raw_img):
        img   = ee.Image(raw_img)
        mask  = get_otsu_mask(img, scale=scale)
        water = mask.select("water").toFloat().rename("water_km2")
        land  = mask.select("water").Not().toFloat().rename("land_km2")
        date  = img.date().format("YYYY-MM-dd")
        return (_reduce_area_km2(ee.Image.cat([land, water]), region_fc, scale)
                .map(lambda f: f.set("date", date)))

    img_list = col.sort("system:time_start").toList(col_size)
    all_fcs  = img_list.map(process_one)
    flat_fc  = ee.FeatureCollection(all_fcs).flatten()

    if export:
        task = ee.batch.Export.table.toDrive(
            collection  = flat_fc,
            description = "sylt_land_area",
            fileFormat  = "CSV",
        )
        task.start()
        print(f"Export task started ({task.id})")
        return None

    print("Fetching results")
    features = flat_fc.getInfo()["features"]

    rows = []
    for feat in features:
        p     = feat["properties"]
        land  = p.get("land_km2",  0) or 0.0
        water = p.get("water_km2", 0) or 0.0
        total = land + water or 1e-9
        rows.append({
            "date":          p["date"],
            "region":        p["name"],
            "land_km2":      round(land,        4),
            "water_km2":     round(water,       4),
            "land_fraction": round(land / total, 5),
        })

    df = (pd.DataFrame(rows)
          .assign(date=lambda d: pd.to_datetime(d["date"]))
          .sort_values(["region", "date"])
          .reset_index(drop=True))

    _print_timeseries_summary(df)
    return df


def _print_timeseries_summary(df: pd.DataFrame):
    print("\nLand-area summary per region (seasonal envelope):\n")
    grouped = df.sort_values(["region", "date"]).groupby("region")["land_km2"].agg(
        Min="min", Max="max", First="first", Last="last"
    )
    display_df = pd.DataFrame({
        "Region":              grouped.index,
        "Min km2":             grouped["Min"].map("{:.3f}".format),
        "Max km2":             grouped["Max"].map("{:.3f}".format),
        "Amplitude km2":       (grouped["Max"] - grouped["Min"]).map("{:.3f}".format),
        "Start-end delta km2": (grouped["Last"] - grouped["First"]).map("{:+.3f}".format),
    })
    print(display_df.to_string(index=False))
    ranked = (grouped["Max"] - grouped["Min"]).sort_values(ascending=False).index.tolist()
    print(f"  Amplitude ranking: {' > '.join(ranked)}")


def plot_timeseries(df: pd.DataFrame, save: bool = False):
    """Line plot of land area per region over time."""
    fig, ax = plt.subplots(figsize=(13, 5))

    for region, grp in df.groupby("region"):
        grp = grp.sort_values("date")
        ax.plot(grp["date"], grp["land_km2"], marker=".", markersize=3, label=region)

    ax.set_xlabel("Date")
    ax.set_ylabel("Land area (km²)")
    ax.set_title("Land area over time — tidal-binned SAR (near_msl)")
    ax.legend(fontsize=8)
    plt.tight_layout()

    if save:
        path = f"{OUTPUT_PLOTS}timeseries_land_area.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved to {path}")

    plt.show()


# --------------------------------------------------------
#  Entry point 2 — consecutive-pair erosion/accretion timeseries
# --------------------------------------------------------

def quantify_change_timeseries(col: ee.ImageCollection,
                               scale: int = QUANTIFICATION_SCALE,
                               export: bool = False) -> pd.DataFrame | None:
    """Return region-wise erosion/accretion for every consecutive image pair."""
    n = col.size().getInfo()
    print(f"\nQuantify change timeseries: {n} images -> {n - 1} consecutive pairs")

    region_fc = build_region_fc()
    mask_list = (col.sort("system:time_start")
                 .map(lambda img: get_otsu_mask(img, scale=scale))
                 .toList(n))

    def process_pair(p):
        p         = ee.Number(p).toInt()
        pre_mask  = ee.Image(mask_list.get(p))
        post_mask = ee.Image(mask_list.get(p.add(1)))
        change    = pre_mask.multiply(2).add(post_mask)
        area_img  = ee.Image.cat([
            change.eq(1).rename("erosion_km2"),
            change.eq(2).rename("accretion_km2"),
        ])
        date_pre  = pre_mask.date().format("YYYY-MM-dd")
        date_post = post_mask.date().format("YYYY-MM-dd")
        return (_reduce_area_km2(area_img, region_fc, scale)
                .map(lambda d: d.set("date_pre", date_pre, "date_post", date_post)))

    all_fcs = ee.List.sequence(0, n - 2).map(process_pair)
    flat_fc  = ee.FeatureCollection(all_fcs).flatten()

    if export:
        task = ee.batch.Export.table.toDrive(
            collection  = flat_fc,
            description = "sylt_change_timeseries",
            fileFormat  = "CSV",
        )
        task.start()
        print(f"Export task started ({task.id})")
        return None

    print("Fetching results")
    features = flat_fc.getInfo()["features"]

    rows = []
    for feat in features:
        p = feat["properties"]
        rows.append({
            "date_pre":      p["date_pre"],
            "date_post":     p["date_post"],
            "region":        p["name"],
            "erosion_km2":   round(p.get("erosion_km2",   0) or 0.0, 4),
            "accretion_km2": round(p.get("accretion_km2", 0) or 0.0, 4),
        })

    df = (pd.DataFrame(rows)
          .assign(
              date_pre  = lambda d: pd.to_datetime(d["date_pre"]),
              date_post = lambda d: pd.to_datetime(d["date_post"]),
          )
          .sort_values(["region", "date_post"])
          .reset_index(drop=True))

    _print_change_summary(df)
    return df


def _print_change_summary(df: pd.DataFrame):
    print("\nChange timeseries summary per region:\n")
    grouped = df.groupby("region")[["erosion_km2", "accretion_km2"]].agg(
        median_erosion   = ("erosion_km2",   "median"),
        max_erosion      = ("erosion_km2",   "max"),
        median_accretion = ("accretion_km2", "median"),
        max_accretion    = ("accretion_km2", "max"),
    )
    display_df = pd.DataFrame({
        "Region":           grouped.index,
        "Median erosion":   grouped["median_erosion"].map("{:.4f}".format),
        "Max erosion":      grouped["max_erosion"].map("{:.4f}".format),
        "Median accretion": grouped["median_accretion"].map("{:.4f}".format),
        "Max accretion":    grouped["max_accretion"].map("{:.4f}".format),
    })
    print(display_df.to_string(index=False))
    ranked = grouped["max_erosion"].sort_values(ascending=False).index.tolist()
    print(f"Peak erosion ranking:\n{' > '.join(ranked)}")


def plot_change_timeseries(df: pd.DataFrame, save: bool = False):
    """Region-wise erosion and accretion lines over time (consecutive pairs).

    Points are at their actual post-acquisition date.
    """
    erosion_color   = f"#{VIS_CHANGE_MAP['palette'][1]}"
    accretion_color = f"#{VIS_CHANGE_MAP['palette'][2]}"

    regions   = sorted(df["region"].unique())
    n_regions = len(regions)

    fig, axes = plt.subplots(n_regions, 1, figsize=(13, 3.5 * n_regions), sharex=True)
    if n_regions == 1:
        axes = [axes]

    for ax, region in zip(axes, regions):
        grp = df[df["region"] == region].sort_values("date_post")
        ax.plot(grp["date_post"], grp["erosion_km2"],
                color=erosion_color,   marker=".", markersize=4, label="Erosion km²")
        ax.plot(grp["date_post"], grp["accretion_km2"],
                color=accretion_color, marker=".", markersize=4, label="Accretion km²")
        ax.set_ylabel("Area (km²)")
        ax.set_title(region)
        ax.legend(fontsize=8)

    axes[-1].set_xlabel("Date (post-acquisition)")
    fig.suptitle("Erosion / Accretion between consecutive SAR acquisitions (near_msl bin)", y=1.01)
    plt.tight_layout()

    if save:
        path = f"{OUTPUT_PLOTS}change_timeseries.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved to {path}")

    plt.show()
