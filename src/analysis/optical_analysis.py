import io
import re

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, ImageDraw, ImageSequence
import ee

from utils.collection_utils import (
    _get_aoi, _get_base_collection, _mosaic_by_day, _attach_tidal,
    get_collection_s2,
)
from utils.tidal_utils import filter_bin
from utils.config import (
    START_DATE, END_DATE,
    S2_COLLECTION,
    OUTPUT_ANIMATIONS, OUTPUT_PLOTS,
    VIS_S2_TRUE_COLOR, VIS_S2_NDWI,
    REGIONS_OF_INTEREST, WEST_COAST_AGGREGATE,
    GEO_JSON_SYLT_COMPLETE,
    MONTH_NAMES,
)


# ------------------------------------------------------------ #
#  1. Technical Helper Methods  (mirror sar_core.py)
# ------------------------------------------------------------ #

def _open_image_thumbnail(img: ee.Image, aoi: ee.Geometry, viz: dict) -> Image.Image:
    """Download a GEE thumbnail as a PIL Image"""
    url  = img.getThumbURL({**viz, "region": aoi, "dimensions": 512})
    resp = requests.get(url)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content))


def _img_label(img: ee.Image) -> str:
    """Readable date + tidal info for plot titles"""
    img_dict = img.toDictionary(["system:time_start", "tidal_height_m", "tidal_bin"]).getInfo()
    datetime = pd.to_datetime(img_dict["system:time_start"], unit="ms", utc=True)
    return (f"{datetime.strftime('%Y-%m-%d  %H:%M UTC')}\n"
            f"{img_dict.get('tidal_bin', 'n/a')}  ({img_dict.get('tidal_height_m', 0):.2f} m)")


def _get_image_for_date(col: ee.ImageCollection, target_date: str) -> ee.Image:
    """
    Return the image from the collection whose acquisition date matches target_date
    Looks in a 1-day window; raises a clear error when nothing is found
    """
    day_start = ee.Date(target_date)
    day_end   = day_start.advance(1, "day")
    subset    = col.filterDate(day_start, day_end)

    count = subset.size().getInfo()
    if count == 0:
        available = col.aggregate_array("system:time_start").getInfo()
        dates_str = ", ".join(
            pd.to_datetime(t, unit="ms", utc=True).strftime("%Y-%m-%d")
            for t in sorted(available)
        )
        raise ValueError(f"No S2 image found for {target_date}. Available dates in this collection:\n{dates_str}")

    return subset.sort("system:time_start").first()


def _compute_ndwi(img: ee.Image) -> ee.Image:
    """NDWI = (Green - NIR) / (Green + NIR)  -> positive values = water"""
    return ee.Image(
        img.normalizedDifference(["B3", "B8"])
           .rename("NDWI")
           .copyProperties(img, img.propertyNames())
    )


# ------------------------------------------------------------ #
#  2. Single-Image Display
# ------------------------------------------------------------ #

def plot_single_image_s2(col: ee.ImageCollection, target_date:str,
                          title: str="Sentinel-2",ndwi:bool=False,save:bool=False):
    """
    Display one S2 image from the collection for target_date
    """
    aoi = _get_aoi()
    img = _get_image_for_date(col, target_date)

    if ndwi:
        plot_img = _compute_ndwi(img)
        viz = VIS_S2_NDWI
        mode_tag = "NDWI"
    else:
        # True colour: B4=Red, B3=Green, B2=Blue
        plot_img = img.select(["B4", "B3", "B2"])
        viz = VIS_S2_TRUE_COLOR
        mode_tag = "True colour"

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(_open_image_thumbnail(plot_img, aoi, viz))

    try:
        subtitle = _img_label(img)
        ax.set_title(f"{title} — {mode_tag}\n{subtitle}", fontsize=11)
    except Exception:
        ax.set_title(f"{title} — {mode_tag}", fontsize=11)

    ax.axis("off")
    plt.tight_layout()

    if save:
        full_path = f"{OUTPUT_PLOTS}{title}_{target_date}.png"
        plt.savefig(full_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {full_path}")

    plt.show()


# ------------------------------------------------------------ #
#  3. Time-Series GIF
# ------------------------------------------------------------ #

def generate_s2_timeseries_gif(col:  ee.ImageCollection,ndwi: bool = False, fps:  int  = 1, width: int = 600):
    aoi = _get_aoi()

    times = col.aggregate_array("system:time_start").getInfo()
    dates = [pd.to_datetime(t, unit="ms", utc=True).strftime("%Y-%m-%d") for t in times]

    if ndwi:
        viz = VIS_S2_NDWI
        mode_tag = "ndwi"
        def prep_for_gif(img):
            return _compute_ndwi(img).visualize(**viz)
    else:
        viz = VIS_S2_TRUE_COLOR
        mode_tag = "true_color"
        def prep_for_gif(img):
            return img.select(["B4", "B3", "B2"]).visualize(**viz)

    col_prepared = col.map(prep_for_gif)

    print("Rendering and downloading raw GIF from GEE…")
    gif_url = col_prepared.getVideoThumbURL({
        "dimensions": width,
        "region": aoi,
        "framesPerSecond": fps,
        "crs": "EPSG:3857",
    })
    response = requests.get(gif_url)
    response.raise_for_status()

    raw_gif = Image.open(io.BytesIO(response.content))
    frames  = []

    for i, frame in enumerate(ImageSequence.Iterator(raw_gif)):
        frame     = frame.convert("RGBA")
        draw      = ImageDraw.Draw(frame)
        date_text = dates[i] if i < len(dates) else "Unknown"

        # White text with black outline for readability on any background
        x, y = 15, 15
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            draw.text((x + dx, y + dy), date_text, fill="black")
        draw.text((x, y), date_text, fill="white")

        frames.append(frame)

    output_path = f"{OUTPUT_ANIMATIONS}timeseries_s2_{mode_tag}.gif"
    print(f"Saving GIF ({len(frames)} frames) to {output_path}")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=int(1000 / fps),
    )
    print("Done")


# ------------------------------------------------------------ #
#  4. S2 Availability Assessment 
# ------------------------------------------------------------ #

def assess_s2_availability(cloud_thresholds: tuple[int, int]=(20, 40), tidal_bin: str = "near_msl") -> None:
    """
    Monthly availability matrix for Sentinel-2 over the Sylt AOI

    """
    aoi = _get_aoi()
    thr_lo, thr_hi = cloud_thresholds
    base = _get_base_collection(S2_COLLECTION, aoi, START_DATE, END_DATE)

    def day_month_counts(col):
        times = col.aggregate_array("system:time_start").getInfo()
        if not times:
            return pd.Series(dtype=int)
        dates = pd.to_datetime(times, unit="ms", utc=True)
        unique_days = pd.to_datetime(dates.normalize().unique())
        return pd.Series(pd.DatetimeIndex(unique_days).month).value_counts()

    def tidal_month_counts(pre_mosaic_col):
        col = filter_bin(_attach_tidal(_mosaic_by_day(pre_mosaic_col)), tidal_bin)
        times = col.aggregate_array("system:time_start").getInfo()
        if not times:
            return pd.Series(dtype=int)
        dates = pd.to_datetime(times, unit="ms", utc=True)
        return pd.Series(dates.month).value_counts()

    col_lo = base.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", thr_lo))
    col_hi = base.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", thr_hi))

    print("Fetching S2 scene counts…")
    step_counts = {
        "All":                              day_month_counts(base),
        f"Cloud<{thr_lo}%":                day_month_counts(col_lo),
        f"Cloud<{thr_hi}%":                day_month_counts(col_hi),
        f"Cloud<{thr_lo}%+{tidal_bin}":    tidal_month_counts(col_lo),
        f"Cloud<{thr_hi}%+{tidal_bin}":    tidal_month_counts(col_hi),
    }

    idx = range(1, 13)
    df = pd.DataFrame(index=idx)
    for name, counts in step_counts.items():
        df[name] = counts.reindex(idx, fill_value=0).astype(int)
    df.loc["Total"] = df.sum()
    df.index = MONTH_NAMES + ["Total"]

    print(f"\n{'=' * 80}")
    print(f"S2 Availability  ({START_DATE[:4]}–{END_DATE[:4]}, Sylt AOI, all orbits)")
    print()
    print(df.to_string())
    print("=" * 80)


# ------------------------------------------------------------ #
#  5. Best-Scene Presentation Visual
# ------------------------------------------------------------ #

def _plot_s2_region_boxes(thumb: Image.Image, scene_date: str, save: bool = False) -> None:
    """True-colour thumbnail with axis-aligned bounding boxes for each ROI and the aggregate."""
    aoi_coords = GEO_JSON_SYLT_COMPLETE["features"][0]["geometry"]["coordinates"][0]
    lon_min = min(c[0] for c in aoi_coords)
    lon_max = max(c[0] for c in aoi_coords)
    lat_max = max(c[1] for c in aoi_coords)
    lat_min = min(c[1] for c in aoi_coords)

    w, h = thumb.size

    def geo_to_pix(lon, lat):
        px = (lon - lon_min) / (lon_max - lon_min) * w
        py = (lat_max - lat) / (lat_max - lat_min) * h
        return px, py

    def make_patch(geojson, color, label):
        coords = geojson["features"][0]["geometry"]["coordinates"][0]
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        x0, y0 = geo_to_pix(min(lons), max(lats))   # top-left in pixel space
        x1, y1 = geo_to_pix(max(lons), min(lats))   # bottom-right in pixel space
        return mpatches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            linewidth=2.5, edgecolor=color, facecolor="none", label=label, zorder=5,
        )

    region_configs = [
        (REGIONS_OF_INTEREST["list_ellenbogen"],    "#e6194b", "List / Ellenbogen"),
        (REGIONS_OF_INTEREST["rotes_kliff_kampen"], "#f58231", "Rotes Kliff / Kampen"),
        (REGIONS_OF_INTEREST["hoernum_odde"],       "#4363d8", "Hoernum Odde"),
        (WEST_COAST_AGGREGATE,                      "#3cb44b", "West coast (aggregate)"),
    ]

    fig_h = 7.0
    fig_w = fig_h * w / h
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(thumb)

    for geojson, color, label in region_configs:
        ax.add_patch(make_patch(geojson, color, label))

    ax.legend(loc="lower left", fontsize=8, framealpha=0.85)
    ax.set_title(f"Sentinel-2 {scene_date} — Regions of Interest", fontsize=10)
    ax.axis("off")
    plt.tight_layout()

    if save:
        safe = re.sub(r"[^\w\-]", "_", scene_date)
        path = f"{OUTPUT_PLOTS}s2_best_scene_{safe}_region_boxes.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved to {path}")

    plt.show()


def plot_best_s2_scene(year:int|None=None, date:str|None=None, save:bool=False, region_boxes:bool=False) -> None:
    """
    Selects the cleanest (=lowest CLOUDY_PIXEL_PERCENTAGE), near-MSL summer Sentinel-2 scene over the Sylt AOI  true-colour / NDWI image
    """
    aoi = _get_aoi()
    tidal_bin = "near_msl"

    print("\n--- Building S2 summer near-MSL collection ---")
    col = get_collection_s2()         # summer months, cloud < 20 %
    col = filter_bin(col, tidal_bin)  # near-MSL tidal bin
    if col.size().getInfo() == 0:
        print("No scenes found after tidal filtering.")
        return

    # ── Scene selection ──────────────────────────────────────────────────────
    if date is not None:
        img = _get_image_for_date(col, date)
    elif year is not None:
        year_col = col.filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        if year_col.size().getInfo() == 0:
            raise ValueError(f"No near-MSL summer S2 scene found for year {year}.")
        img = year_col.sort("CLOUDY_PIXEL_PERCENTAGE").first()
    else:
        img = col.sort("CLOUDY_PIXEL_PERCENTAGE").first()

    # ── Print reproducibility info ───────────────────────────────────────────
    info = img.toDictionary(
        ["system:time_start", "CLOUDY_PIXEL_PERCENTAGE", "tidal_bin", "tidal_height_m"]
    ).getInfo()
    scene_date = pd.to_datetime(info["system:time_start"], unit="ms", utc=True).strftime("%Y-%m-%d")
    cloud_pct  = info.get("CLOUDY_PIXEL_PERCENTAGE", float("nan"))
    tbin_label = info.get("tidal_bin",       tidal_bin)
    tide_m     = info.get("tidal_height_m",  float("nan"))
    print(f"Selected scene: {scene_date}  |  cloud={cloud_pct:.1f}%  |  tide={tide_m:+.2f} m  ({tbin_label})")

    # Download thumbnails
    true_colour = img.select(["B4", "B3", "B2"])
    ndwi_img = _compute_ndwi(img)
    tc_thumb = _open_image_thumbnail(true_colour, aoi, VIS_S2_TRUE_COLOR)
    ndwi_thumb = _open_image_thumbnail(ndwi_img, aoi, VIS_S2_NDWI)

    # Plot
    title_str = (f"Sentinel-2  {scene_date}  |  {tbin_label} ({tide_m:+.2f} m)  |  cloud {cloud_pct:.1f}%")
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(title_str, fontsize=11)

    axes[0].imshow(tc_thumb)
    axes[0].set_title("True colour  (B4 * B3 * B2)", fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(ndwi_thumb)
    axes[1].set_title("NDWI  (Green − NIR) / (Green + NIR)", fontsize=9)
    axes[1].axis("off")

    plt.tight_layout()

    if save:
        safe = re.sub(r"[^\w\-]", "_", scene_date)
        path = f"{OUTPUT_PLOTS}s2_best_scene_{safe}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved to {path}")

    plt.show()

    if region_boxes:
        _plot_s2_region_boxes(tc_thumb, scene_date, save=save)