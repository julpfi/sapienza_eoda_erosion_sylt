import io
import os
import requests
import re
import numpy as np
import ee
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from PIL import Image, ImageDraw, ImageSequence

from utils.collection_utils import _get_aoi, get_collection_s1
from utils.config import (
    QUANTIFICATION_SCALE, REGIONS_OF_INTEREST, WEST_COAST_AGGREGATE, SPIT_REGIONS,
    OUTPUT_PLOTS, OUTPUT_ANIMATIONS, OUTPUT_DATA,
    VIS_CHANGE_MAP, VIS_BINARY_WATER_MASK, VIS_SAR_VV,
)
from utils.tidal_utils import filter_bin
from analysis.sar_core import get_otsu_mask

STORM_MONTHS = frozenset({10, 11, 12, 1, 2, 3})
CALM_MONTHS  = frozenset({4, 5, 6, 7, 8, 9})
_MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]


# --------------------------------------------------------
#  Region-of-interest configuration
# --------------------------------------------------------

def build_region_fc() -> ee.FeatureCollection:
    """
    Assemble the analysis FeatureCollection from config
    Regions are clarified under REGIONS_OF_INTEREST + whole WEST_COAST_AGGREGATE
    """
    features = []

    for name, geojson in REGIONS_OF_INTEREST.items():
        if geojson is None:
            print(f"{name}: geometry not set -> skipped")
            continue
        geom = ee.Geometry(geojson["features"][0]["geometry"])
        features.append(ee.Feature(geom, {"name": name, "is_spit": name in SPIT_REGIONS}))

    if WEST_COAST_AGGREGATE is not None:
        agg_geom = ee.Geometry(REGIONS_OF_INTEREST["features"][0]["geometry"])


    features.append(ee.Feature(agg_geom, {"name": "island_aggregate", "is_spit": False}))
    return ee.FeatureCollection(features)


def _reduce_area_km2(image: ee.Image, region_fc: ee.FeatureCollection, scale: int) -> ee.FeatureCollection:
    """Sum pixel area (km2) per band per region in one reduceRegions call."""
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
                                fps: int = 2, width: int = 600, max_frames: int = 40):
    """
    Render a GIF of the SAR timeseries (raw VV or water mask)
    If the collection has more than max_frames images -> even subsample so the GIF stays clean
    """
    aoi = _get_aoi()

    # Pre-apply Otsu before the video pipeline — nesting it inside the render
    # map causes a 400 from getVideoThumbURL (computation graph too complex).
    if mask:
        col = col.map(get_otsu_mask)

    times = sorted(col.aggregate_array("system:time_start").getInfo())
    dates = [pd.to_datetime(t, unit="ms", utc=True).strftime("%Y-%m-%d") for t in times]

    # Subsample if collection is larger than max_frames
    n_total = len(times)
    if n_total > max_frames:
        keep_idx = np.round(np.linspace(0, n_total - 1, max_frames)).astype(int).tolist()
        img_list  = col.sort("system:time_start").toList(n_total)
        col = ee.ImageCollection(ee.List(keep_idx).map(lambda i: img_list.get(i)))
        dates = [dates[i] for i in keep_idx]
        print(f"Subsampled GIF: {n_total} -> {max_frames} frames")

    def prep_for_gif(img):
        if mask:
            return img.select("water").visualize(**VIS_BINARY_WATER_MASK)
        vv_db = img.select("VV").log10().multiply(10)
        return vv_db.visualize(**VIS_SAR_VV)

    col_prepared = col.map(prep_for_gif)

    print("Rendering and downloading raw GIF...")
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
        frame = frame.convert("RGBA")
        draw = ImageDraw.Draw(frame)
        date_text = dates[i] if i < len(dates) else "Unknown"
        x, y = 15, 15
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            draw.text((x + dx, y + dy), date_text, fill="black")
        draw.text((x, y), date_text, fill="white")
        frames.append(frame)

    mode_tag = "mask" if mask else "vv"
    output_path = f"{OUTPUT_ANIMATIONS}timeseries_sar_{mode_tag}.gif"

    print(f"Saving GIF ({len(frames)} frames) to {output_path}...")
    frames[0].save(output_path, save_all=True, append_images=frames[1:], loop=0, duration=int(1000/fps))
    print("Finished")



# --------------------------------------------------------
#  Entry point 1 — land/water area timeseries
# --------------------------------------------------------

def quantify_timeseries(col: ee.ImageCollection=None, scale: int=QUANTIFICATION_SCALE,
                        export:bool=False, cache_csv:str|None=None, batch_size:int=5) -> pd.DataFrame|None:
    """
    Return long-format land/water area per region per date
    Maps get_otsu_mask over the tidally-binned collection
    Implements caching for computations (stored locally)
    Results are fetched in batches of batch_size images to avoid the GEE limit
    """

    if cache_csv and os.path.exists(cache_csv):
        print(f"Loading land-area timeseries from cache: {cache_csv}")
        df = (pd.read_csv(cache_csv).assign(date=lambda d: pd.to_datetime(d["date"])))
        return df

    # Cache-Miss -> Building timeseries
    if col is None:
        col = get_collection_s1()
        col = filter_bin(col, "near_msl")

    col_size  = col.size().getInfo()
    print(f"\nQuantify_timeseries: {col_size} images (batch_size={batch_size})")

    region_fc = build_region_fc()
    img_list = col.sort("system:time_start").toList(col_size)

    def process_one(raw_img):
        img = ee.Image(raw_img)
        mask = get_otsu_mask(img, scale=scale)
        water = mask.select("water").toFloat().rename("water_km2")
        land = mask.select("water").Not().toFloat().rename("land_km2")
        date = img.date().format("YYYY-MM-dd")
        return (_reduce_area_km2(ee.Image.cat([land, water]), region_fc, scale)
                .map(lambda f: f.set("date", date)))

    if export:
        all_fcs = img_list.map(process_one)
        flat_fc = ee.FeatureCollection(all_fcs).flatten()
        task = ee.batch.Export.table.toDrive(collection=flat_fc, description="sylt_land_area", fileFormat="CSV")
        task.start()
        print(f"Export task started ({task.id})")
        return None

    rows = []
    n_batches = (col_size + batch_size - 1) // batch_size
    for b in range(n_batches):
        start = b * batch_size
        end = min(start + batch_size, col_size)
        print(f"  Batch {b + 1}/{n_batches}: images {start}–{end - 1}")

        batch_list = img_list.slice(start, end)
        batch_fc = ee.FeatureCollection(batch_list.map(process_one)).flatten()
        for feat in batch_fc.getInfo()["features"]:
            p = feat["properties"]
            land = p.get("land_km2",  0) or 0.0
            water = p.get("water_km2", 0) or 0.0
            total = land + water or 1e-9
            rows.append({
                "date": p["date"],
                "region": p["name"],
                "land_km2": round(land, 4),
                "water_km2": round(water, 4),
                "land_fraction": round(land/total, 5)
            })

    df = (pd.DataFrame(rows)
          .assign(date=lambda d: pd.to_datetime(d["date"]))
          .sort_values(["region", "date"])
          .reset_index(drop=True))

    if cache_csv:
        df.to_csv(cache_csv, index=False)
        print(f"Land-area timeseries cached to {cache_csv}")

    return df


def filter_outlier_dates(df:pd.DataFrame, k:float=2.0) -> pd.DataFrame:
    """
    Drop dates where island_aggregate land area > Q3 + k*IQR 
        Oonly upper bound as storms increas misclasified water to land (due to higher backscatter)
    Filters at the date level using island_aggregate as anchor 
    All regions from a flagged date are dropped together
    """
    agg = df[df["region"] == "island_aggregate"][["date", "land_km2"]].copy()
    if agg.empty:
        print("filter_outlier_dates: 'island_aggregate' not found, skipping.")
        return df

    q1, q3 = agg["land_km2"].quantile(0.25), agg["land_km2"].quantile(0.75)
    iqr = q3 - q1
    upper = q3 + k * iqr

    bad_dates = agg.loc[agg["land_km2"] > upper, "date"]
    if bad_dates.empty:
        print(f"filter_outlier_dates: no outliers found (upper fence = {upper:.3f} km2).")
        return df

    print(f"\nOutlier filter (island_aggregate upper fence = Q3 + {k}×IQR = {upper:.3f} km2):")
    for d in sorted(bad_dates):
        val = agg.loc[agg["date"] == d, "land_km2"].values[0]
        print(f"  dropping {d.strftime('%Y-%m-%d')}  island_aggregate={val:.3f} km2")

    cleaned = df[~df["date"].isin(bad_dates)].reset_index(drop=True)
    print(f"\t{len(bad_dates)} date(s) removed, {cleaned['date'].nunique()} remaining.\n")
    return cleaned


def filter_outlier_dates_change(change_df: pd.DataFrame, bad_dates) -> pd.DataFrame:
    """Drop change pairs where either endpoint falls on a flagged date"""
    mask = (change_df["date_pre"].isin(bad_dates) |
               change_df["date_post"].isin(bad_dates))
    cleaned = change_df[~mask].reset_index(drop=True)
    if mask.any():
        print(f"filter_outlier_dates_change: dropped {mask.sum()} pair(s) touching outlier dates.")
    return cleaned


def print_timeseries_summary(df: pd.DataFrame):
    print("\nLand-area summary per region (near_msl bin):\n")
    amplitudes = {}
    for region, grp in df.groupby("region"):
        grp = grp.sort_values("date")
        land = grp["land_km2"]
        n = len(land)

        min_val = land.min()
        max_val = land.max()
        amp = max_val - min_val
        date_min = grp.loc[land.idxmin(), "date"].strftime("%Y-%m-%d")
        date_max = grp.loc[land.idxmax(), "date"].strftime("%Y-%m-%d")
        amplitudes[region] = amp

        # Linear trend: decimal year → land_km2
        dec_yr = grp["date"].apply(lambda d: d.year + (d.dayofyear - 1) / 365.25)
        res = stats.linregress(dec_yr, land)
        sig = "SIGNIFICANT" if res.pvalue <= 0.05 else "no sig. trend"
        trend  = (f"{res.slope:+.4f} ± {2*res.stderr:.4f} km2/yr  "
                  f"r2={res.rvalue**2:.3f}  p={res.pvalue:.3f}  [{sig}]")

        # Seasonal split
        month = grp["date"].dt.month
        storm_mean = land[month.isin(STORM_MONTHS)].mean()
        calm_mean = land[month.isin(CALM_MONTHS)].mean()
        seas_diff = calm_mean - storm_mean

        print(f"  {region}  (n={n})")
        print(f"    mean={land.mean():.3f} km2")
        print(f"    min={min_val:.3f} km2 ({date_min})  "
              f"max={max_val:.3f} km2 ({date_max})  amplitude={amp:.3f} km2")
        print(f"    trend: {trend}")
        print(f"    storm Oct–Mar: {storm_mean:.3f} km2  |  "
              f"calm Apr–Sep: {calm_mean:.3f} km2  |  seasonal diff={seas_diff:+.3f} km2")
        print()

    ranked = sorted(amplitudes, key=amplitudes.__getitem__, reverse=True)
    print(f"  Amplitude ranking: {' > '.join(ranked)}")
    print("  [OLS trend; seasonal autocorrelation makes SE anti-conservative → p > 0.05 is a robust null]")


def plot_timeseries(df: pd.DataFrame, save: bool = False):
    """Line plot of land area per region over time."""
    fig, ax = plt.subplots(figsize=(13, 5))

    for region, grp in df.groupby("region"):
        grp = grp.sort_values("date")
        ax.plot(grp["date"], grp["land_km2"], marker=".", markersize=3, label=region)

    ax.set_xlabel("Date")
    ax.set_ylabel("Land area (km2)")
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
                               export: bool = False,
                               cache_csv: str | None = None,
                               batch_size: int = 5) -> pd.DataFrame | None:
    """
    Return region-wise erosion/accretion for every consecutive image pair
    Implements caching for computations (stored locally)
    Pairs are fetched in batches of batch_size to avoid the GEE quota limits
    """
    if cache_csv and os.path.exists(cache_csv):
        print(f"Loading change timeseries from cache: {cache_csv}")
        df = (pd.read_csv(cache_csv)
              .assign(
                  date_pre = lambda d: pd.to_datetime(d["date_pre"]),
                  date_post = lambda d: pd.to_datetime(d["date_post"]),
              ))
        return df

    # Cache-Miss -> Building change timeseries
    n = col.size().getInfo()
    print(f"\nQuantify change timeseries: {n} images -> {n - 1} pairs (batch_size={batch_size})")

    region_fc = build_region_fc()
    mask_list = (col.sort("system:time_start")
                 .map(lambda img: get_otsu_mask(img, scale=scale))
                 .toList(n))

    def process_pair(p):
        p = ee.Number(p).toInt()
        pre_mask = ee.Image(mask_list.get(p))
        post_mask = ee.Image(mask_list.get(p.add(1)))
        change = pre_mask.multiply(2).add(post_mask)
        area_img = ee.Image.cat([
            change.eq(1).rename("erosion_km2"),
            change.eq(2).rename("accretion_km2"),
        ])
        date_pre = pre_mask.date().format("YYYY-MM-dd")
        date_post = post_mask.date().format("YYYY-MM-dd")
        return (_reduce_area_km2(area_img, region_fc, scale)
                .map(lambda d: d.set("date_pre", date_pre, "date_post", date_post)))

    n_pairs = n-1
    all_pairs = ee.List.sequence(0, n_pairs - 1)

    if export:
        flat_fc = ee.FeatureCollection(all_pairs.map(process_pair)).flatten()
        task = ee.batch.Export.table.toDrive(collection  = flat_fc, description = "sylt_change_timeseries", fileFormat  = "CSV")
        task.start()
        print(f"Export task started ({task.id})")
        return None

    rows = []
    n_batches = (n_pairs + batch_size-1) // batch_size
    for b in range(n_batches):
        start = b * batch_size
        end = min(start + batch_size, n_pairs)
        print(f"  Batch {b + 1}/{n_batches}: pairs {start}–{end - 1}")
        batch_pairs = all_pairs.slice(start, end)
        batch_fc    = ee.FeatureCollection(batch_pairs.map(process_pair)).flatten()
        for feat in batch_fc.getInfo()["features"]:
            p = feat["properties"]
            rows.append({
                "date_pre": p["date_pre"],
                "date_post": p["date_post"],
                "region": p["name"],
                "erosion_km2": round(p.get("erosion_km2",   0) or 0.0, 4),
                "accretion_km2": round(p.get("accretion_km2", 0) or 0.0, 4),
            })

    df = (pd.DataFrame(rows)
          .assign(
              date_pre  = lambda d: pd.to_datetime(d["date_pre"]),
              date_post = lambda d: pd.to_datetime(d["date_post"]),
          )
          .sort_values(["region", "date_post"])
          .reset_index(drop=True))

    if cache_csv:
        df.to_csv(cache_csv, index=False)
        print(f"Change timeseries cached to {cache_csv}")

    return df



def print_change_summary(df: pd.DataFrame):
    print("\nChange timeseries summary per region (near_msl bin):\n")
    peak_erosions = {}
    for region, grp in df.groupby("region"):
        is_spit = region in SPIT_REGIONS
        n = len(grp)
        total_erosion = grp["erosion_km2"].sum()
        total_accretion = grp["accretion_km2"].sum()
        net_flux = total_accretion - total_erosion

        peak_idx = grp["erosion_km2"].idxmax()
        peak_val = grp.loc[peak_idx, "erosion_km2"]
        peak_pre = grp.loc[peak_idx, "date_pre"].strftime("%Y-%m-%d")
        peak_post = grp.loc[peak_idx, "date_post"].strftime("%Y-%m-%d")
        peak_erosions[region] = peak_val

        post_month = grp["date_post"].dt.month
        storm_erosion = grp.loc[post_month.isin(STORM_MONTHS), "erosion_km2"].sum()
        calm_erosion = grp.loc[post_month.isin(CALM_MONTHS),  "erosion_km2"].sum()

        print(f"\t{region}  (n={n} intervals)")
        print(f"\total erosion={total_erosion:.4f} km2  |  total accretion={total_accretion:.4f} km2")
        if is_spit:
            print(f"\tnet flux not reported (spit: erosion/accretion on separate flanks)")
        else:
            print(f"\tnet flux={net_flux:+.4f} km2 (accretion − erosion)")
        print(f"\t peak erosion: {peak_val:.4f} km2  ({peak_pre} → {peak_post})")
        print(f"\tstorm Oct–Mar erosion={storm_erosion:.4f} km2 | calm Apr–Sep erosion={calm_erosion:.4f} km2\n")

    ranked = sorted(peak_erosions, key=peak_erosions.__getitem__, reverse=True)
    print(f"\tPeak erosion ranking: {' > '.join(ranked)}")



def plot_change_timeseries(df: pd.DataFrame, save: bool = False):
    """
    Region-wise erosion and accretion lines over time (->consecutive pairs)
    Points are at their actual post-acquisition date
    """
    erosion_color = f"#{VIS_CHANGE_MAP['palette'][1]}"
    accretion_color = f"#{VIS_CHANGE_MAP['palette'][2]}"

    regions   = sorted(df["region"].unique())
    n_regions = len(regions)

    fig, axes = plt.subplots(n_regions, 1, figsize=(13, 3.5 * n_regions), sharex=True)
    if n_regions == 1:
        axes = [axes]

    for ax, region in zip(axes, regions):
        grp = df[df["region"] == region].sort_values("date_post")
        ax.plot(grp["date_post"], grp["erosion_km2"],
                color=erosion_color,   marker=".", markersize=4, label="Erosion km2")
        ax.plot(grp["date_post"], grp["accretion_km2"],
                color=accretion_color, marker=".", markersize=4, label="Accretion km2")
        ax.set_ylabel("Area (km2)")
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


# --------------------------------------------------------
#  Seasonal / climatological plots
# --------------------------------------------------------

def plot_monthly_land_area_cycle(df: pd.DataFrame, save: bool = False):
    """
    Vertically stacked monthly climatology: 
        top = anomaly and  bottom = per-region mean 
    Computation uses year × month means first so each year contributes equally regardless ofnumber of images in that month 
    """

    df2 = df.copy()
    df2["year"] = df2["date"].dt.year
    df2["month"] = df2["date"].dt.month

    # Step 1: year x month cell mean
    ym_mean = (df2.groupby(["region", "year", "month"])["land_km2"]
               .mean().reset_index())

    # Step 2: climatology (mean and inter-annual std) per region x month
    clim = (ym_mean
            .groupby(["region", "month"])["land_km2"]
            .agg(["mean", "std"]).reset_index()
            .rename(columns={"mean": "clim_mean", "std": "clim_std"}))

    # Step 3: region overall mean
    region_mean = ym_mean.groupby("region")["land_km2"].mean()

    regions = sorted(df["region"].unique())
    n_regions = len(regions)
    n_cols = min(3, n_regions)
    n_rows = (n_regions + n_cols - 1) // n_cols
    colors= plt.cm.tab10(np.linspace(0, 1, n_regions))

    fig = plt.figure(figsize=(14, 4 + 3.5 * n_rows))
    gs = gridspec.GridSpec(1 + n_rows, n_cols, figure=fig,
                            hspace=0.55, wspace=0.35)


    # Top panel: anomaly 
    ax_top = fig.add_subplot(gs[0, :])
    months = list(range(1, 13))

    for color, region in zip(colors, regions):
        r      = clim[clim["region"] == region].sort_values("month")
        anom   = r["clim_mean"] - region_mean[region]
        std    = r["clim_std"].fillna(0)
        ax_top.plot(r["month"], anom, marker="o", markersize=5,
                    label=region, color=color)
        ax_top.fill_between(r["month"], anom - std, anom + std,
                            alpha=0.15, color=color)

    ax_top.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_top.set_xticks(months)
    ax_top.set_xticklabels(_MONTH_NAMES)
    ax_top.set_ylabel("Land-area anomaly (km2)")
    ax_top.set_title("Monthly land-area anomaly 2017–2024 (near_msl bin) — shading ±1σ inter-annual")
    ax_top.legend(fontsize=8, ncol=min(4, n_regions))


    # Bottom panels: raw mean per region 
    for idx, (color, region) in enumerate(zip(colors, regions)):
        row = 1 + idx // n_cols
        col = idx % n_cols
        ax  = fig.add_subplot(gs[row, col])
        r   = clim[clim["region"] == region].sort_values("month")
        std = r["clim_std"].fillna(0)
        ax.plot(r["month"], r["clim_mean"], marker="o", markersize=4, color=color)
        ax.fill_between(r["month"],
                        r["clim_mean"] - std, r["clim_mean"] + std,
                        alpha=0.2, color=color)
        ax.set_xticks(months)
        ax.set_xticklabels(_MONTH_NAMES, fontsize=7, rotation=45, ha="right")
        ax.set_title(region, fontsize=9)
        ax.set_ylabel("Land km2", fontsize=8)


    if save:
        path = f"{OUTPUT_PLOTS}monthly_land_area_cycle.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved to {path}")

    plt.show()


def plot_erosion_by_month(change_df: pd.DataFrame, save: bool = False):
    """
    Mean erosion and accretion per calendar month 
    Groups by region x year x month and then averages across years
    Subplots per reagions
    """
    df2 = change_df.copy()
    df2["year"] = df2["date_post"].dt.year
    df2["month"] = df2["date_post"].dt.month

    # Year x month cell means, then climatology across years
    ym = (df2
          .groupby(["region", "year", "month"])[["erosion_km2", "accretion_km2"]]
          .mean().reset_index())
    clim = (ym
            .groupby(["region", "month"])[["erosion_km2", "accretion_km2"]]
            .agg(["mean", "std"]).reset_index())
    
    clim.columns = ["region", "month",
                    "erosion_mean", "erosion_std",
                    "accretion_mean", "accretion_std"]

    erosion_color = f"#{VIS_CHANGE_MAP['palette'][1]}"
    accretion_color = f"#{VIS_CHANGE_MAP['palette'][2]}"

    regions = sorted(change_df["region"].unique())
    n_regions = len(regions)
    x = np.arange(12)
    width = 0.4

    fig, axes = plt.subplots(n_regions, 1, figsize=(12, 3.5 * n_regions), sharex=True)
    if n_regions == 1:
        axes = [axes]

    for ax, region in zip(axes, regions):
        r = (clim[clim["region"] == region]
             .set_index("month").reindex(range(1, 13)).reset_index())

        ax.bar(x - width / 2, r["erosion_mean"], width,
               color=erosion_color, alpha=0.85, label="Erosion km2")
        ax.bar(x + width / 2, r["accretion_mean"], width,
               color=accretion_color, alpha=0.85, label="Accretion km2")
        ax.errorbar(x - width / 2, r["erosion_mean"], yerr=r["erosion_std"],
                    fmt="none", color="black", capsize=3, linewidth=0.8)
        ax.errorbar(x + width / 2, r["accretion_mean"], yerr=r["accretion_std"],
                    fmt="none", color="black", capsize=3, linewidth=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(_MONTH_NAMES)
        ax.set_ylabel("Area (km2)")
        ax.set_title(region)
        ax.legend(fontsize=8)

    fig.suptitle("Mean monthly erosion / accretion 2017–2024 (year-equalized, near_msl bin)\n"
                 "Error bars = ±1σ inter-annual", y=1.01)
    plt.tight_layout()

    if save:
        path = f"{OUTPUT_PLOTS}erosion_by_month.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved to {path}")

    plt.show()


def plot_land_area_monthly_means(df: pd.DataFrame, save: bool = False):
    """
    Monthly-averaged land area over the full 2017–2024 record
    One point per calendar-month x year cell (averages over multiple acquisitions in the same month) 
    """

    df2 = df.copy()
    df2["year_month"] = df2["date"].dt.to_period("M")

    monthly = (df2
               .groupby(["region", "year_month"])["land_km2"]
               .mean().reset_index())
    monthly["date"] = monthly["year_month"].dt.to_timestamp()

    regions = sorted(monthly["region"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(regions)))

    fig, ax = plt.subplots(figsize=(14, 5))

    for color, region in zip(colors, regions):
        grp = monthly[monthly["region"] == region].sort_values("date")
        ax.plot(grp["date"], grp["land_km2"],
                marker=".", markersize=5, linewidth=1.2,
                label=region, color=color)

    ax.set_xlabel("Date")
    ax.set_ylabel("Land area (km2)")
    ax.set_title("Monthly mean land area 2017–2024 (near_msl bin)")
    ax.legend(fontsize=8)
    plt.tight_layout()

    if save:
        path = f"{OUTPUT_PLOTS}land_area_monthly_means.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved to {path}")

    plt.show()
