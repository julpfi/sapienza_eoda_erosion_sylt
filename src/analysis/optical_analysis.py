import io
from collections import Counter

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, ImageDraw, ImageSequence
import ee

from utils.collection_utils import (
    _get_aoi, _get_base_collection, _mosaic_by_day, _attach_tidal,
)
from utils.tidal_utils import filter_bin
from utils.config import (
    START_DATE, END_DATE,
    S2_COLLECTION,
    OUTPUT_ANIMATIONS, OUTPUT_PLOTS,
    VIS_S2_TRUE_COLOR, VIS_S2_NDWI,
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

def plot_single_image_s2(
    col:         ee.ImageCollection,
    target_date: str,
    title:       str  = "Sentinel-2",
    ndwi:        bool = False,
    save:        bool = False,
):
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
#  4. S2 Availability Assessment (Funnel Analysis)
# ------------------------------------------------------------ #

_STORM_MONTHS = {10, 11, 12, 1, 2, 3}   # Oct–Mar
_CALM_MONTHS  = {4, 5, 6, 7, 8, 9}       # Apr–Sep


def assess_s2_availability(
    cloud_thresholds: tuple[int, int] = (20, 40),
    tidal_bin: str = "near_msl",
    save: bool = False,
) -> pd.DataFrame:
    """
    Availability statistics for Sentinel-2 over the Sylt AOI (2017–2024)
    Splits:
      1. All S2 images 
      2. After scene-level cloud filter at each threshold
      3. After tidal filter only
      4. After cloud + tidal
    All relative orbits are mixed
    """
    aoi = _get_aoi()
    thr_lo, thr_hi = cloud_thresholds

    # Step 1 All
    print("Building base S2 collection (all orbits, no cloud filter)…")
    base = _get_base_collection(S2_COLLECTION, aoi, START_DATE, END_DATE)
    col_all = _mosaic_by_day(base)
    n_all = col_all.size().getInfo()

    # Step 2 Cloud filter
    print(f"Applying cloud filter < {thr_lo}% …")
    col_cloud_lo = _mosaic_by_day(
        base.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", thr_lo))
        )
    n_cloud_lo = col_cloud_lo.size().getInfo()

    print(f"Applying cloud filter < {thr_hi}% …")
    col_cloud_hi = _mosaic_by_day(
        base.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", thr_hi))
    )
    n_cloud_hi = col_cloud_hi.size().getInfo()

    # Step 3 tidal filter 
    print("Attaching tidal metadata to all scenes…")
    col_all_tidal = _attach_tidal(col_all)
    col_tidal_only = filter_bin(col_all_tidal, tidal_bin)
    n_tidal_only = col_tidal_only.size().getInfo()

    # Step 4: cloud + tidal 
    print(f"Attaching tidal metadata to cloud-filtered (<{thr_lo}%) scenes…")
    col_lo_tidal = _attach_tidal(col_cloud_lo)
    col_both_lo = filter_bin(col_lo_tidal, tidal_bin)
    n_both_lo = col_both_lo.size().getInfo()

    print(f"Attaching tidal metadata to cloud-filtered (<{thr_hi}%) scenes…")
    col_hi_tidal = _attach_tidal(col_cloud_hi)
    col_both_hi = filter_bin(col_hi_tidal, tidal_bin)
    n_both_hi = col_both_hi.size().getInfo()

    # Print table
    W = 56
    print("\n" + "=" * W)
    print("S2 Availability Funnel  (2017-–2024, Sylt AOI, all orbits)\n")
    print(f"{'Step':<42} {'Count':>6}")
    print("-" * W)
    print(f"{'1. All S2 scenes (unique dates)':<42} {n_all:>6}")
    print(f"{'2. Cloud < ' + str(thr_lo) + '%':<42} {n_cloud_lo:>6}")
    print(f"{'   Cloud < ' + str(thr_hi) + '%':<42} {n_cloud_hi:>6}")
    print(f"{'3. Tidal only (' + tidal_bin + ')':<42} {n_tidal_only:>6}")
    print(f"{'4. Cloud < ' + str(thr_lo) + '% + tidal':<42} {n_both_lo:>6}  ← primary")
    print(f"{'   Cloud < ' + str(thr_hi) + '% + tidal':<42} {n_both_hi:>6}")
    print("=" * W)

    # ── Build date DataFrame from primary set ─────────────────
    times = col_both_lo.aggregate_array("system:time_start").getInfo()
    dates = pd.to_datetime(times, unit="ms", utc=True)
    df = pd.DataFrame({
        "date":   dates,
        "year":   dates.year,
        "month":  dates.month,
    })

    df["season"] = df["month"].apply(lambda m: "storm (Oct–Mar)" if m in _STORM_MONTHS else "calm (Apr–Sep)")

    # Per-year / per-season
    by_year = df.groupby("year").size().rename("scenes")
    by_season = df.groupby("season").size().rename("scenes")

    print(f"\nPer-year (cloud < {thr_lo}% + {tidal_bin}):")
    print(by_year.to_string())
    print(f"\nPer-season (cloud < {thr_lo}% + {tidal_bin}):")
    print(by_season.to_string())


    # Orbit comparison
    print("\nQuerying orbit numbers for comparison…")
    orbit_nums = (
        base.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", thr_lo))
            .aggregate_array("SENSING_ORBIT_NUMBER")
            .getInfo()
    )

    if orbit_nums:
        dominant_orbit = Counter(orbit_nums).most_common(1)[0][0]
        col_single = _mosaic_by_day(
            base
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", thr_lo))
            .filter(ee.Filter.eq("SENSING_ORBIT_NUMBER", int(dominant_orbit)))
        )
        col_single_tidal = _attach_tidal(col_single)
        col_single_both  = filter_bin(col_single_tidal, tidal_bin)
        n_single         = col_single_both.size().getInfo()

        gain = n_both_lo - n_single
        print(f"\nOrbit comparison (cloud < {thr_lo}% + {tidal_bin}):")
        print(f"  All orbits mixed:      {n_both_lo:>4}")
        print(f"  Orbit {dominant_orbit} only:          {n_single:>4}")
        print(f"  Gain from mixing:      {gain:>4}  "
              f"({gain / max(n_single, 1) * 100:.0f}% more scenes)")

    # Plot
    _plot_s2_availability(df, thr_lo, save=save)
    return df


def _plot_s2_availability(df: pd.DataFrame, cloud_thr: int, save: bool = False):
    """Monthly bar chart of usable S2 scenes, shaded by storm / calm season."""
    if df.empty:
        print("No usable scenes — nothing to plot.")
        return

    df = df.copy()
    df["year_month"] = df["date"].dt.to_period("M")
    monthly = df.groupby(["year_month", "season"]).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(15, 4))

    storm_col = "#2255aa"
    calm_col  = "#55aa55"
    months_ts = [p.to_timestamp() for p in monthly.index]

    if "storm (Oct–Mar)" in monthly.columns:
        ax.bar(months_ts, monthly["storm (Oct–Mar)"], width=25,
               color=storm_col, alpha=0.85, label="Storm season (Oct–Mar)")
    if "calm (Apr–Sep)" in monthly.columns:
        bottom = monthly.get("storm (Oct–Mar)", 0)
        ax.bar(months_ts, monthly["calm (Apr–Sep)"], width=25,
               bottom=bottom, color=calm_col, alpha=0.85, label="Calm season (Apr–Sep)")

    ax.set_xlabel("Date")
    ax.set_ylabel("Scenes / month")
    ax.set_title(
        f"Sentinel-2 usable scenes over Sylt AOI\n"
        f"cloud < {cloud_thr}%  +  near-MSL tide  |  all relative orbits  |  2017–2024"
    )
    ax.legend(fontsize=9)
    ax.set_xlim(pd.Timestamp("2017-01-01"), pd.Timestamp("2025-01-01"))
    plt.tight_layout()

    if save:
        path = f"{OUTPUT_PLOTS}s2_availability.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved to {path}")

    plt.show()