import io
import requests
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageSequence
import ee

from collection_utils import _get_aoi
from config import (
    START_DATE, END_DATE,
    OUTPUT_ANIMATIONS, OUTPUT_PLOTS,
    VIS_S2_TRUE_COLOR, VIS_S2_NDWI,
)


# ------------------------------------------------------------ #
#  1. Technical Helper Methods  (mirror sar_analysis.py)
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
    Return the image from the collection whose acquisition date matches target_date.
    Looks in a 1-day window; raises a clear error when nothing is found.
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
        raise ValueError(
            f"No S2 image found for {target_date}.\n"
            f"Available dates in this collection:\n{dates_str}"
        )

    return subset.sort("system:time_start").first()


def _compute_ndwi(img: ee.Image) -> ee.Image:
    """NDWI = (Green − NIR) / (Green + NIR)  →  positive values = water"""
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
    Display one S2 image from the collection for target_date.

    Parameters
    ----------
    col         : mosaiced & tidal-tagged S2 collection (from get_collection_s2)
    target_date : "YYYY-MM-DD" — must match a day present in col
    title       : figure suptitle
    ndwi        : if True, show NDWI instead of true colour
    save        : write PNG to OUTPUT_PLOTS
    """
    aoi = _get_aoi()
    img = _get_image_for_date(col, target_date)

    if ndwi:
        plot_img = _compute_ndwi(img)
        viz      = VIS_S2_NDWI
        mode_tag = "NDWI"
    else:
        # True colour: B4=Red, B3=Green, B2=Blue
        plot_img = img.select(["B4", "B3", "B2"])
        viz      = VIS_S2_TRUE_COLOR
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
        viz      = VIS_S2_NDWI
        mode_tag = "ndwi"
        def prep_for_gif(img):
            return _compute_ndwi(img).visualize(**viz)
    else:
        viz      = VIS_S2_TRUE_COLOR
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