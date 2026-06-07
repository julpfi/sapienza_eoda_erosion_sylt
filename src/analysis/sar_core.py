import io
import re
import requests
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import ee

from utils.collection_utils import _get_aoi, _get_calibration_strip
from utils.config import (
    OUTPUT_PLOTS, VIS_BINARY_WATER_MASK, VIS_SAR_VV, OTSU_MIN_WATER_PIXELS,
)


# --------------------------------------------------------
#  Image helpers
# --------------------------------------------------------

def _open_image_thumbnail(img: ee.Image, aoi: ee.Geometry, viz: dict):
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


# --------------------------------------------------------
#  Otsu coastline detection
# --------------------------------------------------------

def _otsu(histogram: ee.Dictionary) -> ee.Number:
    """GEE Otsu threshold"""
    counts = ee.Array(ee.List(histogram.get("histogram")))
    means  = ee.Array(ee.List(histogram.get("bucketMeans")))

    total      = counts.reduce(ee.Reducer.sum(), [0]).get([0])
    sum_vals   = means.multiply(counts).reduce(ee.Reducer.sum(), [0]).get([0])
    total_mean = sum_vals.divide(total)

    size    = means.length().get([0])
    indices = ee.List.sequence(1, size.subtract(1))

    def between_class_variance(i):
        i        = ee.Number(i).toInt()
        a_counts = counts.slice(0, 0, i)
        a_count  = a_counts.reduce(ee.Reducer.sum(), [0]).get([0])
        a_means  = means.slice(0, 0, i)
        a_mean   = ee.Algorithms.If(
            a_count.gt(0),
            a_means.multiply(a_counts).reduce(ee.Reducer.sum(), [0]).get([0]).divide(a_count),
            0,
        )
        b_count = total.subtract(a_count)
        b_mean  = ee.Algorithms.If(
            b_count.gt(0),
            sum_vals.subtract(ee.Number(a_count).multiply(ee.Number(a_mean))).divide(b_count),
            0,
        )
        return (ee.Number(a_count).multiply(ee.Number(a_mean).subtract(total_mean).pow(2))
                .add(ee.Number(b_count).multiply(ee.Number(b_mean).subtract(total_mean).pow(2))))

    bss       = ee.Array(indices.map(between_class_variance))
    means_cut = means.slice(0, 1)
    return means_cut.sort(bss).get([-1])


def get_otsu_mask(img: ee.Image, scale: int = 40, redefined: bool = True,
                  calibration_geom: ee.Geometry = None,
                  connected_components: bool = True) -> ee.Image:
    """Executes Otsu on Google Earth Engine.

    calibration_geom:      geometry used for histogram sampling; defaults to _get_calibration_strip().
    connected_components:  if True, removes isolated water pockets via connectedPixelCount.
    """
    comp = (img.select("VV").multiply(img.select("VH")).log10().multiply(10)
              .rename("vv_vh_composite"))

    if redefined:
        comp = comp.focal_median(radius=3, kernelType="circle", units="pixels")

    geom = calibration_geom if calibration_geom is not None else _get_calibration_strip()

    hist = comp.reduceRegion(
        reducer    = ee.Reducer.histogram(maxBuckets=256),
        geometry   = geom,
        scale      = scale,
        bestEffort = True,
    )
    threshold = _otsu(ee.Dictionary(hist.get("vv_vh_composite")))

    mask = comp.lt(threshold).rename("water")

    if redefined:
        mask = mask.focal_mode(radius=2, kernelType="circle", units="pixels")

    if connected_components:
        # Keep only water pixels belonging to a connected region >= OTSU_MIN_WATER_PIXELS.
        # The open sea always exceeds this count; isolated inland water bodies do not.
        water_conn = mask.connectedPixelCount(maxSize=1024, eightConnected=False)
        mask = mask.And(water_conn.gte(OTSU_MIN_WATER_PIXELS))

    return ee.Image(mask.rename("water").copyProperties(img, img.propertyNames()))


# --------------------------------------------------------
#  Diagnostic plots
# --------------------------------------------------------

def plot_otsu_comparison(img: ee.Image, scale: int = 40):
    """Side-by-side Otsu mask without vs with connected-component filtering."""
    aoi             = _get_aoi()
    mask_without_cc = get_otsu_mask(img, scale=scale, redefined=True, connected_components=False)
    mask_with_cc    = get_otsu_mask(img, scale=scale, redefined=True, connected_components=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    try:
        fig.suptitle(f"Connected-component filter comparison\n{_img_label(img)}", fontsize=12)
    except Exception:
        fig.suptitle("Connected-component filter comparison", fontsize=12)

    axes[0].imshow(_open_image_thumbnail(mask_without_cc, aoi, VIS_BINARY_WATER_MASK))
    axes[0].set_title("Without connected components", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(_open_image_thumbnail(mask_with_cc, aoi, VIS_BINARY_WATER_MASK))
    axes[1].set_title(f"With connected components (min {OTSU_MIN_WATER_PIXELS} px)", fontsize=10)
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


def plot_single_image(img: ee.Image, title: str = "GEE Image", save: bool = False):
    aoi        = _get_aoi()
    band_names = img.bandNames().getInfo()

    if "VV" in band_names:
        viz      = VIS_SAR_VV
        plot_img = img.select("VV").log10().multiply(10)
    elif "water" in band_names:
        viz      = VIS_BINARY_WATER_MASK
        plot_img = img
    else:
        viz      = {"min": 0, "max": 1, "palette": ["000000", "ffffff"]}
        plot_img = img

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(_open_image_thumbnail(plot_img, aoi, viz))

    try:
        ax.set_title(f"{title}\n{_img_label(img)}", fontsize=11)
    except Exception:
        ax.set_title(title, fontsize=11)

    ax.axis("off")
    plt.tight_layout()

    if save:
        safe_title = re.sub(r"[^\w\-]", "_", title)
        full_path  = f"{OUTPUT_PLOTS}{safe_title}.png"
        plt.savefig(full_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {full_path}")

    plt.show()
