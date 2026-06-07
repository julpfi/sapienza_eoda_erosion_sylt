import io
import re
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, ImageDraw, ImageSequence
import ee

from utils.collection_utils import _get_aoi, _get_calibration_strip, get_collection_s1
from utils.config import (START_DATE, END_DATE,
                    OUTPUT_ANIMATIONS, OUTPUT_PLOTS,
                    VIS_BINARY_WATER_MASK, VIS_CHANGE_MAP, CHANGE_MAP_LABELS, VIS_SAR_VV,
                    OTSU_MIN_WATER_PIXELS, STORM_EVENTS)
from utils.gee_utils import init_gee


# ------------------------------------------------------------ #
#  1. Technical Helpers Methods 
# ------------------------------------------------------------ #

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



def _select_event_date_pair(col: ee.ImageCollection, event_date: str, min_buffer_days: int = 2,
    max_pre_lag_days: int = 14, max_post_lag_days: int = 21, max_tide_diff_m: float = 0.15,) -> tuple[ee.Image, ee.Image]:
    """Returns a tide-matched (pre, post) image pair around event_date

    Pre  = most recent image in [event − max_pre_lag_days, event − min_buffer_days]
    Post = image in [event + min_buffer_days, event + max_post_lag_days] whose
           tidal_height_m is closest to the pre image (tide-matched, not nearest in time)
    """
    event_ee = ee.Date(event_date)

    # --- Pre: most recent image before the storm window ---
    pre_col = col.filterDate(
        event_ee.advance(-max_pre_lag_days, "day"),
        event_ee.advance(-min_buffer_days,  "day"),
    )
    if pre_col.size().getInfo() == 0:
        raise ValueError(f"No pre-event images for '{event_date}' in [event − {max_pre_lag_days}d, event − {min_buffer_days}d]"
                         )
    pre_img = pre_col.sort("system:time_start", False).first()
    zos_pre = float(pre_img.get("tidal_height_m").getInfo())

    # --- Post: image whose tidal height best matches pre ---
    post_col = col.filterDate(
        event_ee.advance(+min_buffer_days,   "day"),
        event_ee.advance(+max_post_lag_days, "day"),
    )
    if post_col.size().getInfo() == 0:
        raise ValueError(f"No post-event images for '{event_date}' in [event + {min_buffer_days}d, event + {max_post_lag_days}d]")
    
    post_img = (post_col
                .map(lambda img: img.set("tide_diff", ee.Number(img.get("tidal_height_m")).subtract(zos_pre).abs()))
                .sort("tide_diff")
                .first()
                )

    # --- Check ---
    pre_info  = pre_img.toDictionary( ["system:time_start", "tidal_height_m"]).getInfo()
    post_info = post_img.toDictionary(["system:time_start", "tidal_height_m"]).getInfo()
    pre_date  = pd.to_datetime(pre_info["system:time_start"],  unit="ms", utc=True).strftime("%Y-%m-%d")
    post_date = pd.to_datetime(post_info["system:time_start"], unit="ms", utc=True).strftime("%Y-%m-%d")
    pre_tide  = float(pre_info.get( "tidal_height_m", float("nan")))
    post_tide = float(post_info.get("tidal_height_m", float("nan")))
    print(f"Pre: {pre_date} ({pre_tide:+.2f} m) \nPost: {post_date} ({post_tide:+.2f} m)  \nDelta tide: {abs(post_tide - pre_tide):.2f} m")

    if abs(post_tide - pre_tide) > max_tide_diff_m:
        print(f"Diff in tide exceeds {max_tide_diff_m} m -> analyze change map carefully")

    return pre_img, post_img


# ------------------------------------------------------------ #
#  2. Coastline Detection Core Methods
# ------------------------------------------------------------ #

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
        i = ee.Number(i).toInt()
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



def get_otsu_mask(img:ee.Image, scale:int=40, redefined:bool=True,
                  calibration_geom:ee.Geometry=None, connected_components:bool=True) -> ee.Image:
    """Executes Otsu on Google Earth Engine.
    calibration_geom:      geometry used for histogram sampling; defaults to _get_calibration_strip().
    connected_components:  if True, removes isolated water pockets via connectedPixelCount.
    """
    comp = (img.select("VV").multiply(img.select("VH")).log10().multiply(10).rename("vv_vh_composite"))

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
        # Remove isolated water pockets (inland ponds, specular surfaces such as airport tarmac)
        # by keeping only water pixels that belong to a connected region >= OTSU_MIN_WATER_PIXELS.
        # The open sea always exceeds this count; isolated inland water bodies do not.
        water_conn = mask.connectedPixelCount(maxSize=1024, eightConnected=False)
        mask = mask.And(water_conn.gte(OTSU_MIN_WATER_PIXELS))

    return ee.Image(mask.rename("water").copyProperties(img, img.propertyNames()))




# ------------------------------------------------------------ #
#  3. General Analyis
# ------------------------------------------------------------ #

def plot_otsu_comparison(img: ee.Image, scale: int = 40):
    """Temporary diagnostic: Otsu mask without vs with connected-component filtering."""
    aoi = _get_aoi()

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


def plot_single_image(img: ee.Image, title:str="GEE Image", save:bool=False):
    aoi = _get_aoi()
    
    # Get bands of image from gee 
    band_names = img.bandNames().getInfo()
   
    if "VV" in band_names:
        viz = VIS_SAR_VV
    elif "water" in band_names:
        viz = VIS_BINARY_WATER_MASK
    else:
        # Only fallback case
        viz = {"min": 0, "max": 1, "palette": ["000000", "ffffff"]}

    fig, ax = plt.subplots(figsize=(8, 8))
    

    if "VV" in band_names:
        plot_img = img.select("VV").log10().multiply(10)
    else:
        plot_img = img


    thumb = _open_image_thumbnail(plot_img, aoi, viz)
    ax.imshow(thumb)

    try:
        subtitle = _img_label(img)
        ax.set_title(f"{title}\n{subtitle}", fontsize=11)
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


# ------------------------------------------------------------ #
#  4. Event Analyis
# ------------------------------------------------------------ #

def plot_sar_event(col: ee.ImageCollection, event_date: str,
                   min_buffer_days: int = 2, max_pre_lag_days: int = 14,
                   max_post_lag_days: int = 21, max_tide_diff_m: float = 0.15):
    """Plot VV (dB) SAR images for the pre- and post-event acquisitions"""
    
    aoi = _get_aoi()
    pre_img, post_img = _select_event_date_pair(
        col, event_date,
        min_buffer_days=min_buffer_days,
        max_pre_lag_days=max_pre_lag_days,
        max_post_lag_days=max_post_lag_days,
        max_tide_diff_m=max_tide_diff_m,
    )

    def vv_db(img):
        return img.select("VV").log10().multiply(10).clip(aoi)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Sentinel-1 VV (dB) of Event on: {event_date}", fontsize=13)

    for ax, img, label in zip(axes, [pre_img, post_img], ["PRE", "POST"]):
        thumb = _open_image_thumbnail(vv_db(img), aoi, VIS_SAR_VV)
        ax.imshow(thumb, cmap="gray")
        ax.set_title(f"{label}\n{_img_label(img)}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_coastline_event(col: ee.ImageCollection, event_date: str,
                         min_buffer_days: int = 2, max_pre_lag_days: int = 14,
                         max_post_lag_days: int = 21, max_tide_diff_m: float = 0.15,
                         mask_scale: int = 40, redefined: bool = True):
    """Plot water masks and a 4-class change map for an event"""
    aoi = _get_aoi()
    pre_img, post_img = _select_event_date_pair(
        col, event_date,
        min_buffer_days=min_buffer_days,
        max_pre_lag_days=max_pre_lag_days,
        max_post_lag_days=max_post_lag_days,
        max_tide_diff_m=max_tide_diff_m,
    )

    pre_mask  = get_otsu_mask(pre_img, mask_scale, redefined=redefined)
    post_mask = get_otsu_mask(post_img, mask_scale, redefined=redefined)
    
    change = pre_mask.multiply(2).add(post_mask)

    print("\n--- Plotting pre-event, post-event, and change map --- ")
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(f"Coastline analysis of event at {event_date}", fontsize=13)

    # Pre mask
    axes[0].imshow(_open_image_thumbnail(pre_mask, aoi, VIS_BINARY_WATER_MASK))
    axes[0].set_title(f"Pre-event water mask\n{_img_label(pre_img)}", fontsize=9)
    axes[0].axis("off")

    # Post mask
    axes[1].imshow(_open_image_thumbnail(post_mask, aoi, VIS_BINARY_WATER_MASK))
    axes[1].set_title(f"Post-event water mask\n{_img_label(post_img)}", fontsize=9)
    axes[1].axis("off")

    # Change map
    axes[2].imshow(_open_image_thumbnail(change, aoi, VIS_CHANGE_MAP))
    axes[2].set_title("Change map  (PRE → POST)", fontsize=9)
    axes[2].axis("off")

    # Dynamically generate legend patches using the config settings 
    legend_patches = [mpatches.Patch(color=f"#{color}", label=label) 
                      for color, label in zip(VIS_CHANGE_MAP["palette"], CHANGE_MAP_LABELS)]
    axes[2].legend(handles=legend_patches, loc="lower left", fontsize=7, framealpha=0.8)

    plt.tight_layout()
    plt.show()




# ------------------------------------------------------------ #
#  4. Time Series
# ------------------------------------------------------------ #

def generate_sar_timeseries_gif(col:ee.ImageCollection, mask:bool, fps:int=2, width:int=600):
    aoi = _get_aoi()

    times = col.aggregate_array("system:time_start").getInfo()
    dates = [pd.to_datetime(t, unit="ms", utc=True).strftime('%Y-%m-%d') for t in times]

    def prep_for_gif(img):
        if mask:
            return img.visualize(**VIS_BINARY_WATER_MASK)
        else: 
            vv_db = img.select("VV").log10().multiply(10)
            return vv_db.visualize(**VIS_SAR_VV)

    col_prepared = col.map(prep_for_gif)

    print("Rendering and downloading raw GIF...")
    gif_url = col_prepared.getVideoThumbURL({
        'dimensions': width,
        'region': aoi,
        'framesPerSecond': fps,
        'crs': 'EPSG:3857'
    })
    response = requests.get(gif_url)
    response.raise_for_status()

    raw_gif = Image.open(io.BytesIO(response.content))
    frames = []

    for i, frame in enumerate(ImageSequence.Iterator(raw_gif)):
        frame = frame.convert("RGBA")
        draw = ImageDraw.Draw(frame)

        date_text = dates[i] if i < len(dates) else "Unknown"

        x, y = 15, 15 
        
        draw.text((x-1, y), date_text, fill="black")
        draw.text((x+1, y), date_text, fill="black")
        draw.text((x, y-1), date_text, fill="black")
        draw.text((x, y+1), date_text, fill="black")
        draw.text((x, y), date_text, fill="white")

        frames.append(frame)

    mode_tag    = "mask" if mask else "vv"
    output_path = f"{OUTPUT_ANIMATIONS}timeseries_sar_{mode_tag}.gif"
    print(f"Saving GIF ({len(frames)} frames) to {output_path}...")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=int(1000 / fps)
    )
    print("Finished")


# ------------------------------------------------------------ #
#  5. Pair-selection diagnostics
# ------------------------------------------------------------ #

def test_pair_selection():
    """Test of _select_event_date_pair on the unfiltered S1 collection
    Three scenarios are looked at:
      A  default windows
      B  wider post window — useful if S1B gap leaves few post candidates
      C  storm config values from STORM_EVENTS
    """
    init_gee()
    col = get_collection_s1()

    for storm_id, val in STORM_EVENTS.items():
        config = val["select"]
        event_date = config["event_date"]

        print(f"\nPair-selection test  storm={storm_id}  event={event_date}")

        print("\nA - Defaults  (pre <= 14days, post <= 21days, min_buffer=2days)")
        _select_event_date_pair(col, event_date)

        print("\nB - Wider post window  (post <= 30days)")
        _select_event_date_pair(col, event_date, max_post_lag_days=30)

        print(f"\nC - Storm config  ({STORM_EVENTS[storm_id]['name']})")
        _select_event_date_pair(col, event_date,
                                min_buffer_days=config["min_buffer_days"],
                                max_pre_lag_days=config["max_pre_lag_days"],
                                max_post_lag_days=config["max_post_lag_days"])