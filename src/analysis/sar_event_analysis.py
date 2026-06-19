import re
import ee
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from utils.collection_utils import _get_aoi, get_collection_s1
from utils.config import (
    OUTPUT_PLOTS, VIS_BINARY_WATER_MASK, VIS_SAR_VV, VIS_CHANGE_MAP,
    CHANGE_MAP_LABELS, STORM_EVENTS, QUANTIFICATION_SCALE,
)
from utils.gee_utils import init_gee
from analysis.sar_core import _open_image_thumbnail, _img_label, get_otsu_mask
from analysis.sar_timeseries_analysis import build_region_fc, _reduce_area_km2


# --------------------------------------------------------
#  Pair selection
# --------------------------------------------------------

def select_event_pair(col: ee.ImageCollection, event_date: str,
                      min_buffer_days: int = 2, max_pre_lag_days: int = 14,
                      max_post_lag_days: int = 21,
                      max_tide_diff_m: float = 0.15) -> tuple[ee.Image, ee.Image]:
    """Returns a tide-matched (pre, post) image pair around event_date.

    Pre  = most recent image in [event − max_pre_lag_days, event − min_buffer_days]
    Post = image in [event + min_buffer_days, event + max_post_lag_days] whose
           tidal_height_m is closest to the pre image (tide-matched, not nearest in time)
    """
    event_ee = ee.Date(event_date)

    pre_col = col.filterDate(
        event_ee.advance(-max_pre_lag_days, "day"),
        event_ee.advance(-min_buffer_days,  "day"),
    )
    if pre_col.size().getInfo() == 0:
        raise ValueError(
            f"No pre-event images for '{event_date}' "
            f"in [event − {max_pre_lag_days}d, event − {min_buffer_days}d]"
        )
    pre_img = pre_col.sort("system:time_start", False).first()
    zos_pre = float(pre_img.get("tidal_height_m").getInfo())

    post_col = col.filterDate(
        event_ee.advance(+min_buffer_days,   "day"),
        event_ee.advance(+max_post_lag_days, "day"),
    )
    if post_col.size().getInfo() == 0:
        raise ValueError(
            f"No post-event images for '{event_date}' "
            f"in [event + {min_buffer_days}d, event + {max_post_lag_days}d]"
        )
    post_img = (post_col
                .map(lambda img: img.set(
                    "tide_diff",
                    ee.Number(img.get("tidal_height_m")).subtract(zos_pre).abs()
                ))
                .sort("tide_diff")
                .first())

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


# --------------------------------------------------------
#  Plotting
# --------------------------------------------------------

def plot_sar_event(pre_img: ee.Image, post_img: ee.Image, event_date: str,
                   save: bool = False):
    """Plot VV (dB) SAR images for the pre- and post-event acquisitions."""
    aoi = _get_aoi()

    def vv_db(img):
        return img.select("VV").log10().multiply(10).clip(aoi)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Sentinel-1 VV (dB) of Event on: {event_date}", fontsize=13)

    for ax, img, label in zip(axes, [pre_img, post_img], ["PRE", "POST"]):
        ax.imshow(_open_image_thumbnail(vv_db(img), aoi, VIS_SAR_VV), cmap="gray")
        ax.set_title(f"{label}\n{_img_label(img)}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()

    if save:
        safe = re.sub(r"[^\w\-]", "_", event_date)
        path = f"{OUTPUT_PLOTS}sar_event_{safe}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved to {path}")

    plt.show()


def plot_coastline_event(pre_img: ee.Image, post_img: ee.Image, event_date: str,
                         mask_scale: int = 40, redefined: bool = True):
    """Plot water masks and a 4-class change map for an event.

    The change-map image is fetched as a thumbnail (fixed pixel budget), so
    mask_scale only affects classification accuracy — the visual appearance
    does not change noticeably.  The scale mainly matters for quantify_event.
    """
    aoi       = _get_aoi()
    pre_mask  = get_otsu_mask(pre_img,  mask_scale, redefined=redefined)
    post_mask = get_otsu_mask(post_img, mask_scale, redefined=redefined)
    change    = pre_mask.multiply(2).add(post_mask)

    print("\n--- Plotting pre-event, post-event, and change map ---")
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(f"Coastline analysis of event at {event_date}", fontsize=13)

    axes[0].imshow(_open_image_thumbnail(pre_mask, aoi, VIS_BINARY_WATER_MASK))
    axes[0].set_title(f"Pre-event water mask\n{_img_label(pre_img)}", fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(_open_image_thumbnail(post_mask, aoi, VIS_BINARY_WATER_MASK))
    axes[1].set_title(f"Post-event water mask\n{_img_label(post_img)}", fontsize=9)
    axes[1].axis("off")

    axes[2].imshow(_open_image_thumbnail(change, aoi, VIS_CHANGE_MAP))
    axes[2].set_title("Change map  (PRE → POST)", fontsize=9)
    axes[2].axis("off")

    legend_patches = [mpatches.Patch(color=f"#{color}", label=label)
                      for color, label in zip(VIS_CHANGE_MAP["palette"], CHANGE_MAP_LABELS)]
    axes[2].legend(handles=legend_patches, loc="lower left", fontsize=7, framealpha=0.8)

    plt.tight_layout()
    plt.show()


# --------------------------------------------------------
#  Quantification
# --------------------------------------------------------

def quantify_event(storm_id: str, pre_img: ee.Image, post_img: ee.Image,
                   scale: int = QUANTIFICATION_SCALE) -> pd.DataFrame:
    """Return region erosion and accretion areas for a storm event.

    Erosion and accretion are always reported separately (never netted).
    Percentage is relative to the SAR-covered pre-event land area of the region.
    """
    if storm_id not in STORM_EVENTS:
        raise ValueError(f"Unknown storm_id: {storm_id}")

    event = STORM_EVENTS[storm_id]
    print(f"Quantify event: {event['name']}")

    region_fc = build_region_fc()
    pre_mask  = get_otsu_mask(pre_img,  scale=scale)
    post_mask = get_otsu_mask(post_img, scale=scale)
    change    = pre_mask.multiply(2).add(post_mask)

    area_image = ee.Image.cat([
        change.eq(1).rename("erosion_km2"),                        # land -> water
        change.eq(2).rename("accretion_km2"),                      # water -> land
        pre_mask.select("water").Not().rename("pre_land_km2"),
    ])

    features = _reduce_area_km2(area_image, region_fc, scale).getInfo()["features"]

    rows = []
    for feat in features:
        p         = feat["properties"]
        pre_land  = p.get("pre_land_km2", 0) or 1e-9
        erosion   = p.get("erosion_km2",   0) or 0.0
        accretion = p.get("accretion_km2", 0) or 0.0
        rows.append({
            "region":         p["name"],
            "is_spit":        bool(p.get("is_spit", False)),
            "erosion_km2":    round(erosion,   4),
            "erosion_pct":    round(erosion   / pre_land * 100, 3),
            "accretion_km2":  round(accretion, 4),
            "accretion_pct":  round(accretion / pre_land * 100, 3),
        })

    df = pd.DataFrame(rows).sort_values("region").reset_index(drop=True)
    _print_event_table(df, event["name"])
    return df


def _print_event_table(df: pd.DataFrame, event_name: str):
    print(f"\nStorm: {event_name}")
    display_df = pd.DataFrame({
        "Region":         df["region"],
        "Erosion km2":    df["erosion_km2"].map("{:.4f}".format),
        "Erosion %":      df["erosion_pct"].map("{:.3f}%".format),
        "Accretion km2":  df["accretion_km2"].map("{:.4f}".format),
        "Accretion %":    df["accretion_pct"].map("{:.3f}%".format),
        "Spit":           df["is_spit"].map({True: "+", False: ""}),
    })
    print(display_df.to_string(index=False))
    ranked = df.sort_values("erosion_km2", ascending=False)["region"].tolist()
    print(f"Erosion ranking: {' > '.join(ranked)}")


def plot_event_bars(df: pd.DataFrame, storm_id: str, save: bool = False):
    """Grouped bar chart of erosion/accretion per region for one storm."""
    if storm_id not in STORM_EVENTS:
        raise ValueError(f"Unknown storm_id: {storm_id}")

    regions         = df["region"].tolist()
    x               = range(len(regions))
    width           = 0.35
    erosion_color   = f"#{VIS_CHANGE_MAP['palette'][1]}"
    accretion_color = f"#{VIS_CHANGE_MAP['palette'][2]}"

    fig, ax = plt.subplots(figsize=(max(6, len(regions) * 1.8), 5))
    ax.bar([i - width / 2 for i in x], df["erosion_km2"],   width,
           label="Erosion (land→water)",   color=erosion_color)
    ax.bar([i + width / 2 for i in x], df["accretion_km2"], width,
           label="Accretion (water→land)", color=accretion_color)
    ax.set_xticks(list(x))
    ax.set_xticklabels(regions, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Area (km2)")
    ax.set_title(f"Erosion / Accretion for {STORM_EVENTS[storm_id]['name']}")
    ax.legend(fontsize=9)
    plt.tight_layout()

    if save:
        safe = re.sub(r"[^\w\-]", "_", storm_id)
        path = f"{OUTPUT_PLOTS}event_bars_{safe}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved to {path}")

    plt.show()


# --------------------------------------------------------
#  Orchestrator
# --------------------------------------------------------

def run_event_analysis(storm_id: str, col: ee.ImageCollection,
                       save: bool = False,
                       event_scale: int = 20) -> pd.DataFrame:
    """Full event analysis: select pair once, then plot and quantify.

    event_scale controls both the Otsu mask resolution and the reduceRegions
    area calculation.  Default 20 m (finer than the 40 m timeseries scale)
    because storm retreat of 10–30 m is sub-pixel at 40 m.
    """
    if storm_id not in STORM_EVENTS:
        raise ValueError(f"Unknown storm_id: {storm_id}")

    select = STORM_EVENTS[storm_id]["select"]
    pre_img, post_img = select_event_pair(
        col,
        event_date       = select["event_date"],
        min_buffer_days  = select["min_buffer_days"],
        max_pre_lag_days = select["max_pre_lag_days"],
        max_post_lag_days= select["max_post_lag_days"],
    )

    print("\n--- SAR imagery ---")
    plot_sar_event(pre_img, post_img, select["event_date"], save=save)

    print("\n--- Change map ---")
    plot_coastline_event(pre_img, post_img, select["event_date"],
                         mask_scale=event_scale)

    print("\n--- Area quantification ---")
    df = quantify_event(storm_id, pre_img, post_img, scale=event_scale)
    plot_event_bars(df, storm_id, save=save)
    return df


# --------------------------------------------------------
#  Diagnostics
# --------------------------------------------------------

def compare_scales_sabine():
    """Run Sabine at 40 m, 20 m, and 10 m and print erosion/accretion side by side.

    Use this to decide whether finer resolution sharpens the signal enough to
    justify the extra GEE compute time.  Run once interactively; no plots.
    """
    init_gee()
    col = get_collection_s1()
    storm_id = "sabine_2020"
    select   = STORM_EVENTS[storm_id]["select"]

    pre_img, post_img = select_event_pair(
        col,
        event_date       = select["event_date"],
        min_buffer_days  = select["min_buffer_days"],
        max_pre_lag_days = select["max_pre_lag_days"],
        max_post_lag_days= select["max_post_lag_days"],
    )

    results = {}
    for scale in (40, 20, 10):
        print(f"\n=== scale = {scale} m ===")
        results[scale] = quantify_event(storm_id, pre_img, post_img, scale=scale)

    # Side-by-side comparison
    regions = results[40]["region"].tolist()
    header  = f"{'Region':<20}" + "".join(f"  {s}m eros  {s}m accr" for s in (40, 20, 10))
    print("\n\n" + "=" * len(header))
    print("Sabine: erosion/accretion km² by scale")
    print(header)
    print("-" * len(header))
    for r in regions:
        row = f"{r:<20}"
        for scale in (40, 20, 10):
            df  = results[scale]
            rec = df[df["region"] == r].iloc[0]
            row += f"  {rec['erosion_km2']:>8.4f}  {rec['accretion_km2']:>8.4f}"
        print(row)
    print("=" * len(header))


def test_pair_selection():
    """Test select_event_pair on the unfiltered S1 collection across all configured storms.

    Three scenarios per storm:
      A  default windows
      B  wider post window — useful if S1B gap leaves few post candidates
      C  storm config values from STORM_EVENTS
    """
    init_gee()
    col = get_collection_s1()

    for storm_id, val in STORM_EVENTS.items():
        config     = val["select"]
        event_date = config["event_date"]
        print(f"\nPair-selection test  storm={storm_id}  event={event_date}")

        print("\nA - Defaults  (pre <= 14days, post <= 21days, min_buffer=2days)")
        select_event_pair(col, event_date)

        print("\nB - Wider post window  (post <= 30days)")
        select_event_pair(col, event_date, max_post_lag_days=30)

        print(f"\nC - Storm config  ({val['name']})")
        select_event_pair(col, event_date,
                          min_buffer_days  = config["min_buffer_days"],
                          max_pre_lag_days = config["max_pre_lag_days"],
                          max_post_lag_days= config["max_post_lag_days"])
