import re
import ee
import numpy as np
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

def select_event_pair(col: ee.ImageCollection,
                      block_start: str, block_end: str,
                      max_pre_lag_days: int = 14, max_post_lag_days: int = 21,
                      max_tide_diff_m: float = 0.15) -> tuple[ee.Image, ee.Image]:
    """
    Find the best same-orbit (pre, post) pair across all orbits in col

    Pre window = [block_start - max_pre_lag_days, block_start)
    Post window = (block_end,  block_end + max_post_lag_days] 

    For each orbit, selects:
      pre = latest image in the pre window on that orbit
      post = image in the post window minimising abs(tide(post) - tide(pre)) 
    => Select orbit with minimal tide diff
    """

    bs = ee.Date(block_start)
    be = ee.Date(block_end)

    pre_col = col.filterDate(bs.advance(-max_pre_lag_days, "day"), bs)
    post_col = col.filterDate(be.advance(1, "day"), be.advance(1 + max_post_lag_days, "day"))

    # Pull all candidate data 
    def fetch_meta(c: ee.ImageCollection) -> dict:
        return ee.Dictionary({
            "times":  c.aggregate_array("system:time_start"),
            "tides":  c.aggregate_array("tidal_height_m"),
            "orbits": c.aggregate_array("relativeOrbitNumber_start"),
            "passes": c.aggregate_array("orbitProperties_pass")
        }).getInfo()

    pre_meta = fetch_meta(pre_col)
    post_meta = fetch_meta(post_col)


    # Per-orbit: keep only the LATEST pre image
    pre_by_orbit: dict = {}
    for t, tide, orb, pass_dir in zip(
        pre_meta["times"], pre_meta["tides"], pre_meta["orbits"], pre_meta["passes"]
        ):

        if orb not in pre_by_orbit or t > pre_by_orbit[orb]["time"]:
            pre_by_orbit[orb] = {"time": t, "tide": float(tide), "pass_dir": pass_dir}


    # Per-orbit: all post images
    post_by_orbit: dict = {}
    for t, tide, orb in zip(
        post_meta["times"], post_meta["tides"], post_meta["orbits"]
        ):

        post_by_orbit.setdefault(orb, []).append({"time": t, "tide": float(tide)})

    # Score each orbit
    candidates = []
    for orb, pre in pre_by_orbit.items():
        if orb not in post_by_orbit:
            continue
        pre_tide = pre["tide"]
        best_post = min(post_by_orbit[orb], key=lambda p: abs(p["tide"] - pre_tide))
        tide_diff = abs(best_post["tide"] - pre_tide)
        span_days = (best_post["time"] - pre["time"]) / 86_400_000
        candidates.append({
            "orbit":     orb,
            "pass_dir":  pre["pass_dir"],
            "pre_time":  pre["time"],
            "post_time": best_post["time"],
            "pre_tide":  pre_tide,
            "post_tide": best_post["tide"],
            "tide_diff": tide_diff,
            "span_days": span_days,
        })

    if not candidates:
        raise ValueError(f"No orbit yields a valid pre/post pair for storm block")


    candidates.sort(key=lambda c: (c["tide_diff"], c["span_days"]))
    best = candidates[0]

    pre_date = pd.to_datetime(best["pre_time"],  unit="ms", utc=True).strftime("%Y-%m-%d")
    post_date = pd.to_datetime(best["post_time"], unit="ms", utc=True).strftime("%Y-%m-%d")

    print(f"Selected orbit {best['orbit']} ({best['pass_dir']})")
    print(f"\tPre:         {pre_date}  ({best['pre_tide']:+.2f} m)")
    print(f"\tPost:        {post_date} ({best['post_tide']:+.2f} m)")
    print(f"\tTide diff:   {best['tide_diff']:.3f} m  |  span: {best['span_days']:.0f} days")


    if len(candidates) > 1:
        print(" Other candidate orbits:")
        for c in candidates[1:]:
            pd_ = pd.to_datetime(c["pre_time"],  unit="ms", utc=True).strftime("%Y-%m-%d")
            po_ = pd.to_datetime(c["post_time"], unit="ms", utc=True).strftime("%Y-%m-%d")
            print(f"\tOrbit {c['orbit']} ({c['pass_dir']}): {pd_} / {po_} with tide diff {c['tide_diff']:.3f} m")


    if best["tide_diff"] > max_tide_diff_m:
        print(f"WARNING: tide diff {best['tide_diff']:.3f} m exceeds {max_tide_diff_m} m")

    if best["span_days"] <= 0:
        raise ValueError(f"Degenerate pair for block [{block_start}, {block_end}]")


    # Retrieve the best pre and post images 
    orb_filter = ee.Filter.eq("relativeOrbitNumber_start", best["orbit"])
    pre_img = pre_col.filter(orb_filter).sort("system:time_start", False).first()
    post_img = (post_col
                .filter(orb_filter)
                .filterDate(ee.Date(best["post_time"]),
                            ee.Date(best["post_time"]).advance(1, "day"))
                .first())

    return pre_img, post_img



# --------------------------------------------------------
#  Plotting
# --------------------------------------------------------

def plot_sar_event(pre_img: ee.Image, post_img: ee.Image, event_date: str,
                   save: bool = False):
    """Plot VV (dB) SAR images for the pre- and post-event acquisitions"""
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
                         mask_scale: int = 40, redefined: bool = True, save: bool = False):
    """
    Plot water masks and a 4-class change map for an event
    mask_scale only affects classification accuracy -> the visual appearance does not change really
    """
    aoi = _get_aoi()
    pre_mask = get_otsu_mask(pre_img,  mask_scale, redefined=redefined)
    post_mask = get_otsu_mask(post_img, mask_scale, redefined=redefined)
    change = pre_mask.multiply(2).add(post_mask)

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

    if save:
        safe = re.sub(r"[^\w\-]", "_", event_date)
        path = f"{OUTPUT_PLOTS}change_map_{safe}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved to {path}")

    plt.show()


# --------------------------------------------------------
#  Quantification
# --------------------------------------------------------

def quantify_event(storm_id: str, pre_img: ee.Image, post_img: ee.Image,
                   scale: int = QUANTIFICATION_SCALE) -> pd.DataFrame:
    """
    Return region erosion and accretion areas for a storm event
    Percentage is relative to the SAR-covered pre-event land area of the region
    """
    if storm_id not in STORM_EVENTS:
        raise ValueError(f"Unknown storm_id: {storm_id}")

    event = STORM_EVENTS[storm_id]
    print(f"Quantify event: {event['name']}")

    region_fc = build_region_fc()
    pre_mask = get_otsu_mask(pre_img,  scale=scale)
    post_mask = get_otsu_mask(post_img, scale=scale)
    change = pre_mask.multiply(2).add(post_mask)

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
        "Spit":           df["is_spit"].map({True: "+", False: ""})
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
#  Run whole event analysis
# --------------------------------------------------------

def run_event_analysis(storm_id: str, col: ee.ImageCollection,
                       save: bool = False, event_scale: int = 20) -> pd.DataFrame:
    """
    Full event analysis: select pair once, then plot and quantify
    event_scale controls the Otsu mask resolution and the reduceRegions area calculation
    """
    if storm_id not in STORM_EVENTS:
        raise ValueError(f"Unknown storm_id: {storm_id}")

    select = STORM_EVENTS[storm_id]["select"]
    pre_img, post_img = select_event_pair(
        col,
        block_start      = select["block_start"],
        block_end        = select["block_end"],
        max_pre_lag_days = select["max_pre_lag_days"],
        max_post_lag_days= select["max_post_lag_days"],
    )

    print("\n--- SAR imagery ---")
    plot_sar_event(pre_img, post_img, select["block_start"], save=save)

    print("\n--- Change map ---")
    plot_coastline_event(pre_img, post_img, select["block_start"],
                         mask_scale=event_scale, save=save)

    print("\n--- Area quantification ---")
    df = quantify_event(storm_id, pre_img, post_img, scale=event_scale)
    plot_event_bars(df, storm_id, save=save)
    return df


# --------------------------------------------------------
#  Diagnostics
# --------------------------------------------------------

def compare_scales_sabine(save: bool = False):
    """Run Sabine at 40 m, 20 m, and 10 m and print erosion/accretion side by side.

    Use this to decide whether finer resolution sharpens the signal enough to
    justify the extra GEE compute time.  Optionally saves a grouped-bar figure.
    """
    init_gee()
    col = get_collection_s1(orbit=None)   # all orbits — let select_event_pair choose
    storm_id = "sabine_2020"
    select   = STORM_EVENTS[storm_id]["select"]

    pre_img, post_img = select_event_pair(
        col,
        block_start      = select["block_start"],
        block_end        = select["block_end"],
        max_pre_lag_days = select["max_pre_lag_days"],
        max_post_lag_days= select["max_post_lag_days"],
    )

    scales = (40, 20, 10)
    results = {}
    for scale in scales:
        print(f"\n=== scale = {scale} m ===")
        results[scale] = quantify_event(storm_id, pre_img, post_img, scale=scale)

    # Side-by-side comparison
    regions = results[40]["region"].tolist()
    header  = f"{'Region':<20}" + "".join(f"  {s}m eros  {s}m accr" for s in scales)
    print("\n\n" + "=" * len(header))
    print("Sabine: erosion/accretion km² by scale")
    print(header)
    print("-" * len(header))
    for r in regions:
        row = f"{r:<20}"
        for scale in scales:
            df  = results[scale]
            rec = df[df["region"] == r].iloc[0]
            row += f"  {rec['erosion_km2']:>8.4f}  {rec['accretion_km2']:>8.4f}"
        print(row)
    print("=" * len(header))

    if save:
        scale_colors = ["#AAAAAA", "#555577", "#223344"]
        x = np.arange(len(regions))
        bar_w = 0.25
        fig, ax = plt.subplots(figsize=(max(8, len(regions) * 2), 5))
        for i, (scale, color) in enumerate(zip(scales, scale_colors)):
            erosion_vals = [results[scale].loc[results[scale]["region"] == r, "erosion_km2"].iloc[0]
                            for r in regions]
            ax.bar(x + (i - 1) * bar_w, erosion_vals, bar_w,
                   label=f"{scale} m", color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(regions, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Erosion (km²)")
        ax.set_title("Sabine 2020 — erosion by region and pixel scale (tide-matched pair)")
        ax.legend(title="Scale", fontsize=9)
        plt.tight_layout()
        path = f"{OUTPUT_PLOTS}scale_comparison_sabine.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved to {path}")
        plt.show()



def test_pair_selection():
    """Test select_event_pair on the all-orbit S1 collection for every configured storm

    For each storm:
      A  storm config windows (primary)
      B  wider post window (+14 days) to verify alternatives exist
    """
    init_gee()
    col = get_collection_s1(orbit=None)   # all orbits for event analysis

    for storm_id, val in STORM_EVENTS.items():
        config = val["select"]
        print(f"\n{'=' * 60}")
        print(f"Pair-selection test  storm={storm_id}  ({val['name']})")
        print(f"  Block: [{config['block_start']}, {config['block_end']}]")

        print("\nA - Storm config windows")
        select_event_pair(
            col,
            block_start      = config["block_start"],
            block_end        = config["block_end"],
            max_pre_lag_days = config["max_pre_lag_days"],
            max_post_lag_days= config["max_post_lag_days"],
        )

        print("\nB - Wider post window (+14 days)")
        select_event_pair(
            col,
            block_start      = config["block_start"],
            block_end        = config["block_end"],
            max_pre_lag_days = config["max_pre_lag_days"],
            max_post_lag_days= config["max_post_lag_days"] + 14,
        )
