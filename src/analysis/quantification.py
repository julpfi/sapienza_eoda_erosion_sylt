import re
import ee
import pandas as pd
import matplotlib.pyplot as plt

from utils.collection_utils import _get_aoi, get_collection_s1
from utils.config import (QUANTIFICATION_SCALE, REGIONS_OF_INTEREST,
                    WEST_COAST_AGGREGATE, STORM_EVENTS, SPIT_REGIONS,
                    OUTPUT_PLOTS, VIS_CHANGE_MAP
                    )
from analysis.sar_analysis import _select_event_date_pair, get_otsu_mask
from utils.tidal_utils import filter_bin



# -----------------------------------------------------------------
#  Region-of-interest configuration
# ------------------------------------------------------------------ 


def build_region_fc() -> ee.FeatureCollection:
    """
    Assemble the analysis FeatureCollection from config
    Uses the region of interests and the western aggregate as final 
    """
    features = []

    for name, geojson in REGIONS_OF_INTEREST.items():
        if geojson is None:
            print(f"{name}: geometry not set -> skipped")
            continue
        geom = ee.Geometry(geojson["features"][0]["geometry"])
        features.append(ee.Feature(geom, {"name": name, "is_spit": name in SPIT_REGIONS}))

    # Western coast aggregate (falls back to island AOI if not yet defined)
    if WEST_COAST_AGGREGATE is not None:
        agg_geom = ee.Geometry(WEST_COAST_AGGREGATE["features"][0]["geometry"])
    else:
        print("WEST_COAST_AGGREGATE not set - using island AOI as aggregate")
        agg_geom = _get_aoi()

    features.append(ee.Feature(agg_geom, {"name": "island_aggregate", "is_spit": False}))
    return ee.FeatureCollection(features)



# -------------------------------------------------------------
# Area computation
# -------------------------------------------------------------

def _reduce_area_km2(image:ee.Image, region_fc:ee.FeatureCollection, scale:int) -> ee.FeatureCollection:
    """Sum pixel area (km2) per band per region in one reduceRegions call"""
    pixel_area = ee.Image.pixelArea().divide(1e6)
    return (image.toFloat().multiply(pixel_area)
            .reduceRegions(
                collection=region_fc,
                reducer=ee.Reducer.sum(),
                scale=scale,
                tileScale=4
                )
            )




# ----------------------------------------------------------
#  Entry point 1 - event erosion/accretion
# ----------------------------------------------------------

def quantify_event(storm_id:str, col:ee.ImageCollection, scale:int=QUANTIFICATION_SCALE) -> pd.DataFrame:
    """
    Return region erosion and accretion areas for a storm event (given storm_id)
    Erosion and accretion are always reported separately (never netted)
    Percentage is relative to the SAR-covered area of the region
    """

    if storm_id not in STORM_EVENTS:
        raise ValueError(f"Unknown storm_id: {storm_id}")

    # Get storm info from config.py
    event = STORM_EVENTS[storm_id]
    select = event["select"]
    print(f"Quantify event: {event['name']}")

    # Get region of interests
    region_fc = build_region_fc()

    # Get event pairs using config params and _select_event_date_pair method
    pre_img, post_img = _select_event_date_pair(col,
        event_date = select["event_date"],
        min_buffer_days = select["min_buffer_days"],
        max_pre_lag_days = select["max_pre_lag_days"],
        max_post_lag_days = select["max_post_lag_days"]
    )

    # Mask the pre and post image
    pre_mask = get_otsu_mask(pre_img,  scale=scale)
    post_mask = get_otsu_mask(post_img, scale=scale)


    # 4-class change map: 0=land, 1=erosion, 2=accretion, 3=water
    change = pre_mask.multiply(2).add(post_mask)

    area_image = ee.Image.cat([
        change.eq(1).rename("erosion_km2"),                         # land -> water
        change.eq(2).rename("accretion_km2"),                       # water -> land
        pre_mask.select("water").Not().rename("pre_land_km2"),      # pre-event land
    ])

    features = _reduce_area_km2(area_image, region_fc, scale).getInfo()["features"]

    rows = []
    for feat in features:
        p = feat["properties"]
        pre_land  = p.get("pre_land_km2", 0) or 1e-9
        erosion   = p.get("erosion_km2",   0) or 0.0
        accretion = p.get("accretion_km2", 0) or 0.0

        rows.append({
            "region": p["name"],
            "is_spit": bool(p.get("is_spit", False)),
            "erosion_km2": round(erosion, 4),
            "erosion_pct": round(erosion / pre_land*100, 3), 
            "accretion_km2": round(accretion, 4),
            "accretion_pct": round(accretion / pre_land*100, 3)
        })


    df = (pd.DataFrame(rows).sort_values("region").reset_index(drop=True))
    _print_event_table(df, event["name"])
    
    return df



def _print_event_table(df: pd.DataFrame, event_name:str):
    print(f"\nStorm: {event_name}")

    display_df = pd.DataFrame({
        'Region': df['region'],
        'Erosion km2': df['erosion_km2'].map("{:.4f}".format),
        'Erosion %': df['erosion_pct'].map("{:.3f}%".format),
        'Accretion km2': df['accretion_km2'].map("{:.4f}".format),
        'Accretion %': df['accretion_pct'].map("{:.3f}%".format),
        'Spit': df['is_spit'].map({True: '+', False: ''})
    })
    print(display_df.to_string(index=False))

    ranked = df.sort_values("erosion_km2", ascending=False)["region"].tolist()
    print(f"Erosion ranking: {' > '.join(ranked)}")




def plot_event_bars(df:pd.DataFrame, storm_id:str, save:bool=False):
    """Grouped bar chart of erosion/accretion per region for one storm"""
    if storm_id not in STORM_EVENTS:
        raise ValueError(f"Unknown storm_id: {storm_id}")

    regions = df["region"].tolist()
    x = range(len(regions))
    width = 0.35

    erosion_color = f"#{VIS_CHANGE_MAP['palette'][1]}"
    accretion_color = f"#{VIS_CHANGE_MAP['palette'][2]}"

    fig, ax = plt.subplots(figsize=(max(6, len(regions) * 1.8), 5))

    ax.bar([i - width / 2 for i in x], df["erosion_km2"], width, 
           label="Erosion (land→water)", color=erosion_color)
    
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
#  Entry point 2 — timeseries land/water area
# -------------------------------------------------------

def quantify_timeseries(col:ee.ImageCollection=None, scale:int=QUANTIFICATION_SCALE, export:bool=False
                        ) -> pd.DataFrame|None:
    """
    Returns long-format land/water area per region per date (=image)
    Maps get_otsu_mask over the tidally-binned collectionl
    """

    if col is None:
        col = get_collection_s1()
        col = filter_bin(col, "near_msl")

    col_size = col.size().getInfo()
    print(f"\nQuantify_timeseries: {col_size} images")

    region_fc = build_region_fc()


    def process_one_image(raw_img):
        img = ee.Image(raw_img)
        mask = get_otsu_mask(img, scale=scale)
        water = mask.select("water").toFloat().rename("water_km2")
        land = mask.select("water").Not().toFloat().rename("land_km2")
        date = img.date().format("YYYY-MM-dd")
        return (_reduce_area_km2(ee.Image.cat([land, water]), region_fc, scale)
                .map(lambda f: f.set("date", date)))

    # Create flattend feature collections 
    img_list = col.sort("system:time_start").toList(col_size)
    all_fcs = img_list.map(process_one_image)
    flat_fc  = ee.FeatureCollection(all_fcs).flatten()

    if export:
        task = ee.batch.Export.table.toDrive(collection=flat_fc, description="sylt_land_area", fileFormat="CSV")
        task.start()
        print(f"Export task started ({task.id})")
        return None

    print("Fetching results")
    features = flat_fc.getInfo()["features"]

    rows = []
    for feat in features:
        p = feat["properties"]
        land = p.get("land_km2",  0) or 0.0
        water = p.get("water_km2", 0) or 0.0
        total = land + water or 1e-9
        rows.append({
            "date": p["date"],
            "region": p["name"],
            "land_km2": round(land,  4),
            "water_km2": round(water, 4),
            "land_fraction": round(land/total, 5),
        })

    df = (pd.DataFrame(rows)
          .assign(date=lambda d: pd.to_datetime(d["date"]))
          .sort_values(["region", "date"])
          .reset_index(drop=True))

    _print_timeseries_summary(df)
    return df



def _print_timeseries_summary(df:pd.DataFrame):
    print("\nLand-area summary per region (seasonal envelope):\n")

    df_sorted = df.sort_values(['region', 'date'])
    grouped = df_sorted.groupby('region')['land_km2'].agg(Min='min', Max='max', First='first', Last='last')

    display_df = pd.DataFrame({
        'Region': grouped.index,
        'Min km2': grouped['Min'].map("{:.3f}".format),
        'Max km2': grouped['Max'].map("{:.3f}".format),
        'Amplitude km2': (grouped['Max'] - grouped['Min']).map("{:.3f}".format),
        'Start-end delta km2': (grouped['Last'] - grouped['First']).map("{:+.3f}".format)
    })
    print(display_df.to_string(index=False))

    ranked = (grouped['Max'] - grouped['Min']).sort_values(ascending=False).index.tolist()
    print(f"  Amplitude ranking: {' > '.join(ranked)}")
    


def plot_timeseries(df:pd.DataFrame, save:bool=False):
    """Line plot of land area per region over time"""
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






# -----------------------------------------------------------
#  Entry point 3 — consecutive-pair erosion/accretion timeseries
# -----------------------------------------------------------

def quantify_change_timeseries(col:ee.ImageCollection, scale:int=QUANTIFICATION_SCALE,
                               export: bool = False) -> pd.DataFrame|None:
    """Return region-wise erosion/accretion for every consecutive image pair (tidal filtered collection)"""

    n = col.size().getInfo()
    print(f"\nQuantify change timeseries: {n} images -> {n-1} consecutive pairs")

    region_fc = build_region_fc()

    # Compute all masks once before 
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
            change.eq(2).rename("accretion_km2")
        ])

        date_pre  = pre_mask.date().format("YYYY-MM-dd")
        date_post = post_mask.date().format("YYYY-MM-dd")

        return (_reduce_area_km2(area_img, region_fc, scale)
                .map(lambda d: d.set("date_pre", date_pre, "date_post", date_post)))

    all_fcs = ee.List.sequence(0, n - 2).map(process_pair)
    flat_fc  = ee.FeatureCollection(all_fcs).flatten()

    if export:
        task = ee.batch.Export.table.toDrive(
            collection=flat_fc,
            description="sylt_change_timeseries",
            fileFormat="CSV"
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
            "date_pre": p["date_pre"],
            "date_post": p["date_post"],
            "region": p["name"],
            "erosion_km2": round(p.get("erosion_km2",   0) or 0.0, 4),
            "accretion_km2": round(p.get("accretion_km2", 0) or 0.0, 4)
        })

    df = (pd.DataFrame(rows)
          .assign(
              date_pre=lambda d: pd.to_datetime(d["date_pre"]),
              date_post=lambda d: pd.to_datetime(d["date_post"]),
          )
          .sort_values(["region", "date_post"])
          .reset_index(drop=True))

    _print_change_summary(df)
    return df



def _print_change_summary(df: pd.DataFrame):
    print("\nChange timeseries summary per region:\n")
    grouped = df.groupby("region")[["erosion_km2", "accretion_km2"]].agg(
        median_erosion=("erosion_km2", "median"),
        max_erosion=("erosion_km2", "max"),
        median_accretion=("accretion_km2", "median"),
        max_accretion=("accretion_km2", "max"),
    )
    display_df = pd.DataFrame({
        'Region': grouped.index,
        'Median erosion': grouped['median_erosion'].map("{:.4f}".format),
        'Max erosion': grouped['max_erosion'].map("{:.4f}".format),
        'Median accretion': grouped['median_accretion'].map("{:.4f}".format),
        'Max accretion': grouped['max_accretion'].map("{:.4f}".format)
    })
    print(display_df.to_string(index=False))

    ranked = grouped['max_erosion'].sort_values(ascending=False).index.tolist()
    print(f"Peak erosion ranking:\n{' > '.join(ranked)}")



def plot_change_timeseries(df: pd.DataFrame, save: bool = False):
    """
    Region-wise erosion and accretion lines over time (consecutive pairs)
    Points are at their actual post-acquisition date
    """
    erosion_color = f"#{VIS_CHANGE_MAP['palette'][1]}"
    accretion_color = f"#{VIS_CHANGE_MAP['palette'][2]}"

    regions = sorted(df["region"].unique())
    n_regions = len(regions)

    fig, axes = plt.subplots(n_regions, 1, figsize=(13, 3.5 * n_regions), sharex=True)
    if n_regions == 1:
        axes = [axes]

    for ax, region in zip(axes, regions):
        grp = df[df["region"] == region].sort_values("date_post")
        ax.plot(grp["date_post"], grp["erosion_km2"],
                color=erosion_color, marker=".", markersize=4, label="Erosion km²")
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
