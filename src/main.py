import ee
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

from gee_utils import init_gee
import collection_utils
from tidal_utils import filter_bin
from config import (STORM_EVENTS, BIN_LABELS)
import sar_analysis as sar
import optical_analysis as opt



# ----------- Initialization, get collection & tidal filter --------------------
def get_col(collection:str="S1", tidal_bin:str|None=BIN_LABELS[2]):
    # 1. Initialize GEE
    print("\n--- Initialize GEE ---")
    init_gee()

    # 2. Query type collection
    print(f"\n--- Querying {collection} Collection ---")
    if collection == "S1":
        col = collection_utils.get_collection_s1()
    elif collection == "S2":
        col = collection_utils.get_collection_s2()
    else:
        raise ValueError(f"Collection not defined: {collection}")

    # 3. Tidal filter (skipped if tidal_bin is None)
    if tidal_bin is not None:
        print(f"\n--- Tidal Filter: {tidal_bin} ---")
        col = filter_bin(col, tidal_bin)
        if col.size().getInfo() == 0:
            raise AttributeError("No images remaining after tidal filtering.")

    return col



# ---------------------- SAR -------------------------------
def sar_one_image(date:str, tidal_bin:str=BIN_LABELS[2]):
    s1_col = get_col(collection="S1", tidal_bin=tidal_bin)
    
    # Filter on date (Correctly spanning a full 24 hours)
    print(f"\n--- Filter on date {date} ---")
    start_date = ee.Date(date)
    day_col = s1_col.filterDate(start_date, start_date.advance(1, "day"))
    
    # SAFETY CHECK: Does an image actually exist on this day for this tide?
    if day_col.size().getInfo() == 0:
        print(f"No image found on {date} during a '{tidal_bin}' tide.")
        return

    # Safe to extract the image
    img = day_col.first()  

    sar.plot_single_image(img=img, title=f"Sentinel-1 overpass on {date}")





def event_analyis_sar(storm_id: str):
    storm_config = STORM_EVENTS[storm_id]["select"]
    s1_col = get_col(collection="S1", tidal_bin=None)

    print(f"{s1_col.size().getInfo()} images pass")
    if s1_col.size().getInfo() == 0:
        return

    print("\n--- Coastline / water mask + change map ---")
    sar.plot_coastline_event(s1_col, storm_config["event_date"],
                             min_buffer_days=storm_config["min_buffer_days"],
                             max_pre_lag_days=storm_config["max_pre_lag_days"],
                             max_post_lag_days=storm_config["max_post_lag_days"])


def sar_timeseries(tidal_bin:str=BIN_LABELS[3]):
    s1_col = get_col(collection="S1", tidal_bin=tidal_bin)

    s1_col_masks = s1_col.map(lambda img: sar.get_otsu_mask(img, redefined=True))
    sar.generate_sar_timeseries_gif(col=s1_col_masks, mask=True)



# ---------------------- SAR Otsu test -------------------------------
def sar_otsu_test(date:str=None, tidal_bin:str=BIN_LABELS[2]):
    """Side-by-side comparison of old Otsu (full AOI) vs new Otsu (calibration strip).
    If date is None, uses the most recent image in the tidal bin.
    """
    s1_col = get_col(collection="S1", tidal_bin=tidal_bin)

    if date is not None:
        start_date = ee.Date(date)
        day_col    = s1_col.filterDate(start_date, start_date.advance(1, "day"))
        if day_col.size().getInfo() == 0:
            print(f"No image found on {date} during a '{tidal_bin}' tide.")
            return
        img = day_col.first()
    else:
        img = s1_col.sort("system:time_start", False).first()

    sar.plot_otsu_comparison(img)


# ---------------------- Optical -------------------------------
def opt_timeseries(tidal_bin:str=BIN_LABELS[2]):
    col = get_col(collection="S2", tidal_bin=tidal_bin)

    opt.generate_s2_timeseries_gif(col)                       # true colour GIF
    opt.generate_s2_timeseries_gif(col, ndwi=True)   


def opt_one_image(date:str, tidal_bin:str=BIN_LABELS[2]):
    col = get_col(collection="S2", tidal_bin=tidal_bin)

    opt.plot_single_image_s2(col, date)              # true colour
    opt.plot_single_image_s2(col, date, ndwi=True)   # NDWI water index



if __name__ == "__main__":
    # STORM_ID from "sabine_2020", "ylenia_zeynep_antonia_2022", "zoltan_2023"
    STORM_ID = "sabine_2020"

    event_analyis_sar(STORM_ID)
    # sar.test_pair_selection()
    # sar_timeseries()
    # sar_one_image("2019-06-27")
    # opt_one_image(date="2022-07-15")
    # sar_otsu_test("2019-06-27")                      # easy summer image
    # sar_otsu_test(tidal_bin="very_high")             # most recent storm/surge image
