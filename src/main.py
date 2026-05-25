import ee
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

from gee_utils import init_gee
from collection_utils import get_collection_s1, _get_aoi, inspect_orbits_by_bins
from tidal_utils import filter_bin
from config import (EVENT_DATE_ORKAN_ZEYNEP, BIN_LABELS)
import sar_analysis as sar



# -------------------- Main --------------------
def sar_one_image(date:str, tidal_bin:str=BIN_LABELS[2]):
    # 1. Initialize GEE
    print("\n--- Initialize GEE ---")
    init_gee()
    
    # 2. Query S1 collection
    print("\n--- Querying S1 Collection ---")
    s1_col = get_collection_s1()

    # 3. Tidal filter (Using the variable, not the hardcoded string)
    print(f"\n--- Tidal Filter: {tidal_bin} ---")
    s1_col = filter_bin(s1_col, tidal_bin)

    total_filtered = s1_col.size().getInfo()
    if total_filtered == 0: 
        return

    # 4. Filter on date (Correctly spanning a full 24 hours)
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





def event_analyis_sar(event_date:str, tidal_bin:str=BIN_LABELS[2]):
    print("\n--- Initialize GEE ---")
    init_gee()
 
    print("\n--- Querying S1 Collection (all tidal stages) ---")
    s1_col = get_collection_s1()
 
    # 3. Tidal filter
    print(f"\n--- Tidal Filter: {tidal_bin} ---")
    s1_col = filter_bin(s1_col, tidal_bin)

    print(f"{s1_col.size().getInfo()} images pass {tidal_bin} filter")
    if s1_col.size().getInfo() == 0: 
        return

    # Coastdetection and Change 
    print("\n--- Coastline / water mask + change map ---")
    sar.plot_coastline_event(s1_col, event_date, buffer_days=1)


def sar_timeseries(tidal_bin:str=BIN_LABELS[3]):
    print("\n--- Initialize GEE ---")
    init_gee()
    
    print("\n--- Querying S1 Collection (all tidal stages) ---")
    s1_col = get_collection_s1()
 
    # 3. Tidal filter
    print(f"\n--- Tidal Filter: {tidal_bin} ---")
    s1_col = filter_bin(s1_col, tidal_bin)

    print(f"{s1_col.size().getInfo()} images pass {tidal_bin} filter")
    if s1_col.size().getInfo() == 0: return

    s1_col_masks = s1_col.map(lambda img: sar.get_otsu_mask(img, redefined=True))
    sar.generate_sar_timeseries_gif(col=s1_col_masks, mask=True)



if __name__ == "__main__":
    # event_analyis_sar()
    sar_timeseries()
    #sar_one_image("2019-06-27")
