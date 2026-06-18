import ee
from utils.gee_utils import init_gee
from utils import collection_utils
from utils.tidal_utils import filter_bin
from utils.config import STORM_EVENTS, BIN_LABELS, OUTPUT_DATA
from analysis import sar_core as sar
from analysis import optical_analysis as opt
from analysis import sar_event_analysis as event
from analysis import sar_timeseries_analysis as ts


# ----------- Initialization, get collection & tidal filter --------------------

def get_col(collection: str = "S1", tidal_bin: str | None = BIN_LABELS[2]):
    print("\n--- Initialize GEE ---")
    init_gee()

    print(f"\n--- Querying {collection} Collection ---")
    if collection == "S1":
        col = collection_utils.get_collection_s1()
    elif collection == "S2":
        col = collection_utils.get_collection_s2()
    else:
        raise ValueError(f"Collection not defined: {collection}")

    if tidal_bin is not None:
        print(f"\n--- Tidal Filter: {tidal_bin} ---")
        col = filter_bin(col, tidal_bin)
        if col.size().getInfo() == 0:
            raise AttributeError("No images remaining after tidal filtering.")

    return col


# ---------------------- SAR -------------------------------

def sar_one_image(date: str, tidal_bin: str = BIN_LABELS[2]):
    s1_col = get_col(collection="S1", tidal_bin=tidal_bin)

    print(f"\n--- Filter on date {date} ---")
    start_date = ee.Date(date)
    day_col    = s1_col.filterDate(start_date, start_date.advance(1, "day"))

    if day_col.size().getInfo() == 0:
        print(f"No image found on {date} during a '{tidal_bin}' tide.")
        return

    sar.plot_single_image(img=day_col.first(), title=f"Sentinel-1 overpass on {date}")


def sar_event_analysis(storm_id: str, save: bool = False):
    s1_col = get_col(collection="S1", tidal_bin=None)
    event.run_event_analysis(storm_id, s1_col, save=save)


def seasonal_quantification(tidal_bin: str = "near_msl", save: bool = False):
    col = get_col(collection="S1", tidal_bin=tidal_bin)

    cache_land = f"{OUTPUT_DATA}land_area.csv"
    cache_change = f"{OUTPUT_DATA}change_timeseries.csv"

    print("\n--- Land area timeseries ---")
    df_land_raw = ts.quantify_timeseries(col, cache_csv=cache_land)
    df_land     = ts.filter_outlier_dates(df_land_raw)
    bad_dates   = set(df_land_raw["date"]) - set(df_land["date"])
    ts.plot_timeseries(df_land, save=save)
    ts.plot_land_area_monthly_means(df_land, save=save)
    ts.plot_monthly_land_area_cycle(df_land, save=save)

    print("\n--- Consecutive-pair change timeseries ---")
    df_change = ts.quantify_change_timeseries(col, cache_csv=cache_change)
    df_change = ts.filter_outlier_dates_change(df_change, bad_dates)
    ts.plot_change_timeseries(df_change, save=save)
    ts.plot_erosion_by_month(df_change, save=save)


def sar_timeseries(tidal_bin: str = "near_msl"):
    s1_col = get_col(collection="S1", tidal_bin=tidal_bin)
    ts.generate_sar_timeseries_gif(col=s1_col, mask=True)


# ---------------------- SAR Otsu diagnostics -------------------------------

def sar_otsu_test(date: str = None, tidal_bin: str = BIN_LABELS[2]):
    """Side-by-side Otsu comparison. If date is None, uses the most recent image."""
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

def opt_timeseries(tidal_bin: str = BIN_LABELS[2]):
    col = get_col(collection="S2", tidal_bin=tidal_bin)
    opt.generate_s2_timeseries_gif(col)
    opt.generate_s2_timeseries_gif(col, ndwi=True)


def opt_one_image(date: str, tidal_bin: str = BIN_LABELS[2]):
    col = get_col(collection="S2", tidal_bin=tidal_bin)
    opt.plot_single_image_s2(col, date)
    opt.plot_single_image_s2(col, date, ndwi=True)


if __name__ == "__main__":
    # ---------- GENERAL ----------
    # sar_one_image("2019-06-27")
    # opt_one_image(date="2022-07-15")
    # sar_otsu_test("2019-06-27")
    # sar_otsu_test(tidal_bin="very_high")


    # ------- EVENT -----------
    # STORM_ID from "sabine_2020", "ylenia_zeynep_antonia_2022", "zoltan_2023"
    STORM_ID = "ylenia_zeynep_antonia_2022"
    # sar_event_analysis(STORM_ID)
    # event.test_pair_selection()
    

    # ---------- TIMESERIES ------------
    seasonal_quantification(tidal_bin="near_msl", save=True)
    sar_timeseries(tidal_bin="near_msl")
   

    

