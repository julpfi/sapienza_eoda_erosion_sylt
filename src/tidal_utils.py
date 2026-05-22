import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copernicusmarine
import ee

from config import (
    CMEMS_DATASET, CMEMS_VARIABLE,
    CMEMS_LON_MIN, CMEMS_LON_MAX, CMEMS_LAT_MIN, CMEMS_LAT_MAX,
    START_DATE, END_DATE,
    TIDAL_WINDOW_M, CMEMS_DATA_PATH, BIN_EDGES, BIN_LABELS
)



# -------------------- Download/Load --------------------
def download_cmems_data(output=CMEMS_DATA_PATH):
    """Download Sea Surface Height from CMEMS and cache to CSV under given path"""
    os.makedirs(os.path.dirname(output), exist_ok=True)

    print("Connecting to CMEMS...")
    ds = copernicusmarine.open_dataset(
        dataset_id        = CMEMS_DATASET,
        variables         = [CMEMS_VARIABLE],
        minimum_longitude = CMEMS_LON_MIN,
        maximum_longitude = CMEMS_LON_MAX,
        minimum_latitude  = CMEMS_LAT_MIN,
        maximum_latitude  = CMEMS_LAT_MAX,
        start_datetime    = START_DATE + "T00:00:00",
        end_datetime      = END_DATE + "T23:00:00",
    )

    ssh_df = ds[CMEMS_VARIABLE].mean(dim=["longitude", "latitude"]).to_pandas()
    ssh_df.index = pd.to_datetime(ssh_df.index, utc=True)
    ssh_df.name  = "zos_m"
    ssh_df.to_csv(output)

    print(f"Saved SSH series of length {len(ssh_df)} to {output}")
    return ssh_df


def load_cmems_data(cache=CMEMS_DATA_PATH):
    """Load SSH time series from CSV cache (download if not found)"""
    if not os.path.exists(cache):
        return download_cmems_data(output=cache)
    
    ssh_df = pd.read_csv(cache, index_col=0, parse_dates=True)
    ssh_df.index = pd.to_datetime(ssh_df.index, utc=True)
    return ssh_df["zos_m"]



# -------------------- Helper methods --------------------
def _level_at(dt, ssh_df):
    """Return SSH (m) at the nearest hour to datetime dt"""
    dt = pd.to_datetime(dt, utc=True)
    return float(ssh_df.iloc[ssh_df.index.get_indexer([dt], method="nearest")[0]])


def _build_lookup(col, ssh_df):
    """
    Return {system:index -> water_level_m} for every image in the collection
    """
    indices = col.aggregate_array("system:index").getInfo()
    times   = col.aggregate_array("system:time_start").getInfo()

    return {
        idx: _level_at(pd.to_datetime(t, unit="ms", utc=True), ssh_df)
        for idx, t in zip(indices, times)
    }


def _get_bin_label(tidal_height):
    for i in range(len(BIN_LABELS)):
        if BIN_EDGES[i] <= tidal_height < BIN_EDGES[i + 1]:
            return BIN_LABELS[i]
    return "NA"



# ---------- Append Sea Surface Height to GEE collection ----------
def append_ssh_height(col):
    """Add 'tidal_height_m' property to every image in the collection."""
    ssh_df      = load_cmems_data()
    lookup      = _build_lookup(col, ssh_df)
    ee_dict     = ee.Dictionary(lookup)

    # Use img.get("system:index") — reliable for both named and computed collections.
    return col.map(lambda img: img.set(
        "tidal_height_m", ee.Number(ee_dict.get(img.get("system:index"), -999))
    ))


def append_ssh_bins(col):
    """Add 'tidal_bin' string property based on tidal stage."""
    ssh_df      = load_cmems_data()
    lookup      = _build_lookup(col, ssh_df)
    bin_lookup  = {idx: _get_bin_label(h) for idx, h in lookup.items()}
    ee_bin_dict = ee.Dictionary(bin_lookup)

    return col.map(lambda img: img.set(
        "tidal_bin", ee_bin_dict.get(img.get("system:index"), "unknown")
    ))



# ---------- Filter strategies ----------
def filter_window(col, bound=TIDAL_WINDOW_M):
    """Option A: keep only images within +/- bound metres of MSL"""
    return col.filter(
        ee.Filter.And(
            ee.Filter.gte("tidal_height_m", -bound),
            ee.Filter.lte("tidal_height_m",  bound),
            ee.Filter.neq("tidal_height_m",  -999),
        )
    )


def filter_bin(col, bin_label):
    """Option B: keep only images belonging to a specific tidal bin"""
    return col.filter(ee.Filter.eq("tidal_bin", bin_label))



# ---------- Diagnostics ----------
if __name__ == "__main__":
    ssh = load_cmems_data()
    print(f"SSH series: {len(ssh)} values  {ssh.index[0].date()} → {ssh.index[-1].date()}")
    print(f"  min {ssh.min():.2f} m  |  max {ssh.max():.2f} m  |  mean {ssh.mean():.2f} m")