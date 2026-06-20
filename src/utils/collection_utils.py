import ee
import pandas as pd

from utils.config import (
    GEO_JSON_SYLT_COMPLETE, GEO_JSON_SYLT_COASTLINE_BOUNDARY,
    START_DATE, END_DATE,
    S1_COLLECTION, S1_PASS, S1_ORBIT,
    S2_COLLECTION, MAX_CLOUD_PERC, MONTH_NAMES,
)

from utils.tidal_utils import append_ssh_height, append_ssh_bins, filter_bin



# -------------------- Helpers Methods --------------------

def _get_aoi(option:str="Complete") -> ee.Geometry:
    if option == "Complete":
        return ee.Geometry(GEO_JSON_SYLT_COMPLETE["features"][0]["geometry"])
    return None


def _get_calibration_strip() -> ee.Geometry:
    """Intertidal coastal strip for Otsu histogram calibration.
    Falls back to full AOI if the coastal boundary is not yet configured.
    """
    if GEO_JSON_SYLT_COASTLINE_BOUNDARY is None:
        return _get_aoi()
    return ee.Geometry(GEO_JSON_SYLT_COASTLINE_BOUNDARY["features"][0]["geometry"])


def _get_base_collection(collection_id:str, aoi:ee.Geometry, start:str, end:str) -> ee.ImageCollection:
    """Base collection filtered to AOI and date range"""
    return (ee.ImageCollection(collection_id)
            .filterBounds(aoi)
            .filterDate(ee.Date(start), ee.Date(end)))


def _mosaic_by_day(col: ee.ImageCollection) -> ee.ImageCollection:
    """
    Mosaic same-day, same-orbit slices into one image.

    Groups by (date, relativeOrbitNumber_start) so slices from different orbits
    on the same calendar day are never merged (which would mix SAR geometry).
    For collections without that property (e.g. S2), falls back to date-only grouping.
    Each output image inherits all properties from the earliest slice in its group.
    """
    if col.size().getInfo() == 0:
        return ee.ImageCollection([])

    meta = ee.Dictionary({
        "times":  col.aggregate_array("system:time_start"),
        "orbits": col.aggregate_array("relativeOrbitNumber_start"),
    }).getInfo()

    times  = meta["times"]
    orbits = meta["orbits"]
    orbit_aware = any(o is not None for o in orbits)

    seen  = set()
    pairs = []
    for t, orb in zip(times, orbits):
        date_str = pd.to_datetime(t, unit="ms", utc=True).strftime("%Y-%m-%d")
        key = (date_str, orb if orbit_aware else None)
        if key not in seen:
            seen.add(key)
            pairs.append((date_str, orb if orbit_aware else None))

    mosaic_list = []
    for date_str, orbit in pairs:
        date_start = ee.Date(date_str)
        date_end   = date_start.advance(1, "day")
        slices = col.filterDate(date_start, date_end)
        if orbit is not None:
            slices = slices.filter(ee.Filter.eq("relativeOrbitNumber_start", orbit))
        base   = slices.sort("system:time_start").first()
        mosaic = ee.Image(slices.mosaic().copyProperties(base, base.propertyNames()))
        mosaic_list.append(mosaic)

    return ee.ImageCollection.fromImages(mosaic_list)


def _attach_tidal(col:ee.ImageCollection) -> ee.ImageCollection:
    """Attach tidal height and bin label to every image"""
    col = append_ssh_height(col)
    col = append_ssh_bins(col)
    return col



# -------------------- Sentinel-1 --------------------

def get_collection_s1(start: str = START_DATE, end: str = END_DATE,
                      pass_dir: str = S1_PASS, orbit: int | None = S1_ORBIT) -> ee.ImageCollection:
    """
    Build a tidal-annotated S1 collection mosaiced per (day, orbit).

    orbit=<int>  — fixed orbit (default S1_ORBIT=139); also restricts to pass_dir.
                   Use for timeseries where cross-date radiometric consistency is required.
    orbit=None   — all orbits and pass directions included.
                   Use for event analysis where the best orbit is selected per storm.
    """
    aoi = _get_aoi()
    col = _get_base_collection(S1_COLLECTION, aoi, start, end)
    col = col.filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
    if orbit is not None:
        col = col.filter(ee.Filter.eq("orbitProperties_pass", pass_dir))
        col = col.filter(ee.Filter.eq("relativeOrbitNumber_start", orbit))

    col = _mosaic_by_day(col)
    col = _attach_tidal(col)

    orbit_str = str(orbit) if orbit is not None else "all"
    print(f"S1 collection: {col.size().getInfo()} mosaiced images")
    print(f"\tpass={pass_dir if orbit is not None else 'all'}  orbit={orbit_str}  {start} → {end}")
    return col



# -------------------- Sentinel-2 --------------------

def get_collection_s2(start:str=START_DATE, end:str=END_DATE,
                      max_cloud:int=MAX_CLOUD_PERC) -> ee.ImageCollection:
    aoi = _get_aoi()

    col = _get_base_collection(S2_COLLECTION, aoi, start, end)
    col = col.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud))

    # Same mosaicking logic as S1: adjacent S2 tiles over Sylt → one image per day.
    col = _mosaic_by_day(col)
    col = _attach_tidal(col)

    print(f"S2 collection: {col.size().getInfo()} mosaiced images")
    print(f"\tcloud<{max_cloud}% \n\t from {start} to {end}")
    return col



# -------------------- Get S1 Image Numbers per Orbit --------------------
 
def inspect_orbits_by_bins(start:str=START_DATE, end:str=END_DATE, pass_dir:str=S1_PASS) -> dict:
    col = _get_base_collection(S1_COLLECTION, _get_aoi(), start, end)
    col = col.filter(ee.Filter.eq("orbitProperties_pass", pass_dir))
    col = col.filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
 
    orbits = col.aggregate_array("relativeOrbitNumber_start").distinct().getInfo()
 

    per_orbit = [_mosaic_by_day(col.filter(ee.Filter.eq("relativeOrbitNumber_start", orbit))) 
                 for orbit in sorted(orbits)]
    orbit_mosaics = per_orbit[0]
    for col in per_orbit[1:]:
        orbit_mosaics = orbit_mosaics.merge(col)
 
    orbit_mosaics = _attach_tidal(orbit_mosaics)
 
    values = orbit_mosaics.reduceColumns(
        reducer=ee.Reducer.frequencyHistogram().group(groupField=1, groupName="orbit"),
        selectors=["tidal_bin", "relativeOrbitNumber_start"],).getInfo()
 
    print("\n","-"*50)
    print(f"S1 (Mosaic) Image Availability by Orbit and Tidal Bin ({pass_dir})")
    for group in values["groups"]:
        orbit      = group["orbit"]
        bin_counts = group["histogram"]
        total      = sum(bin_counts.values())
        print(f"\nOrbit {orbit} with {total} total) imgaes:")
        for bin_name in sorted(bin_counts):
            print(f"\t{bin_name}: {bin_counts[bin_name]} images")
 
    return values


def assess_s1_availability(tidal_bin: str = "near_msl", start: str = START_DATE, end: str = END_DATE) -> None:
    """
    Monthly availability matrices for Sentinel-1 over the Sylt AOI.
    Prints two tables (rows=months, cols=orbits): one without and one with the tidal filter.
    """
    aoi = _get_aoi()
    base = (_get_base_collection(S1_COLLECTION, aoi, start, end)
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV")))

    print("Discovering S1 orbits…")
    orbit_list = base.aggregate_array("relativeOrbitNumber_start").distinct().sort().getInfo()

    raw_counts   = {}
    tidal_counts = {}

    for orbit in orbit_list:
        orbit_col = base.filter(ee.Filter.eq("relativeOrbitNumber_start", orbit))
        pass_dir  = orbit_col.first().get("orbitProperties_pass").getInfo()
        label = f"Orbit {orbit} ({pass_dir[:3]})"

        mosaiced = _mosaic_by_day(orbit_col)
        times    = mosaiced.aggregate_array("system:time_start").getInfo()
        dates    = pd.to_datetime(times, unit="ms", utc=True)
        raw_counts[label] = pd.Series(dates.month).value_counts()
        print(f"  {label}: {len(times)} scenes")

        tidal_col = filter_bin(_attach_tidal(mosaiced), tidal_bin)
        times_t   = tidal_col.aggregate_array("system:time_start").getInfo()
        dates_t   = pd.to_datetime(times_t, unit="ms", utc=True)
        tidal_counts[label] = pd.Series(dates_t.month).value_counts()
        print(f"    → {tidal_bin}: {len(times_t)} scenes")

    idx = range(1, 13)

    def build_df(counts_dict):
        df = pd.DataFrame(index=idx)
        for label, counts in counts_dict.items():
            df[label] = counts.reindex(idx, fill_value=0).astype(int)
        df["Total"] = df.sum(axis=1)
        df.loc["Total"] = df.sum()
        df.index = MONTH_NAMES + ["Total"]
        return df

    df_raw   = build_df(raw_counts)
    df_tidal = build_df(tidal_counts)

    print(f"\n{'=' * 70}")
    print(f"S1 Availability  ({start[:4]}–{end[:4]}, Sylt AOI, VV polarisation)")
    print()
    print("No tidal filter")
    print(df_raw.to_string())
    print()
    print(f"{tidal_bin} tidal filter")
    print(df_tidal.to_string())
    print("=" * 70)
