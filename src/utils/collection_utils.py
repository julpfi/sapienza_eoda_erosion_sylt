import ee

from utils.config import (
    GEO_JSON_SYLT_COMPLETE, GEO_JSON_SYLT_COASTLINE_BOUNDARY,
    START_DATE, END_DATE,
    S1_COLLECTION, S1_PASS, S1_ORBIT,
    S2_COLLECTION, MAX_CLOUD_PERC, OPTICAL_MONTHS,
)

from utils.tidal_utils import append_ssh_height, append_ssh_bins



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


def _mosaic_by_day(col:ee.ImageCollection) -> ee.ImageCollection:
    """
    Mosaic all image slices that fall on the same calendar day into one image
    - Works for both S1 (two spatial slices from the same pass) and S2 (adjacenttiles)
    - The base image is the earliest slice of the day
    """
    dates = (col.aggregate_array("system:time_start")
               .map(lambda t: ee.Date(t).format("YYYY-MM-dd"))
               .distinct()
               .getInfo())

    mosaic_list = []
    for date in dates:
        date_start = ee.Date(date)
        date_end   = date_start.advance(1, "day")

        day_slices = col.filterDate(date_start, date_end)
        base_image = day_slices.sort("system:time_start").first()

        mosaic = ee.Image(day_slices.mosaic().copyProperties(base_image, base_image.propertyNames()))
        mosaic_list.append(mosaic)

    return ee.ImageCollection.fromImages(mosaic_list)


def _attach_tidal(col:ee.ImageCollection) -> ee.ImageCollection:
    """Attach tidal height and bin label to every image"""
    col = append_ssh_height(col)
    col = append_ssh_bins(col)
    return col



# -------------------- Sentinel-1 --------------------

def get_collection_s1(start:str=START_DATE, end:str=END_DATE, pass_dir:str=S1_PASS, orbit:int=S1_ORBIT) -> ee.ImageCollection:
    aoi = _get_aoi()

    col = _get_base_collection(S1_COLLECTION, aoi, start, end)
    col = col.filter(ee.Filter.eq("orbitProperties_pass", pass_dir))
    col = col.filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
    col = col.filter(ee.Filter.eq("relativeOrbitNumber_start", orbit))

    col = _mosaic_by_day(col)
    col = _attach_tidal(col)

    print(f"S1 collection: {col.size().getInfo()} mosaiced images")
    print(f"\tpass={pass_dir} \n\torbit={orbit} \n\tfrom {start} to {end}")
    return col



# -------------------- Sentinel-2 --------------------

def get_collection_s2(start:str=START_DATE, end:str=END_DATE,
                      months:str=OPTICAL_MONTHS, max_cloud:int=MAX_CLOUD_PERC) -> ee.ImageCollection:
    aoi = _get_aoi()

    col = _get_base_collection(S2_COLLECTION, aoi, start, end)
    col = col.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud))

    if months:
        col = col.filter(ee.Filter.calendarRange(min(months), max(months), "month"))

    # Same mosaicking logic as S1: adjacent S2 tiles over Sylt → one image per day.
    col = _mosaic_by_day(col)
    col = _attach_tidal(col)

    print(f"S2 collection: {col.size().getInfo()} mosaiced images")
    print(f"\tmonths={months} \n\tcloud<{max_cloud}% \n\t from {start} to {end}")
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
 