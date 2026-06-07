import ee
from utils.config import GEE_PROJECT_ID, GEO_JSON_SYLT_COMPLETE



def _connect():
    """Authenticate + initialise GEE. Falls back to browser login if no cached credentials."""
    try:
        ee.Initialize(project=GEE_PROJECT_ID)
    except Exception:
        print("No cached credentials – opening browser for authentication...")
        ee.Authenticate()
        ee.Initialize(project=GEE_PROJECT_ID)


def _check():
    """Connection check: print complete sylt AOI area"""
    aoi = ee.Geometry(GEO_JSON_SYLT_COMPLETE["features"][0]["geometry"])
    area_km2 = aoi.area(maxError=10).divide(1e6).getInfo()
    print(f"Sylt Complete AOI area : {area_km2:.1f} km^2")
    print("Connection working...\n")


# -------------- GEE Initialization Entry Point ----------------
def init_gee():
    """Connect to GEE and run a connection check"""
    _connect()
    _check()
    print(f"GEE connected to project: {GEE_PROJECT_ID}")


# ---------- Main Method for checking gee_utils.py ----------
if __name__ == "__main__":
    init_gee()