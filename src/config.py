# GEE Config --------------------------------------------------
GEE_PROJECT_ID = 'eoda-2026-ee-pfingsten'

# GEO Json ----------------------------------------------------
GEO_JSON_SYLT_COMPLETE = {
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {},
      "geometry": {
        "coordinates": [
          [
            [8.247309267123967, 54.72748987748213],
            [8.49587975055735,  54.72748987748213],
            [8.49587975055735,  55.070835539021544],
            [8.247309267123967, 55.070835539021544],
            [8.247309267123967, 54.72748987748213]
          ]
        ],
        "type": "Polygon"
      }
    }
  ]
}


# Date range --------------------------------------------------
START_DATE  = "2017-01-01"
END_DATE    = "2024-12-31"


# SAR Config --------------------------------------------------
S1_COLLECTION = "COPERNICUS/S1_GRD"
S1_PASS = "DESCENDING"
S1_ORBIT = None   #TODO Set after inspecting via .aggregate_array("relativeOrbitNumber_start")


# Optical Config ----------------------------------------------
S2_COLLECTION    = "COPERNICUS/S2_SR_HARMONIZED"
MAX_CLOUD_PERC   = 20
OPTICAL_MONTHS   = [5, 6, 7, 8, 9] #TODO 


# Tidal Control -----------------------------------------------
# Using Copernicus Marine Service (CMEMS)
# Dataset : cmems_mod_nws_phy-ssh_my_7km-2D_PT1H-i

CMEMS_DATASET  = "cmems_mod_nws_phy-ssh_my_7km-2D_PT1H-i"
# Variable: zos = Sea surface height above geoid (m)
CMEMS_VARIABLE = "zos" 

# Bounding box of sylt
CMEMS_LON_MIN  =  8.3
CMEMS_LON_MAX  =  8.6
CMEMS_LAT_MIN  = 54.9
CMEMS_LAT_MAX  = 55.1

# CSV cache for the downloaded sea surface height (ssh)
CMEMS_DATA_PATH = "data/cmems/sea_surface_heigth_sylt.csv"

# Tidal filter ------------------------------------------------
# Option A – window = reject images outside +/- TIDAL_WINDOW_M of MSL
TIDAL_WINDOW_M = 0.25   # metres either side of MSL (Option A)

# Option B – binning = group images into tidal bins and select bin 
BIN_EDGES  = [-3.0, -0.75, -0.25, 0.25, 0.75, 3.0]
BIN_LABELS = ["very_low", "low_mid", "near_msl", "high_mid", "very_high"]

