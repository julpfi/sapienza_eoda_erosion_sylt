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

# Output Paths
OUTPUT_PLOTS = "outputs/plots/"
OUTPUT_ANIMATIONS = "outputs/animations/"

# Date range --------------------------------------------------
START_DATE  = "2017-01-01"
END_DATE    = "2024-12-31"


# Visualization Settings --------------------------------------
VIS_SAR_VV = {
    "min": -25, 
    "max": 0, 
    "palette": ["000000", "ffffff"]
}

# Binary water mask (0 = land, 1 = water)
VIS_BINARY_WATER_MASK = {
    "min": 0, 
    "max": 1, 
    "palette": ["d4d4d4", "2255aa"]
}

# Change Map
VIS_CHANGE_MAP = {
    "min": 0, 
    "max": 3,
    "palette": [
        "888888",   # 0 – Consistent land
        "00ccff",   # 1 – new water (erosion)
        "ff3333",   # 2 – new land (recovery)
        "224488",   # 3 – Consistent water
    ]
}

CHANGE_MAP_LABELS = [
    "Consistent land",
    "New water (erosion)",
    "New land (recovery)",
    "Consistent water"
]


VIS_S2_TRUE_COLOR = {
    "min": 0,
    "max": 3000,
}
 
# NDWI  = (Green – NIR) / (Green + NIR)  →  water > 0
VIS_S2_NDWI = {
    "min": -0.3,
    "max":  0.5,
    "palette": ["d4d4d4", "aaddff", "0066cc"],
}


# SAR Config --------------------------------------------------
S1_COLLECTION = "COPERNICUS/S1_GRD_FLOAT"
S1_PASS = "DESCENDING"
S1_ORBIT = 139

"""
Number of images available between "2017-01-01" and "2024-12-31" grouped by the tidal bins 
Orbit 37  (DESCENDING)
- high_mid: 87 images
- low_mid: 60 images
- near_msl: 95 images
- very_high: 17 images
- very_low: 121 images

Orbit 139 (DESCENDING)
- high_mid: 88 images
- low_mid: 60 images
- near_msl: 105 images
- very_high: 19 images
- very_low: 117 images

Orbit 15 (ASCENDING)
- high_mid: 109 images
- low_mid: 54 images
- near_msl: 87 images
- very_high: 28 images
- very_low: 99 images

Orbit 117  (ASCENDING)
- high_mid: 117 images
- low_mid: 59 images
- near_msl: 83 images
- very_high: 26 images
- very_low: 97 images
"""

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



# SAR Analyis 
EVENT_DATE_ORKAN_ZEYNEP = "2022-02-19"