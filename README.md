# ET Downscaling Dataset Pipeline

Python and Google Earth Engine workflow for building a multiscale dataset for evapotranspiration (ET) downscaling.

The pipeline integrates MODIS evapotranspiration with Sentinel-2, Sentinel-1, ERA5-Land, and CHIRPS predictors at two spatial supports:

- MODIS footprint
- Local 60 × 60 m window

The resulting dataset is intended for statistical and machine-learning experiments for fine-resolution ET estimation.

## Objective

Develop a reproducible workflow to construct a multiscale evapotranspiration dataset by integrating MODIS ET observations with Sentinel-2, Sentinel-1, ERA5-Land, and CHIRPS predictors for subsequent ET downscaling.

## Workflow

```text
Station samples
      │
      ▼
MODIS ET observations
      │
      ▼
Valid observation periods
      │
      ├── Sentinel-2
      ├── Sentinel-1
      ├── ERA5-Land
      └── CHIRPS
            │
            ▼
    Predictor statistics
            │
    ┌───────┴───────┐
    ▼               ▼
MODIS footprint   Local 60 × 60 m
    │               │
    └───────┬───────┘
            ▼
    Multiscale dataset
```

Only observations with valid MODIS, Sentinel-2, Sentinel-1, and complete meteorological information are retained.

## Data sources

| Source | Google Earth Engine collection | Role |
|---|---|---|
| MODIS | `MODIS/061/MOD16A2GF` | ET target and coarse spatial support |
| Sentinel-2 | `COPERNICUS/S2_SR_HARMONIZED` | Optical predictors |
| Cloud Score+ | `GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED` | Sentinel-2 quality masking |
| Sentinel-1 | `COPERNICUS/S1_GRD` | Radar predictors |
| ERA5-Land | `ECMWF/ERA5_LAND/HOURLY` | Meteorological predictors |
| CHIRPS | `UCSB-CHG/CHIRPS/DAILY` | Precipitation predictors |

Sentinel-2 surface reflectance is converted from the source scaled values using a factor of `0.0001` before compositing and predictor calculation.

Because ERA5-Land contains only land pixels, coastal stations may fall in cells without valid values. The workflow therefore identifies the **nearest valid ERA5-Land pixel** within a predefined search radius and reuses that support for all observations from the station. Sampling coordinates and distance are retained for traceability.

CHIRPS precipitation remains sampled at the original station location.

## Main variables and units

| Variable group | Variables | Unit |
|---|---|---|
| Evapotranspiration | `ET_mm_period` | mm per MODIS period |
| Evapotranspiration | `ET_mm_day` | mm day⁻¹ |
| Sentinel-2 bands | Blue, Green, Red, Red Edge, NIR, SWIR | Dimensionless surface reflectance |
| Spectral indices | NDVI, NDMI, NDWI | Dimensionless |
| Sentinel-1 backscatter | VV, VH, VV − VH | dB |
| Sentinel-1 incidence angle | Angle | degrees |
| Air temperature | Mean, maximum | °C |
| Vapor pressure deficit | Mean, maximum | kPa |
| Solar radiation | Mean daily incoming radiation | MJ m⁻² day⁻¹ |
| Wind speed | Mean wind speed | m s⁻¹ |
| Precipitation | MODIS period and previous 30 days | mm |
| Spatial support / distance | Footprint and ERA5-Land support metadata | m or m² |
| Spatial coverage | Sentinel-1 and Sentinel-2 coverage | % |
| Coordinates | Longitude, latitude | decimal degrees |

Optical and radar predictors are summarized using spatial **mean** and **standard deviation**. Standard deviations retain the units of their corresponding variables.

## Multiscale structure

Each valid observation produces two records.

### `footprint`

Predictor statistics are calculated over the MODIS spatial footprint.

These records contain predictors and ET targets at compatible coarse spatial support and are intended for **model training and validation**.

### `local_60m`

Predictor statistics are calculated over a local 60 × 60 m window centered on the station.

These records contain fine-scale predictor information intended for subsequent **model application and downscaling**.

> **Important:** the ET value associated with a `local_60m` record remains the parent MODIS ET observation. It is not an independently observed 60 m ET value. Therefore, `local_60m` records should not be treated as independent training labels.

Meteorological variables are associated with the observation period and are shared between the `footprint` and `local_60m` records.

## Quality control

The final dataset requires:

- Valid MODIS ET
- Valid Sentinel-2 coverage
- Valid Sentinel-1 coverage
- Complete ERA5-Land temporal information
- Complete CHIRPS temporal information
- Complete meteorological variables
- Complete optical and radar predictor statistics

Quality-control and spatial-support metadata are retained in the output dataset.

## Current validated configuration

```text
Period:                 2021-01-01 to 2024-01-01
Analysis CRS:           EPSG:32618
Analysis scale:         20 m
Local support:          60 × 60 m
S2 Cloud Score+:        cs_cdf ≥ 0.60
S1 orbit pass:          ASCENDING
S1 relative orbit:      77
ERA5 search radius:     50 km
```

The validated run produced:

```text
Valid observations:     304
Footprint rows:          304
Local 60 m rows:         304
Total rows:              608
Output columns:           83
```

These values describe the current validated study configuration and are not fixed software requirements.

## Project structure

```text
26-et-downscaling-fundacion/
├── scripts/
│   └── build_training_dataset.py
├── src/
│   └── et_downscaling/
│       ├── __init__.py
│       ├── config.py
│       ├── dataset.py
│       ├── export.py
│       ├── meteorology.py
│       ├── modis.py
│       ├── schema.py
│       ├── sentinel1.py
│       ├── sentinel2.py
│       └── spatial.py
├── environment.yml
├── pyproject.toml
├── .gitignore
└── README.md
```

The `src/et_downscaling/` package contains the processing logic, while `scripts/build_training_dataset.py` provides the main executable workflow.

## Installation

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate et-fundacion
```

Install the project in editable mode:

```bash
pip install -e .
```

Authenticate Google Earth Engine:

```bash
earthengine authenticate
```

A Google Cloud Project with Earth Engine access is required.

## Running the pipeline

From the repository root:

```bash
python scripts/build_training_dataset.py
```

The Google Cloud Project ID is requested interactively.

Generated datasets are written to:

```text
outputs/
```

The `outputs/` directory is excluded from Git because it contains reproducible generated products rather than source code.

## Development status

**Completed**

- Multisource dataset construction
- MODIS ET target processing
- Sentinel-2 optical predictors and spectral indices
- Sentinel-1 radar predictors
- ERA5-Land and CHIRPS meteorology
- Coastal ERA5-Land support handling
- Multiscale spatial statistics
- Quality-control checks
- Partitioned processing and export
- Final dataset validation

## License

Copyright (C) 2026 C. Montes-Chaura.
