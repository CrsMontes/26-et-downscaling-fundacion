# ET Downscaling Dataset Pipeline

Reproducible Python workflow for constructing a multiscale evapotranspiration dataset by integrating MODIS ET observations with Sentinel-2, Sentinel-1, ERA5-Land, and CHIRPS predictors for subsequent ET downscaling.

## Workflow

```text
MODIS ET
   +
Sentinel-2
Sentinel-1
ERA5-Land
CHIRPS
   ↓
Quality control
   ↓
Multiscale predictor extraction
   ├── MODIS footprint
   └── Local 60 × 60 m support
   ↓
ET modeling dataset
```

## Data sources

| Component | Dataset |
|---|---|
| Evapotranspiration | `MODIS/061/MOD16A2GF` |
| Optical predictors | `COPERNICUS/S2_SR_HARMONIZED` |
| Cloud quality | `GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED` |
| Radar predictors | `COPERNICUS/S1_GRD` |
| Meteorology | `ECMWF/ERA5_LAND/HOURLY` |
| Precipitation | `UCSB-CHG/CHIRPS/DAILY` |

Sentinel-2 surface reflectance is scaled before predictor calculation. The dataset includes spectral bands, NDVI, NDMI, NDWI, Sentinel-1 backscatter metrics, meteorological variables, precipitation, and ET targets.

## Multiscale structure

Each valid MODIS observation generates two records:

- `footprint`: predictors summarized over the MODIS ET footprint.
- `local_60m`: predictors summarized over a local 60 × 60 m area.

The ET value associated with `local_60m` remains the parent MODIS footprint ET observation and is **not an independent 60 m ET measurement**.

For model development, `scale == "footprint"` should be used as the training dataset. The `local_60m` records are intended for subsequent model application.

## Configuration

The main processing parameters are defined in:

```text
src/et_downscaling/config.py
```

This includes the analysis period, spatial resolution, Sentinel-2 quality threshold, Sentinel-1 orbit configuration, ERA5-Land search radius, and output filename.

`START_DATE` is inclusive and `END_DATE` is exclusive.

## Usage

Clone the repository:

```bash
git clone https://github.com/CrsMontes/26-et-downscaling-fundacion.git
cd 26-et-downscaling-fundacion
```

Create the environment and install the project:

```bash
conda env create -f environment.yml
conda activate et-fundacion
pip install -e .
```

Authenticate Google Earth Engine:

```bash
earthengine authenticate
```

Run the pipeline:

```bash
python scripts/build_training_dataset.py
```

The workflow requests a Google Cloud Project ID with access to Earth Engine. Generated datasets are written to `outputs/`.

### Google Colab

The repository can also be cloned and executed in Google Colab:

```python
!git clone https://github.com/CrsMontes/26-et-downscaling-fundacion.git
%cd 26-et-downscaling-fundacion
!pip install -q earthengine-api==1.7.38
!pip install -e .
```

Then authenticate Earth Engine with your own Google account before running the pipeline.

## Project structure

```text
26-et-downscaling-fundacion/
├── scripts/
│   └── build_training_dataset.py
├── src/
│   └── et_downscaling/
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
├── LICENSE
└── README.md
```

## License

Licensed under the GNU General Public License v3.0 (`GPL-3.0-only`).

Copyright (C) 2026 C. Montes-Chaura.