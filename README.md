# ET Downscaling Dataset Pipeline

Reproducible Python and Google Earth Engine workflow for constructing a multiscale evapotranspiration (ET) dataset by integrating MODIS ET observations with Sentinel-2, Sentinel-1, ERA5-Land, and CHIRPS predictors for subsequent ET downscaling.

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

The predictor set includes Sentinel-2 surface reflectance, NDVI, NDMI, NDWI, Sentinel-1 backscatter metrics, meteorological variables, precipitation, and MODIS ET targets.

## Multiscale structure

Each valid MODIS observation generates two records:

- `footprint`: predictors summarized over the MODIS ET footprint.
- `local_60m`: predictors summarized over a local 60 × 60 m area.

The ET value associated with `local_60m` remains the parent MODIS footprint ET observation and is **not an independent 60 m ET measurement**.

For model development, records with `scale == "footprint"` should be used for training. The `local_60m` records are intended for subsequent model application.

## Requirements

- Python ≥ 3.12
- Google Earth Engine account
- Google Cloud Project with Earth Engine access

The default sample asset is:

```text
projects/ee-change/assets/ETP_samples
```

The sample asset and processing parameters can be changed in:

```text
src/et_downscaling/config.py
```

## Installation and usage

### Local environment

Clone the repository:

```bash
git clone https://github.com/CrsMontes/26-et-downscaling-fundacion.git
cd 26-et-downscaling-fundacion
```

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

Run the dataset pipeline:

```bash
python scripts/build_training_dataset.py
```

The workflow will request a Google Cloud Project ID with Earth Engine access.

### Google Colab

Start from a new Colab runtime and run:

```python
!git clone https://github.com/CrsMontes/26-et-downscaling-fundacion.git
%cd 26-et-downscaling-fundacion
%pip install .
```

Verify that the package was installed correctly:

```python
import et_downscaling
print(et_downscaling.__file__)
```

Authenticate Google Earth Engine:

```python
import ee
ee.Authenticate()
```

Run the pipeline:

```python
%run scripts/build_training_dataset.py
```

When prompted, enter a Google Cloud Project ID available to your Earth Engine account.

Generated datasets are written to:

```text
outputs/
```

In Google Colab, the `outputs/` directory is temporary. Download or copy any required results before the runtime is deleted.

## Configuration

The main processing parameters are defined in:

```text
src/et_downscaling/config.py
```

This includes:

- Analysis start and end dates
- Spatial resolution and CRS
- Sentinel-2 quality threshold
- Sentinel-1 orbit configuration
- ERA5-Land search radius
- Sample asset
- Output filename

`START_DATE` is inclusive and `END_DATE` is exclusive.

## Project structure

```text
26-et-downscaling-fundacion/
├── .github/
│   └── CODEOWNERS
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
├── .gitignore
├── environment.yml
├── LICENSE
├── pyproject.toml
└── README.md
```

## License

This project is licensed under the GNU General Public License v3.0 (`GPL-3.0-only`). See the `LICENSE` file for details.

Copyright (C) 2026 C. Montes-Chaura.