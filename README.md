# ET Downscaling — Fundación River Basin

Reproducible workflow for estimating and downscaling MODIS evapotranspiration
in the Fundación River Basin, Colombia, using Sentinel-2, Sentinel-1,
ERA5-Land, and CHIRPS.

The current production design was rebuilt after a methodological diagnostic of
the original workflow. The active pipeline deliberately separates coarse-target
training from fine-grid prediction to avoid pseudoreplication and false claims
of 20 m validation.

## Current production design

```text
MODIS MOD16A2GF ET
        +
Sentinel-2 predictors
Sentinel-1 predictors
ERA5-Land + CHIRPS
        ↓
MODIS-footprint training table
        ↓
Kc = ET_MODIS / ETo
        ↓
Spatially blocked model validation
        ↓
Selected RF: 25 predictors
        ↓
Sentinel-2 20 m production grid
        ↓
Fine Kc spatial pattern
        ↓
MODIS-constrained ET product
```

### Spatial support

- MODIS ET is the coarse target.
- One training observation is one station × one MODIS footprint × one MODIS
  period.
- Fine optical/radar pixels inside a MODIS footprint are not treated as
  independent ET observations.
- Sentinel-2 uses a 20 m working and prediction grid.
- Sentinel-1 is temporally composited first and summarized at 10 m only when
  producing footprint-scale training statistics; 10 m is not an ET validation
  resolution.
- Meteorological variables retain coarse atmospheric support.

## Accepted primary configuration

| Component | Current choice |
|---|---|
| ET target | `MODIS/061/MOD16A2GF` |
| Target variable | `Kc_target = ET_MODIS / ETo` |
| Primary optical source | Sentinel-2 SR Harmonized |
| S2 cloud QA | Cloud Score+ `cs_cdf >= 0.60` |
| Prediction grid | 20 m |
| Sentinel-1 | relative orbit 77, ascending |
| Radar temporal composite | median |
| Meteorology | ERA5-Land + CHIRPS |
| Final model | Random Forest, 25 predictors |
| Primary validation | ~10 km spatial blocks |
| Basin boundary | `data/boundaries/fundacion_basin.geojson` |

Combined HLS S30 + L30 remains an operational 30 m alternative, but it is not
the selected primary production source.

## Final model selection

The final Sentinel-2 training population contains 349 station-period
observations at optical coverage >= 90%.

Spatial validation produced:

| Model | R2 | RMSE | MAE | KGE |
|---|---:|---:|---:|---:|
| RF common, 25 predictors | 0.233 | 0.286 | 0.214 | 0.250 |
| RF S2 full, 31 predictors | 0.225 | 0.287 | 0.213 | 0.238 |
| MODIS persistence baseline | 0.463 | 0.239 | 0.154 | 0.724 |

The 25-predictor RF is selected for parsimony. The stronger persistence
baseline is retained as an important limitation: coarse temporal persistence is
more predictive than the RF, but it does not provide subpixel spatial
information.

See `docs/decisions/11_final_kc_model.md` for the full decision record.

## Reproducible execution to the current checkpoint

Create the environment and install the package:

```bash
conda env create -f environment.yml
conda activate et-fundacion
pip install -e .
```

Authenticate Earth Engine:

```bash
earthengine authenticate
```

Export reusable meteorological inputs:

```bash
python scripts/export_meteorology_data.py
```

Export Sentinel-2 + Sentinel-1 footprint predictors:

```bash
python scripts/export_satellite_data.py --optical-source S2
```

Build the local training master:

```bash
python scripts/build_training_dataset.py --optical-source S2
```

Train and validate the candidate models and write the selected production RF:

```bash
python scripts/train_s2_kc_models.py
```

Generated data and fitted models are written under `outputs/`, which is ignored
by Git because the files are reproducible and can be large.

## Project structure

```text
26-et-downscaling-fundacion/
├── config/
│   └── fvc_endmembers.json
├── data/
│   └── boundaries/
│       └── fundacion_basin.geojson
├── docs/
│   └── decisions/
├── scripts/
│   ├── build_training_dataset.py
│   ├── export_meteorology_data.py
│   ├── export_satellite_data.py
│   ├── recalibrate_hls_fvc.py
│   └── train_s2_kc_models.py
├── src/
│   └── et_downscaling/
│       ├── config.py
│       ├── dataset.py
│       ├── hls.py
│       ├── local_training.py
│       ├── meteorology_export.py
│       ├── model_spec.py
│       ├── modis.py
│       ├── optical.py
│       ├── reference_et_local.py
│       ├── sentinel1.py
│       └── sentinel2.py
├── environment.yml
├── pyproject.toml
└── README.md
```

## Methodological decision records

The `docs/decisions/` directory records changes relative to the historical
workflow. Important corrections include MODIS QC handling, S2/HLS source
selection, the Kc target, Sentinel-1 orbit selection, spatial support,
local meteorological processing, HLS FVC recalibration, and final model
selection.

Historical diagnostic code and removed experimental modules remain recoverable
through Git history rather than being kept in the active production source tree.

## License

GNU General Public License v3.0 (`GPL-3.0-only`).
