# ET Downscaling — Fundación River Basin

Reproducible workflow for estimating and spatially downscaling MODIS evapotranspiration in the Fundación River Basin, Colombia, using Sentinel-2, Sentinel-1, ERA5-Land, and CHIRPS.

**Authors:** Cristian Montes-Chaura and Manuel Coy Pertuz.

The current workflow was rebuilt after a methodological diagnostic of the original implementation. Training is performed at MODIS support, while the 20 m grid is used only for spatial prediction. This avoids treating fine pixels as independent ET observations and prevents unsupported claims of validation at 20 m.

## Methodological workflow

```text
MODIS MOD16A2GF ET
        +
Sentinel-2 predictors
Sentinel-1 predictors
ERA5-Land + CHIRPS
        ↓
Station × MODIS footprint × MODIS period
        ↓
Kc = ET_MODIS / ETo
        ↓
Spatially blocked validation
        ↓
Random Forest
300 trees · 25 common predictors
        ↓
Kc prediction on a 20 m grid
        ↓
Fine-scale ET spatial pattern
        ↓
3-pass MODIS reconciliation
        ↓
DI / AOA / eligible
        ↓
Optional 4-band GeoTIFF
```

## Spatial and temporal support

* MODIS MOD16A2GF v6.1 ET is the coarse target.
* One training observation represents one station × one MODIS footprint × one MODIS period.
* Fine Sentinel-2 and Sentinel-1 pixels are not treated as independent ET observations.
* Sentinel-2 defines the 20 m production grid.
* Sentinel-1 R077 ascending observations provide VV, VH, and VV−VH spatial predictors.
* ERA5-Land and CHIRPS retain their coarse atmospheric support.
* Each output map represents accumulated ET for one MODIS composite, normally 8 days.
* The 20 m grid is a prediction scale, not an independently validated ET observation scale.

## Production configuration

| Component              | Current choice                          |
| ---------------------- | --------------------------------------- |
| ET target              | `MODIS/061/MOD16A2GF`                   |
| Target variable        | `Kc_target = ET_MODIS / ETo`            |
| Optical source         | Sentinel-2 SR Harmonized                |
| S2 cloud QA            | Cloud Score+ `cs_cdf >= 0.60`           |
| Production grid        | 20 m                                    |
| Sentinel-1             | relative orbit 77, ascending            |
| Radar composite        | median                                  |
| Meteorology            | ERA5-Land + CHIRPS                      |
| Final model            | Random Forest, 300 trees, 25 predictors |
| Primary validation     | spatial blocks (~10 km)                 |
| MODIS reconciliation   | 3 passes                                |
| Conservation tolerance | 0.01 mm per MODIS period                |

## Model performance

The final Sentinel-2 training population contains 349 station-period observations with optical coverage ≥ 90%.

Spatial validation:

| Model                     |   n |     R² |  RMSE |   MAE |   Bias |    KGE |
| ------------------------- | --: | -----: | ----: | ----: | -----: | -----: |
| RF common, 25 predictors  | 349 |  0.235 | 0.285 | 0.213 | -0.017 |  0.250 |
| RF S2 full, 31 predictors | 349 |  0.217 | 0.289 | 0.214 | -0.029 |  0.231 |
| Global mean               | 349 | -0.138 | 0.348 | 0.264 | -0.010 | -0.686 |
| MODIS persistence         | 349 |  0.463 | 0.239 | 0.154 |  0.011 |  0.724 |

The 25-predictor RF was retained as the production model to generate predictor-conditioned subpixel spatial variability with a parsimonious predictor set.

In production, `Kc_raw` supplies relative subpixel weights. The current map is
not calculated by simply multiplying a 20 m Kc field by ETo. For every eligible
MODIS parent, the initial reconstruction is:

```text
ET_fine,i = Kc_filled,i * ET_MODIS / mean_parent(Kc_filled)
```

Three proportional passes then reconcile the non-nested MODIS and UTM grids.
ETo defines the training target; MODIS fixes the final coarse ET magnitude.

The stronger MODIS persistence baseline is an important limitation: temporal persistence predicts the coarse-scale target better under spatial validation, but it cannot generate spatial variability within a MODIS footprint.

Field comparisons are diagnostic and do not constitute independent validation of true ET at 20 m.

## Area of applicability

Prediction support is evaluated using predictor-space distance and an Area of Applicability (AOA).

Production outputs distinguish:

* `eligible = 0`: required prediction support is unavailable.
* `eligible = 1`, `AOA = 0`: prediction is outside the represented predictor domain.
* `eligible = 1`, `AOA = 1`: prediction is inside the represented predictor domain.

AOA describes model applicability, not prediction accuracy.

## Reproducible execution

Create the environment and install the package:

```bash
conda env create -f environment-lock.yml
conda activate et-fundacion
pip install -e .
```

Authenticate Earth Engine:

```bash
earthengine authenticate
```

Run or resume the complete accepted workflow:

```bash
python scripts/run_pipeline.py --project <EE_PROJECT_ID>
```

The pipeline reuses existing raw exports, training products, fitted models, validation results, and QA outputs when available.

The exact stage order, required artifact contract, expected row counts, and
rebuild semantics are documented in [`docs/WORKFLOW_2021_2023.md`](docs/WORKFLOW_2021_2023.md).
The derived-artifact retention and reproducibility classification is recorded
in [`docs/ARTIFACT_INVENTORY_2021_2023.md`](docs/ARTIFACT_INVENTORY_2021_2023.md).

After a complete run, verify every local checkpoint without contacting Earth
Engine:

```bash
python scripts/audit_reproducibility.py --strict
```

The accepted QA sequence includes a bounded native-MODIS test of real fine
fill and conservation across multiple parents:

```bash
python scripts/qa_conservative_reconciliation.py --project <EE_PROJECT_ID>
```

Useful options:

```bash
python scripts/run_pipeline.py --project <EE_PROJECT_ID> --no-map
python scripts/run_pipeline.py --project <EE_PROJECT_ID> --rebuild-model
python scripts/run_pipeline.py --project <EE_PROJECT_ID> --rebuild-all
```

A single MODIS-period product can also be evaluated or exported directly after the production model has been generated:

```bash
python scripts/run_et_prediction.py \
    --project <EE_PROJECT_ID> \
    --period-start YYYY-MM-DD \
    --export
```

The requested date must correspond to an available MODIS period start. A period without the required Sentinel-1 R077 ascending acquisition is rejected; no silent fallback is applied.

The optional GeoTIFF contains four bands:

```text
ET_mm_period
DI
AOA
eligible
```

Generated data, fitted models, validation outputs, and exported products are written under `outputs/`, which is excluded from Git because these files are reproducible and may be large.

## Key project structure

```text
26-et-downscaling-fundacion/
├── config/
│   └── fvc_endmembers.json
├── data/
│   ├── boundaries/
│   │   └── fundacion_basin.geojson
│   ├── stations/
│   │   └── fundacion_stations.geojson
│   └── field/
│       └── field_etgage.csv
├── docs/
│   └── decisions/
├── scripts/
│   ├── export_meteorology_data.py
│   ├── export_satellite_data.py
│   ├── build_training_dataset.py
│   ├── train_s2_kc_models.py
│   ├── analyze_field_diagnostics.py
│   ├── validate_field_downscaling.py
│   ├── smoke_test_spatial_prediction.py
│   ├── run_et_prediction.py
│   └── run_pipeline.py
├── src/
│   └── et_downscaling/
│       ├── config.py
│       ├── dataset.py
│       ├── model_spec.py
│       ├── model_transfer.py
│       ├── production.py
│       ├── aoa.py
│       ├── modis.py
│       ├── sentinel1.py
│       ├── sentinel2.py
│       └── reference_et_local.py
├── environment.yml
├── pyproject.toml
└── README.md
```

## Methodological decision records

The `docs/decisions/` directory documents the main methodological changes and their justification, including MODIS QC, optical-source selection, the Kc target, Sentinel-1 configuration, spatial support, meteorological processing, FVC calibration, thermal predictors, model selection, and other diagnostic decisions.

Historical implementations and removed experiments remain recoverable through Git history rather than being retained in the active production workflow.

## License

GNU General Public License v3.0 (`GPL-3.0-only`).
