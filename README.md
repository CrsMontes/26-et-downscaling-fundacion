# ET downscaling - Fundacion River basin - Virtual Station

This repository's `main` branch contains the Virtual Station V5 design selected
for the basin-wide ET downscaling workflow. The previous Field Station / Stable5
state is preserved separately in the `field-station-stable` branch checkout and
its own external workspace.

## Frozen Virtual Station V5 state

- Ridge25, alpha = 1.
- Target: `Kc_target = ET_MODIS / ETo`.
- 25 predictors: 16 Sentinel-2 optical variables, 5 ERA5-Land variables and
  4 temporal harmonics.
- Whole-basin sequential random GE90 support design.
- Seed: 42.
- 10 native MODIS supports in 10 fixed UTM 10 km blocks.
- 1454 training rows.
- Equal-weight standardized Euclidean AOA threshold:
  `0.8007328515330622`.
- Final mapped periods preserved locally:
  `2020-03-13`, `2021-11-25`, `2022-03-30`.

Virtual supports are MODIS-derived training supports. They are not field
stations and do not provide independent 20 m validation.

## External workspace

Generated data remain outside Git:

```text
ET_fundacion_workspace_virtual_station/
  current/      operational V5 raster/product layout
  training/     frozen selection, extraction caches and master
  evaluation/   QA, CV, transfer, field-proxy and basin-coverage evidence
  final/        reserved for promoted publication deliverables
```

The previous Field Station state is preserved separately under
`ET_fundacion_workspace_field_station`.

## Workflow

Activate the `et-fundacion` environment and ensure this checkout's package is
used:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python scripts/run_pipeline.py
```

Without a subcommand, the dispatcher only prints help.

Local validation and summary commands:

```powershell
python scripts/run_pipeline.py validate
python scripts/run_pipeline.py audit
python scripts/run_pipeline.py summarize
```

Commands that may query Earth Engine, rebuild training outputs or generate
rasters require explicit subcommands. See `scripts/README.md`.

Stable5 comparisons require an explicit reference workspace:

```powershell
--reference-workspace ..\ET_fundacion_workspace_field_station
```

No Google Drive output or persistent Earth Engine asset is part of the
production workflow.

## Interpretation guardrails

Internal CV, reciprocal transfer and AOA use a MODIS-derived Kc target; they do
not constitute independent 20 m ET validation. Field comparison uses a
field-derived ET proxy and is reported as such.

## Validation

```powershell
python -m pytest -q
git diff --check
```

Migration details are documented in `MIGRATION_REPORT.md`.

Authors: Cristian C. Montes-Chaura; Manuel Coy Pertuz.
