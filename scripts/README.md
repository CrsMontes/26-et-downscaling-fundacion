# Virtual Station operational scripts

The `main` branch is the Virtual Station V5 workflow.

Normal entry point:

```powershell
python scripts/run_pipeline.py <command>
```

Running `run_pipeline.py` without a command only prints help. No selection,
training, Earth Engine query, download, or raster production is started
implicitly.

## Local validation and derived-result commands

```powershell
python scripts/run_pipeline.py validate
python scripts/run_pipeline.py audit
python scripts/run_pipeline.py summarize
python scripts/run_pipeline.py compare-coverage `
    --reference-workspace ..\ET_fundacion_workspace_field_station
```

`validate` and `audit` are read-only by default. `summarize` and `compare-coverage` do not retrain models or download data, but they may rewrite derived summary/comparison tables in the Virtual Station evaluation workspace.

## Explicit reproduction commands

These commands may query Earth Engine or rewrite generated outputs and therefore
must be requested explicitly:

```powershell
python scripts/run_pipeline.py reproduce-selection `
    --project ee-sneiderquintero

python scripts/run_pipeline.py reproduce-training `
    --project ee-sneiderquintero

python scripts/run_pipeline.py evaluate-field `
    --project ee-sneiderquintero `
    --reference-workspace ..\ET_fundacion_workspace_field_station

python scripts/run_pipeline.py produce `
    --project ee-sneiderquintero `
    --date 2020-03-13
```

The frozen V5 selection uses seed 42, 10 supports in 10 fixed UTM 10 km blocks,
and the pre-specified GE90 availability rule. Reproduction must not change those
scientific decisions.

`compare_v5_basin_coverage.py` is intentionally read-only: it compares existing
V5 and Stable5 rasters and never regenerates them.

`produce_virtual_rasters.py` reconstructs Ridge25 and the equal-weight AOA from
the frozen V5 training population and generates only missing explicitly
requested dates. Existing scientific rasters are never overwritten.

Stable5 is external to this repository. Any Stable5 comparison must receive an
explicit `--reference-workspace` pointing to
`ET_fundacion_workspace_field_station`.
