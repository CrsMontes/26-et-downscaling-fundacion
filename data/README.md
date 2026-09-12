# Data

This directory contains the tracked local inputs for the ET downscaling
workflow and its separate field comparison.

## Active inputs

- `boundaries/fundacion_basin.geojson`  
  Fundación River Basin boundary.

- `stations/fundacion_stations.geojson`  
  Station geometry, stable station identifiers, and station-level metadata.

- `field/field_etgage.csv`  
  Curated daily ETgage observations for separate field-proxy validation. This
  file is not required by the canonical RF-25 pipeline.

## Stations

| station_id | station |
|---|---|
| ST01 | Clean pasture |
| ST02 | Oil palm |
| ST03 | Banana |
| ST04 | Mangrove |
| ST05 | Dry forest |

Historical field workbooks and intermediate reconstruction files may be kept
locally under `field/archive/`. This directory is ignored by Git and is not
required to run the workflow.

Satellite and meteorological inputs are retrieved from their remote data
collections. Generated datasets, models, tables, and figures are written under
`outputs/`.
