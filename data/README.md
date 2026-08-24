# Data

This directory contains the local inputs required by the ET downscaling
workflow.

## Active inputs

- `boundaries/fundacion_basin.geojson`  
  Fundaci?n River Basin boundary.

- `stations/fundacion_stations.geojson`  
  Station geometry, stable station identifiers, and station-level metadata.

- `field/field_etgage.csv`  
  Curated daily ETgage observations.

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
