# Predictor catalog

The repository distinguishes **materialized candidate predictors** from the **frozen RF-25 model predictors**. Downloading a candidate does not make it eligible for the final model.

## Final RF-25 predictors

### Sentinel-2 optical — 16

`Blue_mean`, `Green_mean`, `Red_mean`, `NIR_mean`, `SWIR1_mean`, `SWIR2_mean`, `NDVI_mean`, `EVI_mean`, `SAVI_mean`, `NDWI_mean`, `NDMI_mean`, `RedEdge1_mean`, `RedEdge2_mean`, `RedEdge3_mean`, `NIR_Broad_mean`, `NDRE_mean`.

The operational optical composite remains the deterministic Sentinel-2 temporal medoid on the 20 m prediction grid with the established clear-pixel QA configuration.

### ERA5-Land — 5

`Tair_mean_C`, `Tair_max_C`, `VPD_mean_kPa`, `SolarRad_MJ_m2_day`, `Wind_mean_ms`.

These retain their coarse meteorological support even though they are distributed onto the 20 m prediction grid.

### Seasonality — 4

`doy_sin1`, `doy_cos1`, `doy_sin2`, `doy_cos2`.

## Materialized candidate archive

`scripts/download_all_candidate_predictors.py` reconstructs all candidate families already implemented and audited in the repository for the canonical 2020–2024 five-field-station candidate universe. These files are for provenance, diagnostics and possible future controlled experiments; they do not alter RF-25.

- Sentinel-2 common optical variables.
- Sentinel-2 red-edge variables and NDRE.
- Sentinel-2 albedo and FVC.
- HLS S30/L30 common optical variables.
- HLS albedo and FVC.
- Sentinel-1 R077: VV, VH and VV−VH.
- Sentinel-1 R142: VV, VH and VV−VH.
- ERA5-Land final variables plus `VPD_max_kPa` and reference-ET derivatives retained in the candidate store.
- CHIRPS `Precip_period_mm` and `Precip_prev30d_mm`.
- Landsat L8/L9 surface-temperature candidate `LST_parent_mean_K` plus thermal QA/provenance fields.
- Static footprint mean elevation from the implemented station-support extraction.
- Seasonal harmonics.

### Important scope distinction

The comprehensive candidate archive is reconstructed on the canonical five field-station MODIS footprints because that is the support on which the historical candidate experiments were implemented. The **final Virtual10 training extraction is intentionally limited to the frozen 25 RF predictors**. This prevents unused candidate availability from redefining the Virtual10 training population or creating a hidden complete-case filter.

Terrain slope/aspect were discussed historically but are not currently materialized as audited candidate predictors. They are therefore not silently added to the archive or model.
