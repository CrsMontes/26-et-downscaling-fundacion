# Field station identity migration, 2026-09-14

The field station nomenclature is versioned as `field_station_nomenclature_v2_20260914`.
Station IDs are scientific join/selection keys, not just display labels. The
crosswalk below moves whole physical stations, including measurements, coordinates,
dates, reference ET, canvas, basin membership and source-sheet provenance.

| Old ID | Current ID | Physical station | Reference | Historical / FAO Kc | Domain |
|---|---|---|---|---|---|
| ST04 | ST01 | Mangrove | ETr | NDVI / NDVI | Validation extension, coastal ERA5-Land support |
| ST01 | ST02 | Pasture | ETo | 0.85 / 0.75 | Fundación basin |
| ST02 | ST03 | Oil palm plantation | ETr | 0.95 / 1.00 | Fundación basin |
| ST03 | ST04 | Banana plantation | ETr | 1.10 / 1.10 | Fundación basin |
| ST05 | ST05 | Dry forest | ETr | NDVI / NDVI | Fundación basin |

The physical identity registry is `src/et_downscaling/field_station_identity.py`.
Stable `station_uid` values retain the existing physical slugs. Original
`source_sheet` names are unchanged: `Mangrove`, `Pastos Limpios`, `Palmera`,
`Banana`, and `Fragmento de bosque`. The original installation/removal dates,
canvas types, conformity flags and coordinates stay with those identities.

Only Mangrove and Dry forest use NDVI Kc in historical/FAO comparisons. The
pre-existing `ndvi20_all` sensitivity remains an explicitly separate scenario
for all five stations. Removing it would change the scientific methodology.

Changing just the station ID can silently change fixed Kc, ETo/ETr joins,
installation QC, basin selection, coastal filling and raster extraction. Active
field inputs now require physical identity/version evidence, and station geometry
is checked against the registry. External daily-reference, hourly ERA5, support
and satellite tables must be explicitly migrated before reuse. A set containing
the five expected IDs is insufficient to establish physical identity.

## Files and scientific preservation

Active GeoJSON, observations, field validation logic, review builder, halo scripts,
tests, workbook, notebook, derived tables, sidecars and five station figures use
the new nomenclature. Mangrove scripts use role-based `*_mangrove_*` filenames.
The extension directory is `outputs/evaluation/field_validation/st01_validation_extension`.
Its identity is ST01; ST04 is the banana plantation inside the basin.

The migration uses one crosswalk lookup per original identity. Files are assembled
in `.station-migration-stage`, checked against `E:/ET_backup_20260914`, then the
complete field directory is swapped. No cyclic in-place station renames occur.
The untouched original field directory is retained in `.station-migration-legacy`.
Both local directories are recovery artifacts, not active scientific inputs.

The new manifest is
`outputs/evaluation/field_validation/station_identity_migration_manifest.json`.
It records every baseline path, destination and before/after SHA-256, raster
invariants, tests and preserved historical evidence. Numeric CSV values are
retained; station-dependent pair-key digests are recomputed. Workbook edits only
change station IDs and land-cover names. The notebook rebuilds derived tables and
figures from those unchanged measurements; it does not change the source workbook.

The 56 published RF25 TIFFs for renamed stations need an embedded `station_id`
tag update. Their file hashes change; arrays, masks, CRS, transforms and support do
not. The other 156 field TIFFs retain their bytes after physical path matching.
Exactly 70 `RF25_halo7_*.tif` products remain in the active halo directory.
RF25 models, AOA and all basin production files are preserved byte-for-byte.
No training, Virtual10 selection, downloads, Earth Engine or raster production
is needed for this identity migration.

Historical audit manifests, logs, method identifiers, source hashes, legacy
signatures and frozen CV figures remain evidence in their original namespace.
`docs/FIELD_RF25_AUDIT.md`, `scripts/migrate_field_rf25_signatures.py` and the
retrospective ST02/ST03 note in `scripts/candidates/build_candidate_feature_store.py`
retain historical IDs. Interpret those through this crosswalk, not the current ID
registry. Original field metadata are retained intact in the legacy bundle and
external backup. Current copies link to the new migration manifest.

The method ID `st04_validation_extension_era5_nearest_valid_land_pixel_fill_v1`
is intentionally unchanged because it is hashed into the mangrove scientific
signature. It is an algorithm identifier, not current station ownership.

## Offline checks and limitations

`python -B scripts/migrate_field_station_ids.py verify-current` checks the
recovered inventories against the current physical station/date records. Focused
tests include `tests/test_field_station_identity.py`, which rejects coordinate,
attribute and mixed-namespace swaps and checks current Kc and harmonization logic.
The workbook has 615 DAILY rows, 80 PERIOD rows and 37 common validation pairs:
ST01 8, ST02 5, ST03 7, ST04 7, ST05 10. ETgage ET remains a field-derived proxy;
station-pixel comparisons are not independent validation at 20 m support.

The original raw field acquisition cache was not recovered. Raw-input rebuilding
of the review therefore requires an explicitly migrated local cache supplied via
`build_field_validation_review.py --cache-dir`; this migration does not acquire
missing inputs. The validated workbook and notebook remain usable offline.

Checkpoint: `pre-station-id-migration-20260914`, baseline
`f8d96bda7e5182bd8f3ed696360d583eb85133ec`, external inventory 402 field-validation
files plus three core files. No commit is created by the migration.

The scientifically reviewed, text-only evidence intended for version control is
in [the migration audit bundle](audits/station_identity_migration_20260914/README.md).
It includes the completed manifest, file/hash inventories, test results, both
notebook reconciliations and the current 30-cell notebook source snapshot.
Generated outputs, full notebooks and temporary recovery bundles stay ignored.
