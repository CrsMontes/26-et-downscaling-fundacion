# Reviewed station identity migration evidence

Scientific review passed before this migration commit was prepared. This bundle
preserves the verification evidence as it existed before committing; fields such
as `committed: false`, baseline HEAD, local paths and source hashes in copied
receipts describe that checkpoint and are deliberately not rewritten afterward.
The containing Git commit records the approved repository changes.

Baseline: `f8d96bda7e5182bd8f3ed696360d583eb85133ec`, tagged
`pre-station-id-migration-20260914`. External backup: `E:/ET_backup_20260914`.
The migration was performed on `station-id-migration`; no merge or production
run is part of this operation.

## Physical identity and invariance

| Historical ID | Final ID | Physical station | Common observations |
|---|---|---|---:|
| ST04 | ST01 | Mangrove | 8 |
| ST01 | ST02 | Pasture | 5 |
| ST02 | ST03 | Oil palm plantation | 7 |
| ST03 | ST04 | Banana plantation | 7 |
| ST05 | ST05 | Dry forest | 10 |

- DAILY: 615 rows; PERIOD: 80; common sample: 37; fixed-Kc subset: 19.
- Physical measurements, dates, coordinates, reference ET, Kc and QC unchanged.
- Maximum pooled-metric roundoff: `7.105427357601002e-15`.
- All 212 field TIFF arrays, masks, CRS and geotransforms unchanged.
- 56 RF25 TIFF hashes changed only for embedded station-ID metadata; 156 field
  TIFFs remain byte-identical after matching physical station paths.
- All 1,593 protected production files, including 792 TIFFs, retain their hashes.
- Full suite: 114 passed, 1 skipped. Final targeted identity run: 6 passed.
  The full-suite result was reused after notebook reconciliation because no
  executable project code changed. The skipped test lacks its historical
  before-code-change prediction snapshot.
- All 402 original field-validation backup files remain byte-identical.

## Retained evidence

- [Migration manifest](migration_manifest.json): completed verification,
  physical crosswalk, file hashes, invariance checks, test results and legacy policy.
- [File/path/hash inventory](file_changes.csv): all 405 recovered baseline entries;
  289 paths changed and 227 file hashes changed, with reasons. This is a local
  artifact inventory, not the list of files committed to Git.
- [Protected production hashes](protected_current_sha256.json) and
  [physical crosswalk](crosswalk.json).
- [Full-suite result](full_tests.xml) and
  [final targeted result](identity_tests_current.xml).
- [Original notebook reconciliation](notebook_reconciliation/comparison.md) and
  [receipt](notebook_reconciliation/reconciliation_receipt.json).
- [Authoritative 30-cell reconciliation](notebook_reconciliation_30/comparison.md),
  [receipt](notebook_reconciliation_30/reconciliation_receipt.json), and
  [complete cell 25 source delta](notebook_reconciliation_30/cell_25_source_diff.json).
- [Notebook source snapshot](notebook_sources.json): all 30 current cell IDs,
  types and source contents, without embedded figures or execution outputs.

The notebook is preserved locally byte-for-byte: 30 cells, 12,090,289 bytes,
SHA-256 `40f3faad4e519c192d0c3b9ae95b670b9b705c199f56d3ddae7c969c5e8c9187`.
Cell 25's intentional Figure 3 edits preserve 417 daily observations and values.
Added cells 28-30 contain spatial comparison, Figure 11 agreement plots/metrics,
and five-station temporal comparison. Both reconciled CV sources and all eight
CV artifact hashes are unchanged. Other existing-cell output differences are
serialization-only. No unexplained notebook change remains.

The source diff is stored as a JSON string with its original hash to preserve
unified-diff context whitespace while keeping Git whitespace checks clean.
The copied comparison report refers to the original local `.diff` filename.

## Commit boundaries and provenance

Only source, versioned station data, tests, documentation and this text evidence
bundle are tracked. No TIFF is tracked at either its historical or migrated path;
none of the 56 metadata-modified TIFFs belongs in this commit. Existing ignore
rules continue to exclude generated outputs and full local notebooks. The
workbook, raster products, manuscript figures, analysis tables and full notebook
remain available locally and are represented here by hashes and source evidence.

`.station-migration-stage` and `.station-migration-legacy` remain local recovery
material, excluded from the commit. Notebook snapshots referenced by the original
receipt remain in the local audit directory; they are not duplicated here.
The unrelated `rf25_eval_search.txt` is intentionally excluded.

Historical IDs in audit documents, source-sheet names, logs, legacy signatures,
provenance hashes and the `st04_validation_extension_era5_nearest_valid_land_pixel_fill_v1`
method identifier remain historical evidence. Mangrove's current identity and
extension ownership are ST01; ST04 is Banana plantation.

Known recovery limitation: the original raw acquisition cache is unavailable;
no download or raw-input workbook rebuild was attempted. No RF25 training, AOA
rebuild, Virtual10 selection or production raster regeneration occurred. The
separate 13-date full-basin production task remains paused.
