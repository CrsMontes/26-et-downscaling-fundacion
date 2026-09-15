# Current 30-cell notebook reconciliation

PASS: all differences explained. Current notebook preserved byte-for-byte; no figures, tables or rasters regenerated.
Cell numbers below are one-based; matching used persistent cell IDs and source content.

Previous: 27 cells, 3,347,839 bytes, SHA-256 `56045ab8d7698a4dfa07db60f1209b722a9585b8483b35f90e9669e34916f440`.
Current: 30 cells, 12,090,289 bytes, SHA-256 `40f3faad4e519c192d0c3b9ae95b670b9b705c199f56d3ddae7c969c5e8c9187`.
File growth: 8,742,450 bytes, from intentional code additions and stored output/image changes; byte growth is not a scientific-value change.
All 27 existing IDs remain in order. No markdown, notebook metadata or cell metadata changed.
Existing cells 3, 4, 6, 8, 10, 12, 14, 17, 18, 19, 21, 23, 26 and 27 have output serialization differences only: normalizing strings versus lists yields identical output content. Cell 25 alone has a newly rendered embedded image. Every other existing decoded image, including CV, is identical.

## Cell-level classification

| Cell | ID | Changed fields | Classification | Image content changed |
|---:|---|---|---|---|
| 1 | field-analysis-01 | none | unchanged | False |
| 2 | field-analysis-02 | none | unchanged | False |
| 3 | field-analysis-03 | outputs | regenerated notebook output (including execution state/serialization) | False |
| 4 | field-analysis-04 | outputs | regenerated notebook output (including execution state/serialization) | False |
| 5 | field-analysis-05 | none | unchanged | False |
| 6 | field-analysis-06 | outputs | regenerated notebook output (including execution state/serialization) | False |
| 7 | field-analysis-07 | none | unchanged | False |
| 8 | field-analysis-08 | outputs | regenerated notebook output (including execution state/serialization) | False |
| 9 | field-analysis-09 | none | unchanged | False |
| 10 | field-analysis-10 | outputs | regenerated notebook output (including execution state/serialization) | False |
| 11 | field-analysis-11 | none | unchanged | False |
| 12 | field-analysis-12 | outputs | regenerated notebook output (including execution state/serialization) | False |
| 13 | field-analysis-13 | none | unchanged | False |
| 14 | field-analysis-14 | outputs | regenerated notebook output (including execution state/serialization) | False |
| 15 | field-analysis-15 | none | unchanged | False |
| 16 | field-analysis-16 | none | unchanged | False |
| 17 | field-analysis-17 | outputs | regenerated notebook output (including execution state/serialization) | False |
| 18 | field-analysis-18 | outputs | regenerated notebook output (including execution state/serialization) | False |
| 19 | field-analysis-19 | outputs | regenerated notebook output (including execution state/serialization) | False |
| 20 | field-analysis-20 | none | unchanged | False |
| 21 | field-analysis-21 | outputs | regenerated notebook output (including execution state/serialization) | False |
| 22 | field-analysis-22 | none | unchanged | False |
| 23 | field-analysis-23 | outputs | regenerated notebook output (including execution state/serialization) | False |
| 24 | bd02fc5c | none | unchanged | False |
| 25 | 17afb957 | execution_count, outputs, source | intentional post-reconciliation manuscript/figure edit; station-ID migration change: explicit final-identity guards/order, preserving already migrated physical rules; regenerated notebook output (including execution state/serialization) | True |
| 26 | rf25-cv-inputs-20260914 | outputs | regenerated notebook output (including execution state/serialization) | False |
| 27 | 589bd454 | outputs | regenerated notebook output (including execution state/serialization) | False |
| 28 | b6151f49 | cell_type, execution_count, id, metadata, outputs, source | intentional post-reconciliation manuscript/figure edit; regenerated notebook output (including execution state/serialization) | True |
| 29 | cc33380e | cell_type, execution_count, id, metadata, outputs, source | intentional post-reconciliation manuscript/figure edit; regenerated notebook output (including execution state/serialization) | True |
| 30 | 541ccbea | cell_type, execution_count, id, metadata, outputs, source | intentional post-reconciliation manuscript/figure edit; regenerated notebook output (including execution state/serialization) | True |

Execution counts: only existing cell 25 changed (14 to 4); other existing counts unchanged. Added-cell counts are recorded in the receipt.

## Intentional source changes

Cell 25 (`17afb957`): Figure 3 now discovers the repository/workbook locally, reads via pandas, asserts the final ID-to-cover mapping, and explicitly selects fixed Kc for ST02/ST03/ST04 and NDVI Kc for ST01/ST05. It adds frozen daily-count checks and orders the boxes by final ID. Layout, fonts, line/outlier styling and save behavior changed: a 300-dpi final_ids PNG replaces this cell's former 600-dpi color/grayscale writes. Physical colors remain associated with the same covers. See `cell_25_source.diff` for the complete code delta.
The former explicit QC filter is absent from the new cell, but the existing workbook already masks invalid ETo-equivalent observations. Independently recomputed old/new March-June sample masks and every daily ET value are identical: 417 observations (66, 82, 82, 112, 75 by final ID). No methodology change results for this validated workbook.

- Added cell 28 (`b6151f49`): Spatial manuscript comparison of existing MODIS/RF25 rasters for 2020-03-13, 2024-07-11 and 2022-03-30; geographic display and shared color scale.
- Added cell 29 (`cc33380e`): Figure 11: field-proxy versus MODIS/reconciled RF25 scatter and agreement metrics, all 37 pairs and fixed-Kc subset of 19.
- Added cell 30 (`541ccbea`): Five-panel station temporal comparison of field proxy, MODIS and reconciled RF25, with final station names.

## CV and scientific checks

Both previously reconciled CV sources (`rf25-cv-inputs-20260914`, `589bd454`) are text-identical and AST-identical. Their numerical and plotting logic is unchanged. All eight preserved CV files retain their hashes. Stored output representation changes do not replace or alter CV code.
No remaining active old station identities were found in notebook source. Historical/provenance descriptions remain explicitly historical. Final physical mapping and Kc assignments match the migrated workbook.
The added Figure 11 and temporal selections each match the canonical 37 row indices exactly; fixed-Kc subset is 19. Their alternative proxy-column selection yields exactly the historical-reference values on all 80 periods. Figure 11 metric function agrees with repository metrics and its saved CSV within 1.78e-15.
Common counts: ST01 Mangrove 8; ST02 Pasture 5; ST03 Oil palm plantation 7; ST04 Banana plantation 7; ST05 Dry forest 10.
All code cells parse successfully and have no stored error outputs. Only the isolated lightweight metric function was evaluated; the notebook was not executed or rewritten.
The spatial-comparison cell reads existing rasters and reprojects only display arrays in memory. No training, AOA, Virtual10 selection, raster production, download or Earth Engine call was executed. The separate 13-date production task remains paused.
Complete source/output/image hashes, execution counts, file hashes and categories are in `reconciliation_receipt.json`.
