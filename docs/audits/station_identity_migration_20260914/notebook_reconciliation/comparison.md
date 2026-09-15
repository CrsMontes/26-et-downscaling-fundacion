# Notebook reconciliation before promotion

Cell numbers are one-based. All three input notebooks have 26 cells with matching IDs.

Only cell 26 (ID `589bd454`) differs between backup and active. This is manuscript CV figure work:
VF01–VF10 block mapping/order, revised GridSpec proportions/spacing, fonts/annotations and `_vf_supports` filenames.
Its former input-loading/metric-check code was removed and replaced by a requirement for existing kernel objects.
No ST01–ST05 migration appears in the active edit. Markdown, other cells and notebook/cell metadata are unchanged.

Exact file growth: 344,669 bytes.
Serialized source contribution: -8,788; execution count: +0; outputs: +353,457 bytes.
Execution count changed 26 to 32. Outputs changed from six to two, including a different embedded PNG.

The previous staged cell 26 was a migration-created display-only shortcut and must be replaced with the authoritative active plotting logic.

| Comparison | Cell / ID | Type | Source | Execution count | Outputs | Cell metadata |
|---|---|---|---|---|---|---|
| baseline_vs_active | 26 / 589bd454 | code | True | 26 → 32 | True | False |
| baseline_vs_staged | 3 / field-analysis-03 | code | True | 1 → 1 | True | False |
| baseline_vs_staged | 4 / field-analysis-04 | code | True | 2 → 2 | True | False |
| baseline_vs_staged | 6 / field-analysis-06 | code | False | 3 → 3 | True | False |
| baseline_vs_staged | 8 / field-analysis-08 | code | True | 4 → 4 | True | False |
| baseline_vs_staged | 10 / field-analysis-10 | code | False | 5 → 5 | True | False |
| baseline_vs_staged | 12 / field-analysis-12 | code | False | 6 → 6 | True | False |
| baseline_vs_staged | 14 / field-analysis-14 | code | False | 7 → 7 | True | False |
| baseline_vs_staged | 15 / field-analysis-15 | markdown | True | None → None | False | False |
| baseline_vs_staged | 16 / field-analysis-16 | code | True | 8 → 8 | False | False |
| baseline_vs_staged | 17 / field-analysis-17 | code | False | 9 → 9 | True | False |
| baseline_vs_staged | 18 / field-analysis-18 | code | False | 10 → 10 | True | False |
| baseline_vs_staged | 19 / field-analysis-19 | code | True | 11 → 11 | True | False |
| baseline_vs_staged | 20 / field-analysis-20 | markdown | True | None → None | False | False |
| baseline_vs_staged | 21 / field-analysis-21 | code | False | 12 → 12 | True | False |
| baseline_vs_staged | 23 / field-analysis-23 | code | True | 13 → 13 | True | False |
| baseline_vs_staged | 24 / bd02fc5c | code | False | None → 14 | True | False |
| baseline_vs_staged | 25 / 17afb957 | code | True | 22 → 14 | True | False |
| baseline_vs_staged | 26 / 589bd454 | code | True | 26 → 15 | True | False |
| active_vs_staged | 3 / field-analysis-03 | code | True | 1 → 1 | True | False |
| active_vs_staged | 4 / field-analysis-04 | code | True | 2 → 2 | True | False |
| active_vs_staged | 6 / field-analysis-06 | code | False | 3 → 3 | True | False |
| active_vs_staged | 8 / field-analysis-08 | code | True | 4 → 4 | True | False |
| active_vs_staged | 10 / field-analysis-10 | code | False | 5 → 5 | True | False |
| active_vs_staged | 12 / field-analysis-12 | code | False | 6 → 6 | True | False |
| active_vs_staged | 14 / field-analysis-14 | code | False | 7 → 7 | True | False |
| active_vs_staged | 15 / field-analysis-15 | markdown | True | None → None | False | False |
| active_vs_staged | 16 / field-analysis-16 | code | True | 8 → 8 | False | False |
| active_vs_staged | 17 / field-analysis-17 | code | False | 9 → 9 | True | False |
| active_vs_staged | 18 / field-analysis-18 | code | False | 10 → 10 | True | False |
| active_vs_staged | 19 / field-analysis-19 | code | True | 11 → 11 | True | False |
| active_vs_staged | 20 / field-analysis-20 | markdown | True | None → None | False | False |
| active_vs_staged | 21 / field-analysis-21 | code | False | 12 → 12 | True | False |
| active_vs_staged | 23 / field-analysis-23 | code | True | 13 → 13 | True | False |
| active_vs_staged | 24 / bd02fc5c | code | False | None → 14 | True | False |
| active_vs_staged | 25 / 17afb957 | code | True | 22 → 14 | True | False |
| active_vs_staged | 26 / 589bd454 | code | True | 32 → 15 | True | False |

Full source diffs and byte-exact snapshots of all three states accompany this report.
Merge plan: retain already verified migrated cells 1–25, restore the required read-only CV input/check code as a separate cell,
and preserve the current manuscript plot body inside a function. CV regeneration defaults to false.
The existing `_vf_supports` image is displayed; no CV PNG/PDF, OOF values, model or selection is regenerated.

## Embedded-image differences

- baseline_vs_active: cells 26.
- baseline_vs_staged: cells 17, 18, 19, 24, 25.
- active_vs_staged: cells 17, 18, 19, 24, 25, 26.

Reconciliation verified: latest plotting body has identical Python AST, latest embedded CV image has identical SHA-256, all eight CV files and both OOF CSVs remain byte-identical. The block-to-VF mapping matches every preserved spatial OOF row.
