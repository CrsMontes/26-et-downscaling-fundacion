# Methodological decision index

Decision files are retained as a chronological scientific record. Some older
files describe methods that were later superseded; they are not instructions
for the current pipeline. Duplicate historical numbers (13-15) are preserved
to avoid rewriting provenance.

## Current authoritative path

For the active `experiment-5year` workflow, read these first:

1. `11_final_kc_model.md` — model-selection basis.
2. `13_ridge25_final_model.md` — final Ridge-25 specification.
3. `14_clean_run_architecture.md` — rebuild-from-source architecture.
4. `16_local_tiled_production.md` — local production constraints.
5. `17_single_entry_pipeline.md` — `scripts/run_pipeline.py` as the normal entry point.
6. `19_exact_overlap_global_reconciliation.md` — **current final conservation method**.

## Important superseded production decisions

- `12_conservative_reconciliation.md`: historical iterative reconciliation;
  superseded by Decision 19.
- `14_adaptive_reconciliation_threshold_diagnostic.md`: diagnostic only.
- `15_fixed_tiles_after_recursive_split_diagnostic.md`: diagnostic evidence.
- `15_ridge25_spatial_production.md`: earlier production stage.
- `18_final_ridge25_production_freeze.md`: superseded where it conflicts with
  Decision 19.

All other decision files remain evidence for target definition, predictor
selection, spatial support, meteorology, AOA and rejected alternatives.
