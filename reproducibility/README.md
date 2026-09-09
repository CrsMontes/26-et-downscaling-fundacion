# Reproducibility archive

This directory contains executable evidence for methodological alternatives,
negative results, diagnostics and superseded workflows used in the Fundación
ET study.

The default operational workflow is not here. It is:

    python scripts/run_pipeline.py --project <earth-engine-project>

The current Virtual Station field-derived ET proxy comparison is:

    python scripts/run_pipeline.py evaluate-field --workspace-root <virtual-workspace> --project <earth-engine-project> --reference-workspace <field-reference-workspace>

This comparison uses the frozen Field Station workspace only as an explicit
reference. The legacy Field-only orchestration workflow is maintained in the
`field-station-stable` branch, not in Virtual Station `main`.

## Layout

- `scripts/`: historical experiments, audits, screenings and superseded
  workflows.
- `script_manifest.md`: explicit classification of current versus
  reproducibility scripts.

Nothing under `reproducibility/scripts/` is called automatically by the
production pipeline.

## Policy

Rejected predictors, models and workflows are retained when they provide
scientific evidence for a documented decision. Moving them here is an
organizational change only; it does not reactivate them. Git history remains
additional provenance.


## Final S2 FVC/albedo recheck

The 2026-09-06 S2-only recheck is implemented by:

    python reproducibility/scripts/recheck_s2_fvc_albedo.py --project <earth-engine-project> --execute

It recalibrates S2 FVC and compares Ridge25 against +albedo, +FVC and
+albedo+FVC without modifying the production configuration. HLS is not part of
this recheck. The result supported retaining the parsimonious Ridge-25 model.

## Final closure diagnostics

Two final audit scripts are retained as non-production evidence:

    python reproducibility/scripts/run_closure_diagnostics.py
    python reproducibility/scripts/run_aoa_map_sensitivity.py --project <earth-engine-project> --dates 2020-03-13

The offline script reproduces the equal-weight versus coefficient-weighted DI
sensitivity, explicit persistence baselines and field valid-day sensitivity.
The map script was used only to quantify the spatial impact of the alternative
coefficient-weighted DI and is not part of routine production.
