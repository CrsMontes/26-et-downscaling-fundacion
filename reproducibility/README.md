# Reproducibility archive

This directory contains executable evidence for methodological alternatives,
negative results, diagnostics and superseded workflows used in the Fundación
ET study.

The default operational workflow is not here. It is:

    python scripts/run_pipeline.py --project <earth-engine-project>

The current field comparison is:

    python scripts/evaluate_field_ridge25.py --project <earth-engine-project>

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
