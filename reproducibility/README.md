# Reproducibility archive

This directory documents and, where appropriate, hosts the experiments used
to justify methodological decisions in the Fundación ET downscaling study.

## Important distinction

The default operational workflow is:

    python scripts/run_pipeline.py

Experimental and historical workflows are preserved for scientific
reproducibility but are not executed by the default pipeline.

## Categories

### Production

The operational workflow currently uses:

- scripts/run_pipeline.py
- scripts/export_meteorology_data.py
- scripts/export_satellite_data.py
- scripts/build_training_dataset.py

### Field evaluation

These scripts are retained for the next scientific phase:

- scripts/validate_field_downscaling.py
- scripts/analyze_field_diagnostics.py

### Reproducibility

All other experimental scripts are retained to reproduce model screening,
predictor availability, RF sensitivity, HLS/FVC experiments, feature
ablations, AOA diagnostics, support-threshold diagnostics and historical
production decisions.

They must not be called automatically by the production pipeline.

## Policy

A negative or superseded experiment is not deleted merely because it was not
selected. Its role is to document the evidence supporting the final method.

Historical code may eventually be moved under this directory, but only after
its imports, path assumptions and tests have been checked.

Git history is additional provenance, not a substitute for reproducible
scientific evidence.
