# Decision 14 - Clean-run architecture and external workspace

## Problem

Earlier workflows accumulated raw exports, fitted models, diagnostic caches and
figures under the repository `outputs/` directory. The accepted pipeline could
also reuse fitted models. This made it easier to resume development, but a
successful run could depend on artifacts from an earlier execution.

## Decision

The active repository contains only:

- source code;
- tests;
- documentation;
- environment/configuration files;
- three portable local scientific inputs:
  - basin boundary;
  - station geometries;
  - field ETgage table.

All generated data are written to an external workspace. By default this is:

`<repository-parent>/ET_fundacion_workspace/current`

The path can be overridden with `ET_FUNDACION_WORKSPACE`.

## Data levels

1. `raw/`: permissive remote extractions and provenance.
2. `master/`: complete station-period predictor database used to reproduce
   methodological decisions.
3. `runs/<run-id>/`: run-specific derived Ridge-25 training table, OOF
   predictions, statistics, fitted model provenance and diagnostic figures.
4. `rasters/`: optional fine-resolution ET products.

## Reproducibility rule

A fresh clone with the three portable inputs must be able to rebuild the study.
A normal scientific run may reuse only a verified raw extraction cache; it must
rebuild the master-derived training population, cross-validation results and
fitted Ridge model.

Reconciliation is not part of model training or OOF validation. It is applied
only when a fine-resolution product is generated.
