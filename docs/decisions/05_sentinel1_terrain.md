# Sentinel-1 terrain treatment

## Problem
Determine whether an additional radiometric terrain correction is required beyond the preprocessing already represented by the Earth Engine Sentinel-1 GRD product.

## Alternatives
- Current Sentinel-1 GRD preprocessing.
- Additional radiometric terrain flattening / RTC workflow.
- Alternative RTC products.

## Evidence
No empirical evidence has yet demonstrated that an additional RTC step improves the transferable ET model for these data.

Introducing another terrain correction would change the diagnosed processing strategy and must therefore be supported by a controlled experiment.

## Decision
Do not add an additional RTC operation to the primary pipeline at this stage.

Retain acquisition geometry and incidence-angle information for QA.

Additional RTC remains a sensitivity experiment if later residual analyses show a systematic topographic radar bias.

## Status
Primary treatment accepted; sensitivity remains available.
