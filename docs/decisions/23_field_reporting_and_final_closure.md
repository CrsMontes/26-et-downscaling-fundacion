# Decision 23 ? Field reporting correction and final closure

## Problem

`virtual_native_*_with_AOA` metrics were inadvertently conditioned on Stable5
availability.

## Evidence

Independent reconstruction showed that missing Stable5 predictions reduced
Virtual10 native samples from 20 to 17 and from 11 to 9 in key scenarios.

## Decision

- `virtual_native_*`: evaluate MODIS and Virtual10 on Virtual10's own valid
  domain.
- `matched_*`: evaluate MODIS, Stable5 and Virtual10 on common complete cases.

No prediction is imputed.

## Result

Correct WITH-AOA sample sizes:

- primary all: n = 20
- primary fixed-Kc: n = 11
- strict all: n = 14
- strict fixed-Kc: n = 7

Matched metrics remained unchanged to numerical precision.

The complete suite passed 135 tests plus 10 subtests.

This was a reporting error, not a model, target, AOA or raster-production error.

Closure commit:

`62eb897e5be8cf30bda59f77aa452284a5cd043c`
