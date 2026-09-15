"""Audit RF25 predictor availability at ST01."""

import ee

from et_downscaling.rf25 import RF25_MODEL_FEATURES
from et_downscaling.rf25_production import build_rf25_production_stack

PROJECT = "ee-sneiderquintero"
DATE = "2022-03-30"

# ST01 Mangrove
LON = -74.360002
LAT = 10.766952

ee.Initialize(project=PROJECT)
ee.Number(1).getInfo()

point = ee.Geometry.Point([LON, LAT])

# Local geometry only for this diagnostic.
stack_info = build_rf25_production_stack(
    DATE,
    point.buffer(5000),
)

stack = stack_info["stack"]

values = stack.reduceRegion(
    reducer=ee.Reducer.first(),
    geometry=point,
    scale=20,
    maxPixels=10000,
).getInfo()

print()
print("=" * 70)
print("ST01 RF25 PREDICTOR AVAILABILITY")
print("=" * 70)

missing = []

for feature in RF25_MODEL_FEATURES:
    value = values.get(feature)
    status = "MISSING" if value is None else "OK"
    if value is None:
        missing.append(feature)

    print(
        f"{feature:28s} {status:8s} "
        f"{'' if value is None else value}"
    )

print()
print("Missing predictors:", len(missing))
for feature in missing:
    print(" -", feature)
