"""Audit ST01 RF25 state after nearest-valid ERA5-Land coastal support."""

from pathlib import Path
import sys

import ee
import joblib
import numpy as np

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))

import produce_field_rf25_halo as base

from et_downscaling.meteorology_export import (
    get_era5_collection,
    get_nearest_valid_era5_point,
)
from et_downscaling.rf25 import RF25_MODEL_FEATURES
from et_downscaling.rf25_local_state import score_local_rf25
from et_downscaling.rf25_production import (
    build_rf25_meteorological_predictors,
    build_rf25_production_stack,
)

PROJECT = "ee-sneiderquintero"
DATE = "2022-03-30"
LON = -74.360002
LAT = 10.766952

ee.Initialize(project=PROJECT)
ee.Number(1).getInfo()

point = ee.Geometry.Point([LON, LAT])

workspace = base.get_workspace_paths(ROOT).ensure()

model = joblib.load(
    workspace.models / base.RF25_MODEL_FILENAME
)
aoa = joblib.load(
    workspace.models / base.RF25_AOA_FILENAME
)

context = build_rf25_production_stack(
    DATE,
    point.buffer(5000),
)

# Resolve the same nearest-valid ERA5-Land support logic used
# by the station meteorology workflow.
era5_collection = get_era5_collection()
era5_reference = (
    ee.Image(era5_collection.first())
    .select("temperature_2m")
)
era5_projection = era5_reference.projection()
era5_scale = era5_projection.nominalScale()

support = get_nearest_valid_era5_point(
    point,
    era5_reference,
    era5_projection,
    era5_scale,
)

support_geometry = support.geometry()
support_coordinates = support_geometry.coordinates().getInfo()
support_distance_m = support.get("era5_distance_m").getInfo()

# Build the period meteorology over a small geometry around
# the resolved valid ERA5-Land grid point.
meteorology = build_rf25_meteorological_predictors(
    period_start=context["period_start"],
    period_end=context["period_end"],
    number_days=context["number_days"],
    processing_geometry=support_geometry.buffer(2000),
)

optical_values = context["optical"].reduceRegion(
    reducer=ee.Reducer.first(),
    geometry=point.buffer(20),
    scale=20,
    maxPixels=10000,
).getInfo()

harmonic_values = context["harmonics"].reduceRegion(
    reducer=ee.Reducer.first(),
    geometry=point.buffer(20),
    scale=20,
    maxPixels=10000,
).getInfo()

meteorology_values = meteorology.reduceRegion(
    reducer=ee.Reducer.first(),
    geometry=support_geometry.buffer(100),
    crs=era5_projection,
    scale=era5_scale,
    maxPixels=10000,
).getInfo()

values = {}
values.update(optical_values)
values.update(meteorology_values)
values.update(harmonic_values)

missing = [
    name
    for name in RF25_MODEL_FEATURES
    if values.get(name) is None
]

print()
print("=" * 78)
print("ST01 NEAREST-VALID ERA5-LAND EXTENSION")
print("=" * 78)
print("ERA5 support longitude:", support_coordinates[0])
print("ERA5 support latitude :", support_coordinates[1])
print("ERA5 distance (m)     :", support_distance_m)

print()
for name in RF25_MODEL_FEATURES:
    print(
        f"{name:28s}",
        "MISSING" if values.get(name) is None else values[name],
    )

print()
print("Missing predictors:", len(missing))

if missing:
    raise RuntimeError(
        "Predictors remain missing: " + ", ".join(missing)
    )

cube = np.asarray(
    [[[
        float(values[name])
        for name in RF25_MODEL_FEATURES
    ]]],
    dtype=float,
)

state = score_local_rf25(
    predictor_cube=cube,
    model=model,
    aoa_parameters=aoa,
)

print()
print("=" * 78)
print("RF25 / AOA RESULT")
print("=" * 78)
print("stack_valid :", bool(state.stack_valid[0, 0]))
print("Kc_raw      :", float(state.kc_raw[0, 0]))
print(
    "DI          :",
    float(state.dissimilarity_index[0, 0]),
)
print(
    "AOA threshold:",
    float(aoa.threshold),
)
print("LPD         :", int(state.local_point_density[0, 0]))
print("AOA_inside  :", bool(state.aoa_inside[0, 0]))
print("usable      :", bool(state.usable[0, 0]))
