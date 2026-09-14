"""Focused read-only integration checks against the existing fitted RF25 artifacts.

No fitting, Earth Engine or raster production is performed. Artifact-dependent
checks skip in fresh checkouts where the ignored final joblib files are absent.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

from et_downscaling.rf25 import (
    RF25_AOA_FILENAME, RF25_MODEL_FILENAME, RF25_MODEL_FEATURES,
    _signature_array, rf25_model_signature,
)
from et_downscaling.rf25_overlap_production import build_production_scientific_signature

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "outputs/current/models"


@pytest.fixture
def artifacts():
    paths = [MODELS / RF25_MODEL_FILENAME, MODELS / RF25_AOA_FILENAME]
    if not all(path.is_file() for path in paths):
        pytest.skip("Existing final RF25 model and AOA artifacts are required; never train here.")
    return paths


def test_model_and_production_signatures_are_identical_across_processes(artifacts):
    script = """
import json, sys, joblib
from et_downscaling.rf25 import rf25_model_signature, validate_rf25_model
from et_downscaling.rf25_overlap_production import build_production_scientific_signature
results = []
for _ in range(2):
    model, aoa = joblib.load(sys.argv[1]), joblib.load(sys.argv[2])
    before = rf25_model_signature(model)
    validate_rf25_model(model)
    assert rf25_model_signature(model) == before
    results.append([before, build_production_scientific_signature(model, aoa)])
print(json.dumps(results))
"""
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    hashes_before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in artifacts]
    signatures = []
    for seed in ("1", "42", "123"):
        environment["PYTHONHASHSEED"] = seed
        result = subprocess.run([sys.executable, "-B", "-c", script, *map(str, artifacts)],
                                cwd=ROOT, env=environment, check=True, capture_output=True, text=True, timeout=60)
        signatures.extend(json.loads(result.stdout))
    assert len({tuple(pair) for pair in signatures}) == 1
    assert all(len(digest) == 64 for pair in signatures for digest in pair)
    assert [hashlib.sha256(path.read_bytes()).hexdigest() for path in artifacts] == hashes_before


@pytest.mark.parametrize("field", ["left_child", "right_child", "feature", "threshold",
                                    "impurity", "n_node_samples", "weighted_n_node_samples",
                                    "missing_go_to_left", "values"])
def test_named_tree_state_changes_both_signatures(artifacts, field):
    model, aoa = map(joblib.load, artifacts)
    before = rf25_model_signature(model)
    production_before = build_production_scientific_signature(model, aoa)
    state = model.estimators_[0].tree_.__getstate__()
    if field == "values":
        state["values"][0, 0, 0] += 0.125
    elif field == "missing_go_to_left":
        state["nodes"][field][0] ^= 1
    else:
        state["nodes"][field][0] += 1
    assert rf25_model_signature(model) != before
    assert build_production_scientific_signature(model, aoa) != production_before


def test_parameters_and_tree_order_are_part_of_identity(artifacts):
    model = joblib.load(artifacts[0])
    before = rf25_model_signature(model)
    seed = model.estimators_[0].random_state
    model.estimators_[0].random_state = seed + 1
    assert rf25_model_signature(model) != before
    model.estimators_[0].random_state = seed
    assert rf25_model_signature(model) == before
    model.estimators_[0], model.estimators_[1] = model.estimators_[1], model.estimators_[0]
    assert rf25_model_signature(model) != before


def test_padding_does_not_affect_signature(artifacts):
    model = joblib.load(artifacts[0])
    nodes = model.estimators_[0].tree_.__getstate__()["nodes"]
    padding = np.ones(nodes.dtype.itemsize, dtype=bool)
    for name in nodes.dtype.names:
        dtype, offset = nodes.dtype.fields[name][:2]
        padding[offset:offset + dtype.itemsize] = False
    assert padding.any()
    before = rf25_model_signature(model)
    raw = nodes.view(np.uint8).reshape(-1, nodes.dtype.itemsize)
    raw[:, padding] = 0xA5
    assert rf25_model_signature(model) == before
    raw[:, padding] = 0x5A
    assert rf25_model_signature(model) == before


def test_array_encoding_normalizes_endian_width_and_memory_order():
    for dtype in ("f8", "i8", "u8"):
        array = np.arange(12).reshape(3, 4).astype(dtype)
        reference = _signature_array(array).tobytes()
        assert _signature_array(np.asfortranarray(array)).tobytes() == reference
        assert _signature_array(array.astype(">" + dtype)).tobytes() == reference
        assert _signature_array(array.astype(dtype[0] + "4")).tobytes() == reference
    with pytest.raises(TypeError, match="Unsupported"):
        _signature_array(np.zeros(1, dtype=[("x", "i8")]))


def test_signature_preserves_predictions_and_matches_before_code_change(artifacts):
    baseline_path = ROOT / "outputs/evaluation/rf25_signature_audit/before.npz"
    if not baseline_path.is_file():
        pytest.skip("Before-change prediction snapshot is unavailable.")
    with np.load(baseline_path) as snapshot:
        inputs = pd.DataFrame(snapshot["inputs"], columns=RF25_MODEL_FEATURES)
        expected = snapshot["prediction"]
    model, aoa = map(joblib.load, artifacts)
    # Serial accumulation removes unrelated thread-scheduling roundoff from
    # the bitwise comparison; the fitted model's n_jobs remains unchanged (-1).
    with joblib.parallel_backend("sequential"):
        before_hashing = model.predict(inputs)
        rf25_model_signature(model)
        build_production_scientific_signature(model, aoa)
        after_hashing = model.predict(inputs)
    assert before_hashing.tobytes() == expected.tobytes() == after_hashing.tobytes()
    # Also check the unchanged production setting with parallel prediction.
    np.testing.assert_allclose(model.predict(inputs), expected, rtol=1e-14, atol=1e-14)
    assert model.n_jobs == -1
