import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

import et_downscaling.field_rf25_products as module
from et_downscaling.field_validation import sha256


def _write_fake_downloader(root: Path) -> str:
    path = root / module.DOWNLOADER
    path.parent.mkdir(parents=True, exist_ok=True)
    source = "def download_ee_bytes(url):\n    return b'payload'\n"
    path.write_text(source, encoding="utf-8")
    return source


@pytest.mark.parametrize("succeeds", [True, False])
def test_canonical_production_is_isolated_and_only_completed_run_gets_evidence(
    tmp_path, monkeypatch, succeeds
):
    source_models = tmp_path / "protected_models"
    source_models.mkdir()

    names = (
        module.RF25_MODEL_FILENAME,
        module.RF25_AOA_FILENAME,
        module.RF25_METADATA_FILENAME,
    )
    for name in names:
        (source_models / name).write_bytes(name.encode())

    contract = {
        f"models/{name}": sha256(source_models / name)
        for name in names
    }

    execution = {
        "execution/input": "unchanged",
    }

    _write_fake_downloader(tmp_path)

    monkeypatch.setattr(
        module,
        "get_workspace_paths",
        lambda root: SimpleNamespace(models=source_models),
    )
    monkeypatch.setattr(
        module,
        "production_contract",
        lambda root: contract.copy(),
    )
    monkeypatch.setattr(
        module,
        "execution_contract",
        lambda root: execution.copy(),
    )
    monkeypatch.setattr(
        module,
        "scientific_contract",
        lambda execution_contract, downloader_source: contract.copy(),
    )
    monkeypatch.setattr(
        module,
        "check_product_metadata",
        lambda raster_root, date, contract: None,
    )

    def fake_output_hashes(raster_root, date):
        raster, metadata, _ = module.product_paths(raster_root, date)
        return {
            "raster_sha256": sha256(raster),
            "metadata_sha256": sha256(metadata),
        }

    monkeypatch.setattr(module, "output_hashes", fake_output_hashes)

    called = []

    def run(command, cwd, env, check):
        called.append(command)

        assert Path(command[1]) == tmp_path / "scripts/produce_rf25_rasters.py"
        assert command[-4:] == [
            "--date",
            "2022-03-30",
            "--tile-size-m",
            "4000",
        ]

        workspace = Path(env[module.WORKSPACE_ENV_VAR])
        assert workspace.is_relative_to(tmp_path / "evaluation")

        assert all(
            sha256(workspace / "models" / name)
            == contract[f"models/{name}"]
            for name in names
        )

        if not succeeds:
            raise subprocess.CalledProcessError(1, command)

        raster, metadata, _ = module.product_paths(
            workspace / "rasters",
            "2022-03-30",
        )
        raster.parent.mkdir(parents=True, exist_ok=True)
        raster.write_bytes(b"canonical output stub")
        metadata.write_text(
            json.dumps({"period_start": "2022-03-30"}),
            encoding="utf-8",
        )

    monkeypatch.setattr(module.subprocess, "run", run)

    output = tmp_path / "evaluation"

    if succeeds:
        result = module.produce_field_date(
            tmp_path,
            output,
            "2022-03-30",
            "test-project",
            contract,
        )

        assert (
            module.product_status(result, "2022-03-30", contract)
            == "verified_final_product"
        )

        module.produce_field_date(
            tmp_path,
            output,
            "2022-03-30",
            "test-project",
            contract,
        )

        assert len(called) == 1

        inventory = module.inventory_products(
            ["2022-03-30", "2022-04-07"],
            [result],
            contract,
        )
        assert inventory.needs_canonical_production.tolist() == [
            False,
            True,
        ]

        evidence = json.loads(
            module.product_paths(result, "2022-03-30")[2].read_text(
                encoding="utf-8"
            )
        )
        assert evidence["protocol"] == "field_canonical_execution_v2"
        assert evidence["scientific_contract"] == contract
        assert evidence["completed"] is True

    else:
        with pytest.raises(subprocess.CalledProcessError):
            module.produce_field_date(
                tmp_path,
                output,
                "2022-03-30",
                "test-project",
                contract,
            )

        assert not list(output.rglob("field_execution_evidence.json"))

    assert all(
        sha256(source_models / name) == contract[f"models/{name}"]
        for name in names
    )


def test_v2_execution_evidence_rejects_output_tampering(
    tmp_path, monkeypatch
):
    date = "2022-03-30"
    raster_root = tmp_path / "rasters"
    raster, metadata, evidence_path = module.product_paths(
        raster_root, date
    )
    raster.parent.mkdir(parents=True)

    raster.write_bytes(b"original raster")
    metadata.write_bytes(b"original metadata")

    contract = {"scientific/input": "abc"}
    execution = {"execution/input": "xyz"}
    downloader_source = "reviewed downloader snapshot"

    monkeypatch.setattr(
        module,
        "scientific_contract",
        lambda execution_contract, source: contract.copy(),
    )
    monkeypatch.setattr(
        module,
        "check_product_metadata",
        lambda raster_root, date, contract: None,
    )

    def fake_output_hashes(root, period):
        current_raster, current_metadata, _ = module.product_paths(
            root, period
        )
        return {
            "raster_sha256": sha256(current_raster),
            "metadata_sha256": sha256(current_metadata),
        }

    monkeypatch.setattr(module, "output_hashes", fake_output_hashes)

    record = {
        "protocol": "field_canonical_execution_v2",
        "period_start": date,
        "completed": True,
        "input_sha256_before": execution.copy(),
        "input_sha256_after": execution.copy(),
        "scientific_contract": contract.copy(),
        "scientific_identity": module.contract_identity(contract),
        "downloader_source": downloader_source,
        "output_sha256": fake_output_hashes(raster_root, date),
    }

    module.verify_execution_evidence(
        raster_root,
        date,
        contract,
        record,
    )

    raster.write_bytes(b"tampered raster")

    with pytest.raises(ValueError, match="output or metadata SHA-256"):
        module.verify_execution_evidence(
            raster_root,
            date,
            contract,
            record,
        )
