"""Single entry point for the clean RF-25 ET Fundación repository."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import et_downscaling
from et_downscaling.workspace import require_portable_inputs


DEFAULT_DATES = ["2020-03-13", "2021-11-25", "2022-03-30"]


def root() -> Path:
    return Path(__file__).resolve().parents[1]


def validate_imported_package_root(repository_root: Path) -> Path:
    """Fail before execution if et_downscaling comes from another checkout."""
    expected = (Path(repository_root) / "src" / "et_downscaling").resolve()
    imported = Path(et_downscaling.__file__).resolve()
    try:
        imported.relative_to(expected)
    except ValueError:
        raise RuntimeError(
            "The imported et_downscaling package belongs to a different repository.\n"
            f"Expected: {expected}\n"
            f"Imported: {imported}\n"
            "No pipeline work was started.\n"
            "Run `python -m pip install -e .` from this repository first."
        ) from None
    return imported


def run_script(name: str, arguments: list[str]) -> None:
    command = [sys.executable, str(root() / "scripts" / name), *arguments]
    print("\n>", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=root(), check=True)


def clean_outputs() -> None:
    output = root() / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    for child in output.iterdir():
        if child.name == "README.md":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def preflight() -> None:
    inputs = require_portable_inputs(root())
    print("Repository:", root())
    print("Portable inputs:")
    for name, path in inputs.items():
        print(f"  {name}: {path}")
    print("Output root:", root() / "outputs")
    print("Candidate archive: all implemented predictor families on Virtual10")
    print("Final model: RF-25, fixed 25 predictors, no tuning")
    print("AOA: RF-weighted DI + spatial-CV threshold; LPD diagnostic")
    print("Google Drive: not used")


def run_core(project: str, dates: list[str]) -> None:
    run_script(
        "select_virtual_stations.py",
        ["--project", project, "--seed", "42", "--n-supports", "10", "--min-ge90-per-year", "20", "--max-candidates", "9123"],
    )
    run_script(
        "download_all_candidate_predictors.py",
        ["--project", project],
    )
    run_script("build_rf25_training_population.py", [])
    run_script("train_rf25.py", [])
    production_args = ["--project", project]
    for value in dates:
        production_args += ["--date", value]
    run_script("produce_rf25_rasters.py", production_args)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("preflight")

    for name in ("fresh", "run"):
        item = sub.add_parser(name)
        item.add_argument("--project", required=True)
        item.add_argument("--date", dest="dates", action="append", default=None)
        if name == "fresh":
            item.add_argument(
                "--yes",
                action="store_true",
                help="Required acknowledgement that generated outputs will be deleted before the run.",
            )

    select = sub.add_parser("select")
    select.add_argument("--project", required=True)

    extract = sub.add_parser("extract")
    extract.add_argument("--project", required=True)
    extract.add_argument("--force", action="store_true")

    sub.add_parser("train")

    candidates = sub.add_parser("download-candidates")
    candidates.add_argument("--project", required=True)

    produce = sub.add_parser("produce")
    produce.add_argument("--project", required=True)
    produce.add_argument("--date", dest="dates", action="append", default=None)

    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def main() -> None:
    validate_imported_package_root(root())
    parser = build_parser()
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return
    if args.command == "preflight":
        preflight()
        return
    if args.command == "fresh":
        if not args.yes:
            raise SystemExit("fresh requires --yes because it deletes generated outputs/ contents.")
        preflight()
        clean_outputs()
        dates = args.dates or DEFAULT_DATES
        run_core(args.project, dates)
        return
    if args.command == "run":
        preflight()
        dates = args.dates or DEFAULT_DATES
        run_core(args.project, dates)
        return
    if args.command == "select":
        run_script("select_virtual_stations.py", ["--project", args.project, "--seed", "42", "--n-supports", "10", "--min-ge90-per-year", "20", "--max-candidates", "9123"])
        return
    if args.command == "extract":
        if args.force:
            print("Note: --force is implicit for comprehensive candidate re-downloads.")
        run_script("download_all_candidate_predictors.py", ["--project", args.project])
        run_script("build_rf25_training_population.py", [])
        return
    if args.command == "train":
        run_script("train_rf25.py", [])
        return
    if args.command == "download-candidates":
        run_script("download_all_candidate_predictors.py", ["--project", args.project])
        return
    if args.command == "produce":
        cmd = ["--project", args.project]
        for value in args.dates or DEFAULT_DATES:
            cmd += ["--date", value]
        run_script("produce_rf25_rasters.py", cmd)
        return
    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
