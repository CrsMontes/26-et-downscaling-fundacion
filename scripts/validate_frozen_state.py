"""Validate the preserved operational V5 products locally, without regeneration."""

from __future__ import annotations

import argparse
import json

from et_downscaling.virtual_station import resolve_virtual_workspace, validate_frozen_v5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", default=None)
    args = parser.parse_args()
    report = validate_frozen_v5(resolve_virtual_workspace(args.workspace_root))
    print(json.dumps(report, indent=2))
    print("V5 frozen selection, population and existing rasters: PASS")


if __name__ == "__main__":
    main()
