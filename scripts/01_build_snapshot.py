#!/usr/bin/env python3
"""Build the immutable ChEMBL 37 PIM1 input snapshot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pim1_qsar.data import SnapshotConfig, build_snapshot  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "releases" / "chembl37_pim1_20260501",
    )
    parser.add_argument("--expected-release", default="ChEMBL_37")
    parser.add_argument("--expected-release-date", default="2026-05-01")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SnapshotConfig(
        expected_release=args.expected_release,
        expected_release_date=args.expected_release_date,
    )
    manifest = build_snapshot(args.output_dir, config)
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))
    print(f"Snapshot written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
