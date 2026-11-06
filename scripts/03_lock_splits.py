#!/usr/bin/env python3
"""Immutably lock the outcome-blind temporal and shared outer-CV splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pim1_qsar.validation.locking import lock_splits  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Lock the cutoff-2018 temporal split from strict measurement events and "
            "persist model-agnostic repeated scaffold folds. Existing locks are "
            "verified and never overwritten."
        )
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "configs" / "protocol_v2.yaml",
    )
    parser.add_argument(
        "--curated-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "curated" / "chembl37_pim1_v2",
        help=(
            "Directory containing curation_manifest.json, temporal_feasibility.csv, "
            "and strict_measurement_events.csv. Compound-level curation tables are "
            "never accepted as split inputs."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "split_locks",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lock_dir, manifest = lock_splits(
        protocol_path=args.protocol,
        curated_dir=args.curated_dir,
        output_root=args.output_root,
    )
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))
    print(f"Immutable split lock verified at {lock_dir}")


if __name__ == "__main__":
    main()
