#!/usr/bin/env python3
"""Curate the frozen ChEMBL 37 PIM1 snapshot without selecting a split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pim1_qsar.data.curation import CurationConfig, curate_snapshot  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "releases" / "chembl37_pim1_20260501",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "curated" / "chembl37_pim1_v2",
    )
    parser.add_argument("--high-potency-threshold", type=float, default=10.0)
    parser.add_argument("--strict-confidence-score", type=int, default=9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CurationConfig(
        high_potency_threshold=args.high_potency_threshold,
        strict_confidence_score=args.strict_confidence_score,
    )
    manifest = curate_snapshot(args.snapshot_dir, args.output_dir, config=config)
    print(json.dumps(manifest["cohort_counts"], indent=2, sort_keys=True))
    print(
        "Temporal feasibility counts were written for every available cutoff; "
        "no cutoff was selected."
    )
    print(f"Curated data written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

