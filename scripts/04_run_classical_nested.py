#!/usr/bin/env python3
"""Run the prespecified classical benchmark on development data only."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pim1_qsar.artifacts import sha256_file, sha256_named_files  # noqa: E402
from pim1_qsar.benchmark.classical import (  # noqa: E402
    default_candidate_specs,
    run_classical_nested,
    write_classical_benchmark,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "split_locks"
        / "chembl37_strict_temporal2018_scaffold3x5_v1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "runs" / "classical_nested_v2",
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_dir = args.split_dir.resolve()
    development_path = split_dir / "development.csv"
    assignments_path = split_dir / "outer_fold_assignments.csv"
    split_manifest_path = split_dir / "split_lock_manifest.json"
    for path in (development_path, assignments_path, split_manifest_path):
        if not path.exists():
            raise FileNotFoundError(path)

    # The sealed temporal file is intentionally neither opened nor passed to
    # the benchmark.  Its existence is not required for development modeling.
    development = pd.read_csv(development_path)
    assignments = pd.read_csv(assignments_path)
    candidates = default_candidate_specs(smoke=args.smoke)
    result = run_classical_nested(
        development,
        assignments,
        candidates=candidates,
        n_jobs=args.n_jobs,
        smoke=args.smoke,
    )
    input_manifest = {
        "development_sha256": sha256_file(development_path),
        "outer_fold_assignments_sha256": sha256_file(assignments_path),
        "split_manifest_sha256": sha256_file(split_manifest_path),
        "protocol_sha256": sha256_file(PROJECT_ROOT / "configs" / "protocol_v2.yaml"),
        "code_sha256": sha256_named_files(
            {
                "features.py": PROJECT_ROOT / "src" / "pim1_qsar" / "features.py",
                "classical.py": PROJECT_ROOT
                / "src"
                / "pim1_qsar"
                / "benchmark"
                / "classical.py",
                "runner.py": Path(__file__),
            }
        ),
        "python": platform.python_version(),
        "smoke": bool(args.smoke),
        "temporal_test_sha256": None,
    }
    manifest = write_classical_benchmark(
        result,
        args.output_dir,
        input_manifest=input_manifest,
        candidate_manifest=[candidate.manifest() for candidate in candidates],
    )
    print(json.dumps(result.selection, indent=2, sort_keys=True))
    print(f"Run SHA-256: {manifest['run_sha256']}")
    print(f"Artifacts written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
