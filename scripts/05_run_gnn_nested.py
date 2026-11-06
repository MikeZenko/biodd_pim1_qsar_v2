#!/usr/bin/env python3
"""Run resumable GINE and Chemprop D-MPNN benchmarks on the compute VM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pim1_qsar.artifacts import sha256_file, sha256_json, sha256_named_files  # noqa: E402
from pim1_qsar.benchmark.gnn import (  # noqa: E402
    GNNTrainingConfig,
    SUPPORTED_BACKENDS,
    SUPPORTED_VARIANTS,
    run_gnn_nested,
)


DEFAULT_SPLIT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "split_locks"
    / "chembl37_strict_temporal2018_scaffold3x5_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "runs" / "gnn_nested_v2",
    )
    parser.add_argument(
        "--backends", nargs="+", choices=SUPPORTED_BACKENDS, default=list(SUPPORTED_BACKENDS)
    )
    parser.add_argument(
        "--variants", nargs="+", choices=SUPPORTED_VARIANTS, default=list(SUPPORTED_VARIANTS)
    )
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_dir = args.split_dir.resolve()
    development_path = split_dir / "development.csv"
    assignments_path = split_dir / "outer_fold_assignments.csv"
    split_manifest_path = split_dir / "split_lock_manifest.json"
    for path in (development_path, assignments_path, split_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    config = GNNTrainingConfig(
        hidden_dim=args.hidden_dim,
        ffn_hidden_dim=args.hidden_dim,
        max_epochs=3 if args.smoke else args.max_epochs,
        patience=2 if args.smoke else args.patience,
        torch_threads=args.torch_threads,
    )
    source_hashes = {
        "development": sha256_file(development_path),
        "assignments": sha256_file(assignments_path),
        "split_lock": sha256_file(split_manifest_path),
        "protocol": sha256_file(PROJECT_ROOT / "configs" / "protocol_v2.yaml"),
        "environment": sha256_file(PROJECT_ROOT / "requirements-gnn.txt"),
        "code": sha256_named_files(
            {
                "features.py": PROJECT_ROOT / "src" / "pim1_qsar" / "features.py",
                "gnn.py": PROJECT_ROOT / "src" / "pim1_qsar" / "benchmark" / "gnn.py",
                "gine.py": PROJECT_ROOT / "src" / "pim1_qsar" / "models" / "gine.py",
                "gnn_features.py": PROJECT_ROOT
                / "src"
                / "pim1_qsar"
                / "models"
                / "gnn_features.py",
                "runner.py": Path(__file__),
            }
        ),
    }
    contract = {
        "schema": "gnn_nested_contract_v2",
        "source_hashes": source_hashes,
        "backends": list(args.backends),
        "variants": list(args.variants),
        "config": config.as_dict(),
        "smoke": bool(args.smoke),
        "temporal_test_sha256": None,
    }
    contract_sha256 = sha256_json(contract)
    result = run_gnn_nested(
        pd.read_csv(development_path),
        pd.read_csv(assignments_path),
        output_dir=args.output_dir,
        contract_sha256=contract_sha256,
        backends=args.backends,
        variants=args.variants,
        config=config,
        smoke=args.smoke,
    )
    print(result.model_summary.to_string(index=False))
    print(json.dumps({"contract_sha256": contract_sha256}, indent=2))
    print(f"Artifacts written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

