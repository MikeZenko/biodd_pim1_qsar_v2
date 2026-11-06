#!/usr/bin/env python3
"""Select one model using development OOF results; never read the temporal test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pim1_qsar.artifacts import sha256_file  # noqa: E402
from pim1_qsar.benchmark.selection import (  # noqa: E402
    select_development_model,
    write_development_selection,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classical-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "runs" / "classical_nested_v2",
    )
    parser.add_argument(
        "--gnn-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "runs" / "gnn_nested_v2",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "runs" / "development_selection_v2",
    )
    parser.add_argument("--bootstrap", type=int, default=10000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    classical_path = args.classical_dir.resolve() / "oof_predictions.csv"
    gnn_path = args.gnn_dir.resolve() / "oof_predictions.csv"
    for path in (classical_path, gnn_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    result = select_development_model(
        pd.read_csv(classical_path),
        pd.read_csv(gnn_path),
        n_bootstrap=args.bootstrap,
    )
    manifest = write_development_selection(
        result,
        args.output_dir,
        input_hashes={
            "classical_oof_sha256": sha256_file(classical_path),
            "gnn_oof_sha256": sha256_file(gnn_path),
            "protocol_sha256": sha256_file(PROJECT_ROOT / "configs" / "protocol_v2.yaml"),
        },
    )
    print(json.dumps(result["selection"], indent=2, sort_keys=True))
    print(f"Selection SHA-256: {manifest['selection_sha256']}")


if __name__ == "__main__":
    main()

