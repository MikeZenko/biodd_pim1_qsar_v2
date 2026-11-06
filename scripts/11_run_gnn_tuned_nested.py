#!/usr/bin/env python3
"""Run or aggregate the post-hoc tuned graph benchmark on the compute VM."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
import urllib.request

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pim1_qsar.artifacts import sha256_file, sha256_json, sha256_named_files  # noqa: E402
from pim1_qsar.benchmark.gnn_tuned import (  # noqa: E402
    CHEMELEON_SHA256,
    FAMILIES,
    aggregate_results,
    load_search_protocol,
    run_outer_fold_family,
)


DEFAULT_SPLIT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "split_locks"
    / "chembl37_strict_temporal2018_scaffold3x5_v1"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "runs" / "gnn_tuned_posthoc_v1"
DEFAULT_SEARCH_CONFIG = PROJECT_ROOT / "configs" / "gnn_search_v1.yaml"
DEFAULT_CHEMELEON = Path.home() / ".chemprop" / "chemeleon_mp.pt"
DEFAULT_REFERENCE_PREDICTIONS = (
    PROJECT_ROOT / "artifacts" / "runs" / "development_selection_v2" / "mean_oof_predictions.csv"
)
CHEMELEON_URL = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--search-config", type=Path, default=DEFAULT_SEARCH_CONFIG)
    parser.add_argument("--chemeleon-checkpoint", type=Path, default=DEFAULT_CHEMELEON)
    parser.add_argument(
        "--reference-predictions", type=Path, default=DEFAULT_REFERENCE_PREDICTIONS
    )
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--repeat", type=int, choices=range(3))
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--download-chemeleon", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def ensure_checkpoint(path: Path, *, download: bool) -> None:
    if not path.is_file():
        if not download:
            raise FileNotFoundError(
                f"Missing CheMeleon checkpoint: {path}; pass --download-chemeleon"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".partial")
        urllib.request.urlretrieve(CHEMELEON_URL, temporary)
        temporary.replace(path)
    observed = sha256_file(path)
    if observed != CHEMELEON_SHA256:
        raise RuntimeError(
            f"CheMeleon checkpoint hash mismatch: expected {CHEMELEON_SHA256}, "
            f"observed {observed}"
        )


def main() -> None:
    args = parse_args()
    if not args.aggregate_only and None in (args.family, args.repeat, args.fold):
        raise SystemExit("A run requires --family, --repeat, and --fold")
    split_dir = args.split_dir.resolve()
    output_dir = args.output_dir.resolve()
    search_path = args.search_config.resolve()
    checkpoint_path = args.chemeleon_checkpoint.expanduser().resolve()
    reference_predictions_path = args.reference_predictions.resolve()
    ensure_checkpoint(checkpoint_path, download=args.download_chemeleon)
    _, protocol = load_search_protocol(search_path)
    if args.smoke:
        protocol = replace(
            protocol,
            candidate_budget=2,
            rerank_top=1,
            rerank_seeds=protocol.rerank_seeds[:2],
            refit_seeds=protocol.refit_seeds[:2],
            max_epochs=2,
            patience=1,
        )
    development_path = split_dir / "development.csv"
    assignments_path = split_dir / "outer_fold_assignments.csv"
    split_manifest_path = split_dir / "split_lock_manifest.json"
    environment_path = PROJECT_ROOT / "artifacts" / "environment" / "gnn.json"
    for path in (
        development_path,
        assignments_path,
        split_manifest_path,
        environment_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    source_hashes = {
        "development": sha256_file(development_path),
        "assignments": sha256_file(assignments_path),
        "split_lock": sha256_file(split_manifest_path),
        "search_config": sha256_file(search_path),
        "environment_manifest": sha256_file(environment_path),
        "requirements": sha256_file(PROJECT_ROOT / "requirements-gnn.txt"),
        "code": sha256_named_files(
            {
                "gnn_tuned.py": PROJECT_ROOT
                / "src"
                / "pim1_qsar"
                / "benchmark"
                / "gnn_tuned.py",
                "gine_tuned.py": PROJECT_ROOT
                / "src"
                / "pim1_qsar"
                / "models"
                / "gine_tuned.py",
                "gnn.py": PROJECT_ROOT
                / "src"
                / "pim1_qsar"
                / "benchmark"
                / "gnn.py",
                "gnn_features.py": PROJECT_ROOT
                / "src"
                / "pim1_qsar"
                / "models"
                / "gnn_features.py",
                "runner.py": Path(__file__),
            }
        ),
        "pretrained_checkpoint": sha256_file(checkpoint_path),
        "ridge_reference_predictions": sha256_file(reference_predictions_path),
    }
    contract = {
        "schema_version": "gnn_tuned_contract_v1",
        "status": "post_hoc_development_only",
        "source_hashes": source_hashes,
        "families": list(FAMILIES),
        "protocol": {
            "candidate_budget": protocol.candidate_budget,
            "inner_folds": protocol.inner_folds,
            "rerank_top": protocol.rerank_top,
            "rerank_seeds": list(protocol.rerank_seeds),
            "refit_seeds": list(protocol.refit_seeds),
            "max_epochs": protocol.max_epochs,
            "patience": protocol.patience,
            "min_delta": protocol.min_delta,
        },
        "temporal_test_sha256": None,
        "temporal_test_consulted": False,
        "smoke": bool(args.smoke),
    }
    contract_sha256 = sha256_json(contract)
    development = pd.read_csv(development_path)
    if args.aggregate_only:
        manifest = aggregate_results(
            development,
            output_dir=output_dir,
            protocol=protocol,
            contract_sha256=contract_sha256,
            source_hashes=source_hashes,
            search_config_sha256=sha256_file(search_path),
            checkpoint_sha256=sha256_file(checkpoint_path),
            reference_predictions_path=reference_predictions_path,
        )
        print(f"Aggregated tuned-GNN benchmark: {manifest['contract_sha256']}")
        return
    run_outer_fold_family(
        development,
        pd.read_csv(assignments_path),
        family=args.family,
        repeat=args.repeat,
        fold=args.fold,
        protocol=protocol,
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        contract_sha256=contract_sha256,
    )
    print(f"Completed {args.family} repeat={args.repeat} fold={args.fold}")


if __name__ == "__main__":
    main()
