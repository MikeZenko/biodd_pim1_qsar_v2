#!/usr/bin/env python3
"""Evaluate the selected model once on the sealed temporal test partition."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pim1_qsar.artifacts import (  # noqa: E402
    atomic_write_json,
    atomic_write_text,
    sha256_file,
    sha256_json,
    sha256_named_files,
)
from pim1_qsar.benchmark.evaluation import (  # noqa: E402
    evaluate_temporal_predictions,
    fit_selected_and_predict,
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
        "--selection-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "runs" / "development_selection_v2",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "runs" / "temporal_evaluation_v2",
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--bootstrap", type=int, default=10000)
    return parser.parse_args()


def _verify_completed(output_dir: Path) -> dict:
    manifest_path = output_dir / "temporal_evaluation_manifest.json"
    marker = output_dir / "RUN_COMPLETE"
    if not manifest_path.is_file() or not marker.is_file():
        raise RuntimeError("Temporal evaluation directory exists but is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for filename, expected in manifest.get("output_hashes", {}).items():
        path = output_dir / filename
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Completed temporal artifact failed verification: {filename}")
    return manifest


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    # A completed evaluation is only verified; the sealed table is not opened
    # again. Any incomplete directory is an audit failure requiring review.
    if output_dir.exists():
        manifest = _verify_completed(output_dir)
        print(json.dumps(manifest["primary_result"], indent=2, sort_keys=True))
        print("Existing one-time temporal evaluation verified; no labels were reread.")
        return

    split_dir = args.split_dir.resolve()
    selection_dir = args.selection_dir.resolve()
    paths = {
        "development": split_dir / "development.csv",
        "temporal": split_dir / "temporal_test_sealed.csv",
        "split_lock": split_dir / "split_lock_manifest.json",
        "selection": selection_dir / "model_selection.json",
        "all_oof": selection_dir / "all_oof_predictions.csv",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    selection = json.loads(paths["selection"].read_text(encoding="utf-8"))
    if selection.get("temporal_test_consulted") is not False:
        raise RuntimeError("Model selection is not certified temporal-test blind")
    selected_model = str(selection["selected_model"])

    # Development labels and OOF residuals are loaded before the one-time
    # temporal read. The output is built in a staging directory and renamed
    # atomically only after every diagnostic and hash succeeds.
    development = pd.read_csv(paths["development"])
    all_oof = pd.read_csv(paths["all_oof"])
    selected_oof_repeat = all_oof.loc[
        all_oof["model"].astype(str).eq(selected_model)
        & pd.to_numeric(all_oof["repeat"]).eq(0)
    ].copy()
    if len(selected_oof_repeat) != len(development):
        raise RuntimeError("Selected-model repeat-0 OOF calibration inventory is incomplete")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    atomic_write_json(
        staging_dir / "EVALUATION_STARTED.json",
        {
            "selected_model": selected_model,
            "selection_sha256": sha256_file(paths["selection"]),
            "temporal_table_expected_sha256": json.loads(
                paths["split_lock"].read_text(encoding="utf-8")
            )["output_files_sha256"]["temporal_test_sealed.csv"],
            "started_utc": datetime.now(timezone.utc).isoformat(),
        },
    )

    # This is the single permitted label read for the selected model.
    temporal_test = pd.read_csv(paths["temporal"])
    final_prediction = fit_selected_and_predict(
        selected_model,
        development,
        temporal_test,
        model_dir=staging_dir / "model",
        n_jobs=args.n_jobs,
    )
    result = evaluate_temporal_predictions(
        development,
        temporal_test,
        selected_oof_repeat,
        final_prediction,
        n_bootstrap=args.bootstrap,
    )
    outputs = {
        "temporal_predictions.csv": result["predictions"].to_csv(
            index=False, lineterminator="\n", float_format="%.12g"
        ),
        "calibration_bins.csv": result["calibration_bins"].to_csv(
            index=False, lineterminator="\n", float_format="%.12g"
        ),
    }
    for filename, text in outputs.items():
        atomic_write_text(staging_dir / filename, text)
    atomic_write_json(staging_dir / "temporal_metrics.json", result["diagnostics"])
    atomic_write_json(staging_dir / "final_training_manifest.json", result["training_manifest"])
    output_inventory = [
        "EVALUATION_STARTED.json",
        "temporal_predictions.csv",
        "calibration_bins.csv",
        "temporal_metrics.json",
        "final_training_manifest.json",
    ]
    if final_prediction.model_artifact is not None:
        output_inventory.append(
            str(final_prediction.model_artifact.relative_to(staging_dir))
        )
    manifest_core = {
        "schema_version": "temporal_evaluation_manifest_v2",
        "status": "complete",
        "one_time_selected_model_evaluation": True,
        "selected_model": selected_model,
        "input_hashes": {
            "development_sha256": sha256_file(paths["development"]),
            "temporal_test_sealed_sha256": sha256_file(paths["temporal"]),
            "split_lock_manifest_sha256": sha256_file(paths["split_lock"]),
            "model_selection_sha256": sha256_file(paths["selection"]),
            "selected_oof_sha256": sha256_file(paths["all_oof"]),
            "protocol_sha256": sha256_file(PROJECT_ROOT / "configs" / "protocol_v2.yaml"),
            "evaluation_code_sha256": sha256_named_files(
                {
                    "evaluation.py": PROJECT_ROOT
                    / "src"
                    / "pim1_qsar"
                    / "benchmark"
                    / "evaluation.py",
                    "runner.py": Path(__file__),
                }
            ),
        },
        "primary_result": result["diagnostics"]["primary_metrics"],
        "output_hashes": {
            filename: sha256_file(staging_dir / filename)
            for filename in sorted(output_inventory)
        },
    }
    manifest = dict(manifest_core)
    manifest["evaluation_sha256"] = sha256_json(manifest_core)
    atomic_write_json(staging_dir / "temporal_evaluation_manifest.json", manifest)
    atomic_write_text(staging_dir / "RUN_COMPLETE", manifest["evaluation_sha256"] + "\n")
    os.replace(staging_dir, output_dir)
    print(json.dumps(manifest["primary_result"], indent=2, sort_keys=True))
    print(f"One-time temporal evaluation sealed at {output_dir}")


if __name__ == "__main__":
    main()

