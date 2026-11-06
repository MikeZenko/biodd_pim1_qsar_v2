from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pim1_qsar.artifacts import (  # noqa: E402
    sha256_file,
    sha256_json,
    sha256_named_files,
)


RUN_DIR = PROJECT_ROOT / "artifacts" / "runs" / "gnn_tuned_posthoc_v1"
COMPLETE_MARKER = RUN_DIR / "RUN_COMPLETE"
if not COMPLETE_MARKER.is_file():
    pytest.skip(
        "post-hoc tuned-GNN benchmark has not been aggregated",
        allow_module_level=True,
    )

SPLIT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "split_locks"
    / "chembl37_strict_temporal2018_scaffold3x5_v1"
)
DEVELOPMENT_PATH = SPLIT_DIR / "development.csv"
ASSIGNMENTS_PATH = SPLIT_DIR / "outer_fold_assignments.csv"
SPLIT_MANIFEST_PATH = SPLIT_DIR / "split_lock_manifest.json"
SEARCH_CONFIG_PATH = PROJECT_ROOT / "configs" / "gnn_search_v1.yaml"
ENVIRONMENT_PATH = PROJECT_ROOT / "artifacts" / "environment" / "gnn.json"
REQUIREMENTS_PATH = PROJECT_ROOT / "requirements-gnn.txt"
REFERENCE_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "development_selection_v2"
    / "mean_oof_predictions.csv"
)

FAMILIES = ("edge_aware_gine", "chemprop_dmpnn", "chemprop_chemeleon")
REPEATS = (0, 1, 2)
FOLDS = (0, 1, 2, 3, 4)
IDENTITY_COLUMN = "standardized_parent_inchi_key"
SCAFFOLD_COLUMN = "scaffold_smiles"
TARGET_COLUMN = "pIC50"
SCHEMA_VERSION = "gnn_tuned_posthoc_v1"
CHEMELEON_SHA256 = (
    "c376624d3407204e780a0ed13a9ac097cc9bb1c13ef89cdbc633c1715c183651"
)
OUTPUT_FILES = {
    "inner_trial_metrics.csv",
    "selected_hyperparameters.csv",
    "seed_oof_predictions.csv",
    "ensemble_oof_predictions.csv",
    "seed_metrics.csv",
    "repeat_metrics.csv",
    "model_summary.csv",
    "paired_scaffold_bootstrap.csv",
}


def _load_manifest() -> dict[str, object]:
    return json.loads((RUN_DIR / "run_manifest.json").read_text(encoding="utf-8"))


def _load_protocol() -> dict[str, object]:
    return yaml.safe_load(SEARCH_CONFIG_PATH.read_text(encoding="utf-8"))


def _regression_metrics(frame: pd.DataFrame) -> dict[str, float]:
    observed = frame["y_true"].to_numpy(dtype=float)
    predicted = frame["y_pred"].to_numpy(dtype=float)
    residual = predicted - observed
    total = np.square(observed - observed.mean()).sum()
    return {
        "rmse": float(np.sqrt(np.square(residual).mean())),
        "mae": float(np.abs(residual).mean()),
        "r2": float(1.0 - np.square(residual).sum() / total),
    }


def _assert_close(observed: float, expected: float) -> None:
    assert observed == pytest.approx(expected, rel=2e-10, abs=2e-10)


def test_manifest_reconstructs_contract_and_hashes_every_aggregate() -> None:
    manifest = _load_manifest()
    protocol = _load_protocol()
    source_hashes = manifest["source_hashes"]
    assert isinstance(source_hashes, dict)

    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["status"] == "post_hoc_development_only"
    assert manifest["temporal_test_consulted"] is False
    assert manifest["families"] == list(FAMILIES)
    assert manifest["pretrained_checkpoint_sha256"] == CHEMELEON_SHA256
    assert source_hashes["pretrained_checkpoint"] == CHEMELEON_SHA256

    code_hash = sha256_named_files(
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
            "gnn.py": PROJECT_ROOT / "src" / "pim1_qsar" / "benchmark" / "gnn.py",
            "gnn_features.py": PROJECT_ROOT
            / "src"
            / "pim1_qsar"
            / "models"
            / "gnn_features.py",
            "runner.py": PROJECT_ROOT / "scripts" / "11_run_gnn_tuned_nested.py",
        }
    )
    expected_sources = {
        "development": sha256_file(DEVELOPMENT_PATH),
        "assignments": sha256_file(ASSIGNMENTS_PATH),
        "split_lock": sha256_file(SPLIT_MANIFEST_PATH),
        "search_config": sha256_file(SEARCH_CONFIG_PATH),
        "environment_manifest": sha256_file(ENVIRONMENT_PATH),
        "requirements": sha256_file(REQUIREMENTS_PATH),
        "code": code_hash,
        "pretrained_checkpoint": CHEMELEON_SHA256,
        "ridge_reference_predictions": sha256_file(REFERENCE_PATH),
    }
    assert source_hashes == expected_sources
    assert manifest["search_config_sha256"] == expected_sources["search_config"]
    assert (
        manifest["reference_predictions_sha256"]
        == expected_sources["ridge_reference_predictions"]
    )

    contract = {
        "schema_version": "gnn_tuned_contract_v1",
        "status": "post_hoc_development_only",
        "source_hashes": source_hashes,
        "families": list(FAMILIES),
        "protocol": {
            "candidate_budget": protocol["candidate_budget_per_family"],
            "inner_folds": protocol["inner_scaffold_folds"],
            "rerank_top": protocol["rerank_top_candidates"],
            "rerank_seeds": protocol["rerank_initialization_seeds"],
            "refit_seeds": protocol["outer_refit_initialization_seeds"],
            "max_epochs": protocol["max_epochs"],
            "patience": protocol["patience"],
            "min_delta": protocol["min_delta_rmse"],
        },
        "temporal_test_sha256": None,
        "temporal_test_consulted": False,
        "smoke": False,
    }
    assert manifest["contract_sha256"] == sha256_json(contract)
    assert COMPLETE_MARKER.read_text(encoding="utf-8").strip() == manifest[
        "contract_sha256"
    ]

    output_hashes = manifest["output_hashes"]
    assert isinstance(output_hashes, dict)
    assert set(output_hashes) == OUTPUT_FILES
    for filename, expected_hash in output_hashes.items():
        assert sha256_file(RUN_DIR / filename) == expected_hash


def test_raw_trial_selection_and_outer_prediction_inventory_is_complete() -> None:
    manifest = _load_manifest()
    protocol = _load_protocol()
    development = pd.read_csv(DEVELOPMENT_PATH)
    assignments = pd.read_csv(ASSIGNMENTS_PATH)
    contract_sha256 = manifest["contract_sha256"]
    rerank_seeds = tuple(protocol["rerank_initialization_seeds"])
    refit_seeds = tuple(protocol["outer_refit_initialization_seeds"])
    candidate_budget = int(protocol["candidate_budget_per_family"])
    rerank_top = int(protocol["rerank_top_candidates"])
    expected_trials_per_outer_fold = candidate_budget + rerank_top * (
        len(rerank_seeds) - 1
    )

    expected_stems = {
        f"{family}__repeat-{repeat}__fold-{fold}"
        for family in FAMILIES
        for repeat in REPEATS
        for fold in FOLDS
    }
    selection_paths = sorted((RUN_DIR / "selections").glob("*.json"))
    prediction_paths = sorted((RUN_DIR / "outer_predictions").glob("*.csv"))
    assert {path.stem for path in selection_paths} == expected_stems
    assert {path.stem for path in prediction_paths} == expected_stems

    for selection_path in selection_paths:
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        family = selection["family"]
        repeat = int(selection["repeat"])
        fold = int(selection["fold"])
        stem = f"{family}__repeat-{repeat}__fold-{fold}"
        prediction_path = RUN_DIR / "outer_predictions" / f"{stem}.csv"
        expected_validation = assignments.loc[
            assignments["repeat"].eq(repeat)
            & assignments["fold"].eq(fold)
            & assignments["role"].eq("validation"),
            IDENTITY_COLUMN,
        ].astype(str)

        assert selection["schema_version"] == SCHEMA_VERSION
        assert selection["contract_sha256"] == contract_sha256
        assert selection["temporal_test_consulted"] is False
        assert selection["outer_prediction_sha256"] == sha256_file(prediction_path)
        assert selection["n_outer_train"] + selection["n_outer_validation"] == len(
            development
        )
        assert selection["n_outer_validation"] == len(expected_validation)
        assert len(selection["screening"]) == candidate_budget
        assert len({row["candidate_id"] for row in selection["screening"]}) == (
            candidate_budget
        )
        assert len(selection["reranking"]) == rerank_top
        assert selection["selected_candidate_id"] == min(
            selection["reranking"],
            key=lambda row: (row["seed_mean_inner_rmse"], row["candidate_id"]),
        )["candidate_id"]
        assert selection["selected_candidate"]["family"] == family
        assert selection["selected_candidate_id"].startswith(f"{family}-")
        for row in selection["reranking"]:
            assert {result["seed"] for result in row["seed_results"]} == set(
                rerank_seeds
            )

        predictions = pd.read_csv(prediction_path)
        assert len(predictions) == len(expected_validation) * len(refit_seeds)
        assert set(predictions["family"]) == {family}
        assert set(predictions["repeat"]) == {repeat}
        assert set(predictions["fold"]) == {fold}
        assert set(predictions["seed"]) == set(refit_seeds)
        assert set(predictions["candidate_id"]) == {
            selection["selected_candidate_id"]
        }
        assert not predictions.duplicated(["seed", IDENTITY_COLUMN]).any()
        for _, seed_frame in predictions.groupby("seed"):
            assert set(seed_frame[IDENTITY_COLUMN].astype(str)) == set(
                expected_validation
            )
        assert np.isfinite(predictions[["y_true", "y_pred"]]).all().all()
        assert (predictions["refit_epochs"] > 0).all()

    trial_paths = sorted((RUN_DIR / "trials").glob("*.json"))
    assert len(trial_paths) == manifest["trial_count"] == 1350
    trial_inventory: list[tuple[str, int, int, str, int]] = []
    for path in trial_paths:
        trial = json.loads(path.read_text(encoding="utf-8"))
        assert trial["schema_version"] == SCHEMA_VERSION
        assert trial["contract_sha256"] == contract_sha256
        assert trial["temporal_test_consulted"] is False
        assert trial["family"] in FAMILIES
        assert trial["candidate"]["family"] == trial["family"]
        assert trial["candidate_id"].startswith(f"{trial['family']}-")
        assert [row["inner_fold"] for row in trial["inner_folds"]] == [0, 1, 2]
        assert all(math.isfinite(float(row["rmse"])) for row in trial["inner_folds"])
        assert all(int(row["selected_epoch"]) > 0 for row in trial["inner_folds"])
        assert math.isfinite(float(trial["pooled_inner_rmse"]))
        assert float(trial["runtime_seconds"]) >= 0
        trial_inventory.append(
            (
                trial["family"],
                int(trial["repeat"]),
                int(trial["fold"]),
                trial["candidate_id"],
                int(trial["seed"]),
            )
        )
    assert len(set(trial_inventory)) == len(trial_inventory)
    inventory = pd.DataFrame(
        trial_inventory,
        columns=["family", "repeat", "fold", "candidate_id", "seed"],
    )
    for _, group in inventory.groupby(["family", "repeat", "fold"]):
        assert len(group) == expected_trials_per_outer_fold == 30
        assert group["candidate_id"].nunique() == candidate_budget
        assert group["seed"].value_counts().to_dict() == {
            rerank_seeds[0]: candidate_budget,
            rerank_seeds[1]: rerank_top,
            rerank_seeds[2]: rerank_top,
        }
        per_candidate = sorted(group.groupby("candidate_id").size())
        assert per_candidate == [1] * (candidate_budget - rerank_top) + [3] * rerank_top


def test_aggregate_tables_have_exact_rows_keys_and_oof_coverage() -> None:
    manifest = _load_manifest()
    protocol = _load_protocol()
    development = pd.read_csv(DEVELOPMENT_PATH)
    assignments = pd.read_csv(ASSIGNMENTS_PATH)
    n_development = len(development)
    n_families = len(FAMILIES)
    n_repeats = len(REPEATS)
    n_refit_seeds = len(protocol["outer_refit_initialization_seeds"])

    inner = pd.read_csv(RUN_DIR / "inner_trial_metrics.csv")
    selected = pd.read_csv(RUN_DIR / "selected_hyperparameters.csv")
    seed_oof = pd.read_csv(RUN_DIR / "seed_oof_predictions.csv")
    ensemble = pd.read_csv(RUN_DIR / "ensemble_oof_predictions.csv")
    seed_metrics = pd.read_csv(RUN_DIR / "seed_metrics.csv")
    repeat_metrics = pd.read_csv(RUN_DIR / "repeat_metrics.csv")
    model_summary = pd.read_csv(RUN_DIR / "model_summary.csv")
    bootstrap = pd.read_csv(RUN_DIR / "paired_scaffold_bootstrap.csv")

    assert len(inner) == manifest["inner_fit_count"] == 4050
    assert not inner.duplicated(
        ["family", "repeat", "fold", "candidate_id", "seed", "inner_fold"]
    ).any()
    assert set(inner["inner_fold"]) == {0, 1, 2}
    assert np.isfinite(
        inner[
            [
                "rmse",
                "pooled_inner_rmse",
                "runtime_seconds",
                "selected_epoch",
                "parameter_count",
            ]
        ]
    ).all().all()

    assert len(selected) == n_families * n_repeats * len(FOLDS) == 45
    assert not selected.duplicated(["family", "repeat", "fold"]).any()
    assert set(selected["family"]) == set(FAMILIES)
    assert set(selected["repeat"]) == set(REPEATS)
    assert set(selected["fold"]) == set(FOLDS)

    assert len(seed_oof) == n_development * n_families * n_repeats * n_refit_seeds
    assert not seed_oof.duplicated(
        ["family", "repeat", "fold", "seed", IDENTITY_COLUMN]
    ).any()
    assert set(seed_oof["family"]) == set(FAMILIES)
    assert set(seed_oof["repeat"]) == set(REPEATS)
    assert set(seed_oof["seed"]) == set(protocol["outer_refit_initialization_seeds"])
    assert np.isfinite(seed_oof[["y_true", "y_pred"]]).all().all()

    expected_validation = assignments.loc[
        assignments["role"].eq("validation"),
        ["repeat", "fold", IDENTITY_COLUMN],
    ].drop_duplicates()
    observed_validation = seed_oof[
        ["repeat", "fold", IDENTITY_COLUMN]
    ].drop_duplicates()
    pd.testing.assert_frame_equal(
        observed_validation.sort_values(list(observed_validation)).reset_index(drop=True),
        expected_validation.sort_values(list(expected_validation)).reset_index(drop=True),
        check_dtype=False,
    )
    expected_targets = development.set_index(IDENTITY_COLUMN)[TARGET_COLUMN]
    assert np.allclose(
        seed_oof["y_true"],
        seed_oof[IDENTITY_COLUMN].map(expected_targets),
        rtol=0,
        atol=2e-10,
    )

    assert len(ensemble) == n_development * n_families * n_repeats
    assert not ensemble.duplicated(["family", "repeat", IDENTITY_COLUMN]).any()
    assert (ensemble.groupby(["family", "repeat"]).size() == n_development).all()
    assert np.isfinite(
        ensemble[["y_true", "y_pred", "y_pred_seed_sd"]]
    ).all().all()
    assert (ensemble["y_pred_seed_sd"] >= 0).all()

    assert len(seed_metrics) == n_families * n_repeats * n_refit_seeds == 27
    assert not seed_metrics.duplicated(["family", "repeat", "seed"]).any()
    assert set(seed_metrics["n_oof"]) == {n_development}
    assert len(repeat_metrics) == n_families * n_repeats == 9
    assert not repeat_metrics.duplicated(["family", "repeat"]).any()
    assert set(repeat_metrics["n_oof"]) == {n_development}
    assert len(model_summary) == n_families
    assert set(model_summary["family"]) == set(FAMILIES)
    assert set(model_summary["n_repeats"]) == {n_repeats}
    assert set(model_summary["candidate_budget_per_outer_fold"]) == {
        protocol["candidate_budget_per_family"]
    }
    assert set(model_summary["outer_refit_seeds"]) == {n_refit_seeds}

    assert len(bootstrap) == n_families * 2 == 6
    assert not bootstrap.duplicated(["family", "metric"]).any()
    assert set(bootstrap["family"]) == set(FAMILIES)
    assert set(bootstrap["metric"]) == {"RMSE", "MAE"}
    assert set(bootstrap["reference_model"]) == {"ridge_combined"}
    assert set(bootstrap["bootstrap_replicates"]) == {10000}
    assert set(bootstrap["cluster_count"]) == {
        development[SCAFFOLD_COLUMN].astype(str).nunique()
    }
    assert np.isfinite(
        bootstrap[["point_difference", "ci_lower", "ci_upper"]]
    ).all().all()
    assert (bootstrap["ci_lower"] <= bootstrap["ci_upper"]).all()


def test_aggregate_predictions_and_metrics_are_internally_reproducible() -> None:
    seed_oof = pd.read_csv(RUN_DIR / "seed_oof_predictions.csv")
    ensemble = pd.read_csv(RUN_DIR / "ensemble_oof_predictions.csv")
    seed_metrics = pd.read_csv(RUN_DIR / "seed_metrics.csv")
    repeat_metrics = pd.read_csv(RUN_DIR / "repeat_metrics.csv")
    model_summary = pd.read_csv(RUN_DIR / "model_summary.csv")
    bootstrap = pd.read_csv(RUN_DIR / "paired_scaffold_bootstrap.csv")

    reproduced_ensemble = (
        seed_oof.groupby(
            ["family", "repeat", "fold", IDENTITY_COLUMN, SCAFFOLD_COLUMN, "y_true"],
            as_index=False,
            sort=True,
        )["y_pred"]
        .agg(y_pred="mean", y_pred_seed_sd=lambda values: np.std(values, ddof=0))
    )
    compared = ensemble.merge(
        reproduced_ensemble,
        on=["family", "repeat", "fold", IDENTITY_COLUMN, SCAFFOLD_COLUMN, "y_true"],
        how="outer",
        suffixes=("", "_reproduced"),
        validate="one_to_one",
        indicator=True,
    )
    assert set(compared["_merge"]) == {"both"}
    assert np.allclose(
        compared["y_pred"], compared["y_pred_reproduced"], rtol=2e-10, atol=2e-10
    )
    assert np.allclose(
        compared["y_pred_seed_sd"],
        compared["y_pred_seed_sd_reproduced"],
        rtol=2e-10,
        atol=2e-10,
    )

    for keys, frame in seed_oof.groupby(["family", "repeat", "seed"]):
        row = seed_metrics.loc[
            seed_metrics["family"].eq(keys[0])
            & seed_metrics["repeat"].eq(keys[1])
            & seed_metrics["seed"].eq(keys[2])
        ].squeeze()
        assert int(row["n_oof"]) == len(frame)
        for metric, expected in _regression_metrics(frame).items():
            _assert_close(float(row[metric]), expected)

    for keys, frame in ensemble.groupby(["family", "repeat"]):
        row = repeat_metrics.loc[
            repeat_metrics["family"].eq(keys[0])
            & repeat_metrics["repeat"].eq(keys[1])
        ].squeeze()
        assert int(row["n_oof"]) == len(frame)
        for metric, expected in _regression_metrics(frame).items():
            _assert_close(float(row[metric]), expected)
        _assert_close(
            float(row["mean_seed_prediction_sd"]),
            float(frame["y_pred_seed_sd"].mean()),
        )

    for family, frame in repeat_metrics.groupby("family"):
        row = model_summary.loc[model_summary["family"].eq(family)].squeeze()
        for metric in ("rmse", "mae", "r2"):
            _assert_close(float(row[f"{metric}_mean"]), float(frame[metric].mean()))
            _assert_close(
                float(row[f"{metric}_sd"]), float(frame[metric].std(ddof=0))
            )
        _assert_close(
            float(row["mean_seed_prediction_sd"]),
            float(frame["mean_seed_prediction_sd"].mean()),
        )

    reference = pd.read_csv(REFERENCE_PATH)
    reference = reference.loc[reference["model"].eq("ridge_combined")].copy()
    tuned_mean = (
        ensemble.groupby(
            ["family", IDENTITY_COLUMN, SCAFFOLD_COLUMN, "y_true"],
            as_index=False,
            sort=True,
        )["y_pred"]
        .mean()
    )
    for family, frame in tuned_mean.groupby("family"):
        paired = frame.merge(
            reference[[IDENTITY_COLUMN, "y_true", "y_pred"]].rename(
                columns={"y_true": "reference_y_true", "y_pred": "reference_y_pred"}
            ),
            on=IDENTITY_COLUMN,
            validate="one_to_one",
        )
        tuned_metrics = _regression_metrics(paired)
        reference_metrics = _regression_metrics(
            paired.rename(
                columns={
                    "y_true": "tuned_y_true",
                    "reference_y_true": "y_true",
                    "y_pred": "tuned_y_pred",
                    "reference_y_pred": "y_pred",
                }
            )
        )
        for metric in ("RMSE", "MAE"):
            row = bootstrap.loc[
                bootstrap["family"].eq(family) & bootstrap["metric"].eq(metric)
            ].squeeze()
            expected = tuned_metrics[metric.lower()] - reference_metrics[metric.lower()]
            _assert_close(float(row["point_difference"]), expected)
