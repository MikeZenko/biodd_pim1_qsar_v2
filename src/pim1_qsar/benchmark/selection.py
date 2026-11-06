"""Development-only model ranking across classical and graph candidates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from pim1_qsar.artifacts import atomic_write_json, atomic_write_text, sha256_file, sha256_json
from pim1_qsar.benchmark.classical import (
    IDENTITY_COLUMN,
    SCAFFOLD_COLUMN,
    regression_metrics,
)
from pim1_qsar.validation.bootstrap import sample_cluster_weights


SELECTION_SCHEMA_VERSION = "development_selection_v2"


def _validate_oof_table(
    frame: pd.DataFrame,
    *,
    expected_repeats: Sequence[int],
) -> pd.DataFrame:
    required = {
        "model",
        "repeat",
        "fold",
        IDENTITY_COLUMN,
        SCAFFOLD_COLUMN,
        "y_true",
        "y_pred",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"OOF predictions are missing: {', '.join(missing)}")
    normalized = frame.copy()
    normalized["model"] = normalized["model"].astype(str)
    normalized[IDENTITY_COLUMN] = normalized[IDENTITY_COLUMN].astype(str)
    normalized[SCAFFOLD_COLUMN] = normalized[SCAFFOLD_COLUMN].astype(str)
    normalized["repeat"] = pd.to_numeric(normalized["repeat"], errors="raise").astype(int)
    normalized["fold"] = pd.to_numeric(normalized["fold"], errors="raise").astype(int)
    for column in ("y_true", "y_pred"):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
        if not np.isfinite(normalized[column].to_numpy(dtype=float)).all():
            raise ValueError(f"OOF {column} values must be finite")
    if set(normalized["repeat"]) != set(expected_repeats):
        raise ValueError("OOF repeat inventory does not match the locked protocol")
    for model, group in normalized.groupby("model", sort=True):
        if set(group["repeat"]) != set(expected_repeats):
            raise ValueError(
                f"OOF repeat inventory does not match the locked protocol for {model}"
            )
    if normalized.duplicated(["model", "repeat", IDENTITY_COLUMN]).any():
        raise ValueError("A model has duplicate OOF predictions within a repeat")
    inventory: tuple[str, ...] | None = None
    target_by_identity: pd.Series | None = None
    scaffold_by_identity: pd.Series | None = None
    for (model, repeat), group in normalized.groupby(["model", "repeat"], sort=True):
        identities = tuple(sorted(group[IDENTITY_COLUMN]))
        if inventory is None:
            inventory = identities
        elif identities != inventory:
            raise ValueError(f"OOF identity inventory differs for {model}, repeat {repeat}")
        indexed = group.set_index(IDENTITY_COLUMN).sort_index()
        if target_by_identity is None:
            target_by_identity = indexed["y_true"]
            scaffold_by_identity = indexed[SCAFFOLD_COLUMN]
        else:
            np.testing.assert_allclose(indexed["y_true"], target_by_identity)
            if not indexed[SCAFFOLD_COLUMN].equals(scaffold_by_identity):
                raise ValueError("OOF scaffold identities disagree across models")
    return normalized.sort_values(
        ["model", "repeat", IDENTITY_COLUMN], kind="mergesort"
    ).reset_index(drop=True)


def _repeat_metrics(oof: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model, repeat), group in oof.groupby(["model", "repeat"], sort=True):
        rows.append(
            {
                "model": model,
                "repeat": int(repeat),
                "n_oof": len(group),
                **regression_metrics(group["y_true"], group["y_pred"]),
            }
        )
    return pd.DataFrame(rows)


def _model_summary(repeat_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, group in repeat_metrics.groupby("model", sort=True):
        rows.append(
            {
                "model": model,
                "n_repeats": len(group),
                "rmse_mean": float(group["rmse"].mean()),
                "rmse_sd": float(group["rmse"].std(ddof=0)),
                "mae_mean": float(group["mae"].mean()),
                "mae_sd": float(group["mae"].std(ddof=0)),
                "r2_mean": float(group["r2"].mean()),
                "r2_sd": float(group["r2"].std(ddof=0)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["rmse_mean", "model"], kind="mergesort"
    ).reset_index(drop=True)


def _mean_oof_by_identity(oof: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["model", IDENTITY_COLUMN, SCAFFOLD_COLUMN]
    consistency = oof.groupby(group_columns, sort=True)["y_true"].nunique()
    if consistency.gt(1).any():
        raise ValueError("Repeated OOF rows disagree on y_true")
    return (
        oof.groupby(group_columns, sort=True, as_index=False)
        .agg(y_true=("y_true", "first"), y_pred=("y_pred", "mean"))
        .sort_values(["model", IDENTITY_COLUMN], kind="mergesort")
        .reset_index(drop=True)
    )


def paired_scaffold_bootstrap_deltas(
    mean_oof: pd.DataFrame,
    *,
    selected_model: str,
    n_bootstrap: int = 10000,
    seed: int = 3201,
) -> pd.DataFrame:
    """Bootstrap candidate-minus-selected metric differences by scaffold."""

    selected = mean_oof.loc[mean_oof["model"].eq(selected_model)].set_index(
        IDENTITY_COLUMN
    )
    if selected.empty:
        raise ValueError("Selected model is absent from mean OOF predictions")
    rng = np.random.default_rng(seed)
    weights = np.stack(
        [
            sample_cluster_weights(selected[SCAFFOLD_COLUMN], rng=rng)
            for _ in range(n_bootstrap)
        ]
    )
    observed = selected["y_true"].to_numpy(dtype=float)
    selected_prediction = selected["y_pred"].to_numpy(dtype=float)

    def weighted_rmse(prediction: np.ndarray) -> np.ndarray:
        squared = (prediction - observed) ** 2
        return np.sqrt((weights @ squared) / weights.sum(axis=1))

    def weighted_mae(prediction: np.ndarray) -> np.ndarray:
        absolute = np.abs(prediction - observed)
        return (weights @ absolute) / weights.sum(axis=1)

    selected_rmse = weighted_rmse(selected_prediction)
    selected_mae = weighted_mae(selected_prediction)
    rows: list[dict[str, Any]] = []
    for model, group in mean_oof.groupby("model", sort=True):
        candidate = group.set_index(IDENTITY_COLUMN).loc[selected.index]
        prediction = candidate["y_pred"].to_numpy(dtype=float)
        for metric, deltas in (
            ("RMSE", weighted_rmse(prediction) - selected_rmse),
            ("MAE", weighted_mae(prediction) - selected_mae),
        ):
            lower, upper = np.quantile(deltas, [0.025, 0.975])
            point_candidate = regression_metrics(observed, prediction)[metric.casefold()]
            point_selected = regression_metrics(observed, selected_prediction)[
                metric.casefold()
            ]
            rows.append(
                {
                    "candidate_model": model,
                    "reference_model": selected_model,
                    "metric": metric,
                    "candidate_minus_reference": point_candidate - point_selected,
                    "ci_lower": float(lower),
                    "ci_upper": float(upper),
                    "n_bootstrap": n_bootstrap,
                    "n_scaffolds": int(selected[SCAFFOLD_COLUMN].nunique()),
                    "seed": seed,
                }
            )
    return pd.DataFrame(rows)


def select_development_model(
    classical_oof: pd.DataFrame,
    gnn_oof: pd.DataFrame,
    *,
    expected_repeats: Sequence[int] = (0, 1, 2),
    n_bootstrap: int = 10000,
) -> dict[str, Any]:
    combined = _validate_oof_table(
        pd.concat([classical_oof, gnn_oof], ignore_index=True, sort=False),
        expected_repeats=expected_repeats,
    )
    repeat_metrics = _repeat_metrics(combined)
    model_summary = _model_summary(repeat_metrics)
    selected = str(model_summary.iloc[0]["model"])
    selection = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "selection_data": "development repeated scaffold-disjoint OOF predictions only",
        "selection_metric": "mean per-repeat pooled RMSE",
        "direction": "minimize",
        "tie_breaker": "lexicographic model name",
        "selected_model": selected,
        "selected_rmse_mean": float(model_summary.iloc[0]["rmse_mean"]),
        "selected_rmse_sd": float(model_summary.iloc[0]["rmse_sd"]),
        "temporal_test_consulted": False,
    }
    mean_oof = _mean_oof_by_identity(combined)
    comparisons = paired_scaffold_bootstrap_deltas(
        mean_oof,
        selected_model=selected,
        n_bootstrap=n_bootstrap,
    )
    return {
        "oof_predictions": combined,
        "repeat_metrics": repeat_metrics,
        "model_summary": model_summary,
        "mean_oof_predictions": mean_oof,
        "paired_bootstrap": comparisons,
        "selection": selection,
    }


def write_development_selection(
    result: Mapping[str, Any],
    output_dir: Path,
    *,
    input_hashes: Mapping[str, str],
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = {
        "all_oof_predictions.csv": result["oof_predictions"],
        "repeat_metrics.csv": result["repeat_metrics"],
        "model_summary.csv": result["model_summary"],
        "mean_oof_predictions.csv": result["mean_oof_predictions"],
        "paired_scaffold_bootstrap.csv": result["paired_bootstrap"],
    }
    for filename, frame in frames.items():
        atomic_write_text(
            output_dir / filename,
            frame.to_csv(index=False, lineterminator="\n", float_format="%.12g"),
        )
    manifest_core = {
        **dict(result["selection"]),
        "input_hashes": dict(sorted(input_hashes.items())),
        "output_hashes": {
            filename: sha256_file(output_dir / filename) for filename in sorted(frames)
        },
    }
    manifest = dict(manifest_core)
    manifest["selection_sha256"] = sha256_json(manifest_core)
    atomic_write_json(output_dir / "model_selection.json", manifest)
    atomic_write_text(output_dir / "RUN_COMPLETE", manifest["selection_sha256"] + "\n")
    return manifest
