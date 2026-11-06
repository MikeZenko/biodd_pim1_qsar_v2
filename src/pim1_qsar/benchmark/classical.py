"""Fully nested classical baselines on immutable scaffold assignments."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from pim1_qsar.artifacts import atomic_write_json, atomic_write_text, sha256_json
from pim1_qsar.features import MORGAN_BITS, MolecularFeatures, features_from_frame
from pim1_qsar.validation.splits import make_repeated_scaffold_folds


BENCHMARK_SCHEMA_VERSION = "classical_nested_v2"
IDENTITY_COLUMN = "standardized_parent_inchi_key"
SMILES_COLUMN = "standardized_parent_smiles"
SCAFFOLD_COLUMN = "scaffold_smiles"
TARGET_COLUMN = "pIC50"


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {', '.join(missing)}")


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
    observed = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if observed.shape != predicted.shape or observed.ndim != 1 or not len(observed):
        raise ValueError("Regression metric inputs must be aligned non-empty vectors")
    if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise ValueError("Regression metric inputs must be finite")
    return {
        "rmse": float(math.sqrt(mean_squared_error(observed, predicted))),
        "mae": float(mean_absolute_error(observed, predicted)),
        "r2": float(r2_score(observed, predicted)) if len(observed) >= 2 else float("nan"),
    }


class TanimotoKNNRegressor(RegressorMixin, BaseEstimator):
    """Unweighted k-nearest-neighbor baseline using binary Tanimoto similarity."""

    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors

    def fit(self, X: np.ndarray, y: Sequence[float]):
        if isinstance(self.n_neighbors, bool) or not isinstance(self.n_neighbors, int):
            raise TypeError("n_neighbors must be an integer")
        if self.n_neighbors <= 0:
            raise ValueError("n_neighbors must be positive")
        features = np.asarray(X)
        targets = np.asarray(y, dtype=float)
        if features.ndim != 2 or targets.shape != (features.shape[0],):
            raise ValueError("X and y are not aligned")
        if features.shape[0] == 0 or not np.isin(features, [0, 1]).all():
            raise ValueError("TanimotoKNNRegressor requires non-empty binary features")
        if not np.isfinite(targets).all():
            raise ValueError("y must be finite")
        self.X_train_ = features.astype(np.float64, copy=True)
        self.y_train_ = targets.copy()
        self.train_bit_counts_ = self.X_train_.sum(axis=1)
        self.n_features_in_ = features.shape[1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "X_train_"):
            raise RuntimeError("TanimotoKNNRegressor has not been fit")
        features = np.asarray(X)
        if features.ndim != 2 or features.shape[1] != self.n_features_in_:
            raise ValueError("X has the wrong feature shape")
        if not np.isin(features, [0, 1]).all():
            raise ValueError("TanimotoKNNRegressor requires binary features")
        query = features.astype(np.float64, copy=False)
        intersections = query @ self.X_train_.T
        unions = query.sum(axis=1, keepdims=True) + self.train_bit_counts_[None, :] - intersections
        similarities = np.divide(
            intersections,
            unions,
            out=np.zeros_like(intersections, dtype=float),
            where=unions > 0,
        )
        k = min(self.n_neighbors, len(self.y_train_))
        output = np.empty(len(query), dtype=float)
        for index, row in enumerate(similarities):
            neighbors = np.argsort(-row, kind="mergesort")[:k]
            output[index] = float(np.mean(self.y_train_[neighbors]))
        return output


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    representation: str
    estimator_factory: Callable[[int], Any]
    param_grid: Mapping[str, Sequence[Any]]
    tune: bool = True

    def manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "representation": self.representation,
            "param_grid": _json_safe(self.param_grid),
            "tune": self.tune,
        }


def _combined_preprocessor(*, scale_physchem: bool) -> ColumnTransformer:
    physicochemical_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_physchem:
        physicochemical_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        [
            ("ecfp", "passthrough", list(range(MORGAN_BITS))),
            (
                "physchem",
                Pipeline(physicochemical_steps),
                list(range(MORGAN_BITS, MORGAN_BITS + 23)),
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def default_candidate_specs(*, smoke: bool = False) -> tuple[CandidateSpec, ...]:
    """Return the prespecified, deliberately limited model comparison."""

    rf_estimators = 40 if smoke else 200
    return (
        CandidateSpec(
            "development_mean",
            "none",
            lambda seed: DummyRegressor(strategy="mean"),
            {},
            tune=False,
        ),
        CandidateSpec(
            "tanimoto_knn_ecfp6",
            "ecfp6_2048",
            lambda seed: TanimotoKNNRegressor(),
            {"n_neighbors": [5] if smoke else [3, 5, 10, 20]},
        ),
        CandidateSpec(
            "ridge_combined",
            "ecfp6_2048_plus_rdkit2d",
            lambda seed: Pipeline(
                [
                    ("preprocess", _combined_preprocessor(scale_physchem=True)),
                    ("model", Ridge()),
                ]
            ),
            {"model__alpha": [10.0] if smoke else [0.1, 1.0, 10.0, 100.0]},
        ),
        CandidateSpec(
            "random_forest_ecfp6",
            "ecfp6_2048",
            lambda seed: RandomForestRegressor(
                random_state=seed,
                n_jobs=1,
            ),
            {
                "n_estimators": [rf_estimators],
                "max_features": ["sqrt"] if smoke else ["sqrt", 0.25],
                "min_samples_leaf": [1] if smoke else [1, 3],
                "max_depth": [None] if smoke else [None, 20],
            },
        ),
        CandidateSpec(
            "random_forest_combined",
            "ecfp6_2048_plus_rdkit2d",
            lambda seed: Pipeline(
                [
                    ("preprocess", _combined_preprocessor(scale_physchem=False)),
                    (
                        "model",
                        RandomForestRegressor(random_state=seed, n_jobs=1),
                    ),
                ]
            ),
            {
                "model__n_estimators": [rf_estimators],
                "model__max_features": ["sqrt"] if smoke else ["sqrt", 0.25],
                "model__min_samples_leaf": [1] if smoke else [1, 3],
                "model__max_depth": [None] if smoke else [None, 20],
            },
        ),
        CandidateSpec(
            "gradient_boosting_combined",
            "ecfp6_2048_plus_rdkit2d",
            lambda seed: Pipeline(
                [
                    ("preprocess", _combined_preprocessor(scale_physchem=False)),
                    (
                        "model",
                        GradientBoostingRegressor(
                            random_state=seed,
                            max_features="sqrt",
                            min_samples_leaf=3,
                        ),
                    ),
                ]
            ),
            {
                "model__n_estimators": [50] if smoke else [100, 250],
                "model__learning_rate": [0.05] if smoke else [0.03, 0.1],
                "model__max_depth": [2],
            },
        ),
        CandidateSpec(
            "rbf_svr_combined",
            "ecfp6_2048_plus_rdkit2d",
            lambda seed: Pipeline(
                [
                    ("preprocess", _combined_preprocessor(scale_physchem=True)),
                    ("model", SVR(kernel="rbf")),
                ]
            ),
            {
                "model__C": [10.0] if smoke else [1.0, 10.0, 100.0],
                "model__gamma": ["scale"] if smoke else ["scale", 0.001],
                "model__epsilon": [0.1] if smoke else [0.1, 0.2],
            },
        ),
    )


def _feature_matrix(features: MolecularFeatures, representation: str) -> np.ndarray:
    if representation == "none":
        return np.zeros((len(features.row_ids), 1), dtype=np.float64)
    if representation == "ecfp6_2048":
        return features.ecfp6
    if representation == "ecfp6_2048_plus_rdkit2d":
        return features.combined()
    raise ValueError(f"Unsupported representation: {representation}")


def _normalize_assignments(
    assignments: pd.DataFrame,
    *,
    identity_col: str,
) -> pd.DataFrame:
    frame = assignments.copy()
    if "partition" not in frame.columns and "role" in frame.columns:
        frame = frame.rename(columns={"role": "partition"})
    if identity_col not in frame.columns:
        aliases = [name for name in ("sample_id", "identity", "row_id") if name in frame.columns]
        if len(aliases) != 1:
            raise ValueError(f"Split assignments must contain {identity_col} or one identity alias")
        frame = frame.rename(columns={aliases[0]: identity_col})
    _require_columns(frame, ["repeat", "fold", "partition", identity_col], "assignments")
    normalized_partition = frame["partition"].astype(str).str.casefold().replace(
        {
            "outer_train": "train",
            "training": "train",
            "outer_validation": "validation",
            "valid": "validation",
            "val": "validation",
        }
    )
    if not set(normalized_partition).issubset({"train", "validation"}):
        raise ValueError("partition must contain only train/validation values")
    frame["partition"] = normalized_partition
    frame[identity_col] = frame[identity_col].astype(str)
    frame["repeat"] = pd.to_numeric(frame["repeat"], errors="raise").astype(int)
    frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(int)
    if frame.duplicated(["repeat", "fold", "partition", identity_col]).any():
        raise ValueError("Split assignments contain duplicate rows")
    return frame.sort_values(
        ["repeat", "fold", "partition", identity_col], kind="mergesort"
    ).reset_index(drop=True)


def _validate_outer_fold(
    development: pd.DataFrame,
    fold_assignments: pd.DataFrame,
    *,
    identity_col: str,
    scaffold_col: str,
) -> tuple[list[str], list[str]]:
    all_ids = set(development[identity_col].astype(str))
    train_ids = fold_assignments.loc[
        fold_assignments["partition"].eq("train"), identity_col
    ].astype(str).tolist()
    validation_ids = fold_assignments.loc[
        fold_assignments["partition"].eq("validation"), identity_col
    ].astype(str).tolist()
    if not train_ids or not validation_ids:
        raise ValueError("Every outer fold requires non-empty train and validation partitions")
    if set(train_ids) & set(validation_ids):
        raise ValueError("Outer train/validation identity overlap")
    if set(train_ids) | set(validation_ids) != all_ids:
        raise ValueError("Outer fold does not cover the complete development set")
    scaffold_map = development.set_index(identity_col)[scaffold_col].astype(str)
    if set(scaffold_map.loc[train_ids]) & set(scaffold_map.loc[validation_ids]):
        raise ValueError("Outer train/validation scaffold overlap")
    return train_ids, validation_ids


def make_shared_inner_splits(
    outer_training: pd.DataFrame,
    *,
    seed: int,
    n_splits: int = 3,
    identity_col: str = IDENTITY_COLUMN,
    scaffold_col: str = SCAFFOLD_COLUMN,
    target_col: str = TARGET_COLUMN,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Make one deterministic scaffold-disjoint inner split shared by candidates."""

    if len(outer_training) < 2 * n_splits:
        raise ValueError(
            "Outer training requires at least two observations per inner fold"
        )
    # Keep the prespecified maximum of five activity strata, but reduce the
    # count for small smoke fixtures so every stratum can populate each fold.
    n_bins = min(5, max(2, len(outer_training) // n_splits))
    folds = make_repeated_scaffold_folds(
        outer_training[identity_col].astype(str).tolist(),
        outer_training[scaffold_col].astype(str).tolist(),
        outer_training[target_col].to_numpy(dtype=float),
        n_splits=n_splits,
        n_repeats=1,
        seeds=[seed],
        n_bins=n_bins,
    )
    return [
        (
            np.asarray(assignment.train_indices, dtype=int),
            np.asarray(assignment.validation_indices, dtype=int),
        )
        for assignment in sorted(folds, key=lambda item: item.fold)
    ]


@dataclass(frozen=True)
class ClassicalBenchmarkResult:
    predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    repeat_metrics: pd.DataFrame
    model_summary: pd.DataFrame
    selected_hyperparameters: list[dict[str, Any]]
    selection: dict[str, Any]
    feature_manifest: dict[str, Any]


def run_classical_nested(
    development: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    candidates: Sequence[CandidateSpec] | None = None,
    n_jobs: int = 1,
    inner_splits: int = 3,
    base_seed: int = 4101,
    smoke: bool = False,
) -> ClassicalBenchmarkResult:
    """Evaluate all candidates using stored outer folds and nested inner tuning."""

    _require_columns(
        development,
        [IDENTITY_COLUMN, SMILES_COLUMN, SCAFFOLD_COLUMN, TARGET_COLUMN],
        "development",
    )
    data = development.copy()
    data[IDENTITY_COLUMN] = data[IDENTITY_COLUMN].astype(str)
    if data[IDENTITY_COLUMN].duplicated().any():
        raise ValueError("Development identities must be unique")
    targets = pd.to_numeric(data[TARGET_COLUMN], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(targets).all():
        raise ValueError("Development targets must be finite")
    data[TARGET_COLUMN] = targets
    data = data.sort_values(IDENTITY_COLUMN, kind="mergesort").reset_index(drop=True)
    normalized_assignments = _normalize_assignments(
        assignments, identity_col=IDENTITY_COLUMN
    )
    features = features_from_frame(data)
    if tuple(data[IDENTITY_COLUMN]) != features.row_ids:
        raise RuntimeError("Feature rows are not aligned to development identities")
    feature_index = {identity: index for index, identity in enumerate(features.row_ids)}
    candidate_list = tuple(candidates or default_candidate_specs(smoke=smoke))
    if not candidate_list or len({candidate.name for candidate in candidate_list}) != len(
        candidate_list
    ):
        raise ValueError("Candidate names must be unique and non-empty")
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be zero")

    prediction_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    hyperparameter_rows: list[dict[str, Any]] = []
    fold_keys = normalized_assignments[["repeat", "fold"]].drop_duplicates().itertuples(
        index=False, name=None
    )
    keys = list(fold_keys)
    if smoke:
        keys = keys[:1]

    for repeat, fold in keys:
        fold_assignment = normalized_assignments.loc[
            normalized_assignments["repeat"].eq(repeat)
            & normalized_assignments["fold"].eq(fold)
        ]
        train_ids, validation_ids = _validate_outer_fold(
            data,
            fold_assignment,
            identity_col=IDENTITY_COLUMN,
            scaffold_col=SCAFFOLD_COLUMN,
        )
        row_lookup = data.set_index(IDENTITY_COLUMN, drop=False)
        outer_training = row_lookup.loc[train_ids].reset_index(drop=True)
        outer_validation = row_lookup.loc[validation_ids].reset_index(drop=True)
        inner_seed = base_seed + int(repeat) * 100 + int(fold)
        shared_inner = make_shared_inner_splits(
            outer_training,
            seed=inner_seed,
            n_splits=inner_splits,
        )
        train_feature_indices = np.asarray([feature_index[value] for value in train_ids])
        validation_feature_indices = np.asarray(
            [feature_index[value] for value in validation_ids]
        )
        y_train = outer_training[TARGET_COLUMN].to_numpy(dtype=float)
        y_validation = outer_validation[TARGET_COLUMN].to_numpy(dtype=float)

        for candidate in candidate_list:
            matrix = _feature_matrix(features, candidate.representation)
            X_train = matrix[train_feature_indices]
            X_validation = matrix[validation_feature_indices]
            estimator = candidate.estimator_factory(inner_seed)
            start = time.perf_counter()
            if candidate.tune:
                search = GridSearchCV(
                    estimator,
                    param_grid=dict(candidate.param_grid),
                    scoring="neg_root_mean_squared_error",
                    cv=shared_inner,
                    refit=True,
                    n_jobs=n_jobs,
                    error_score="raise",
                    return_train_score=False,
                )
                search.fit(X_train, y_train)
                fitted = search.best_estimator_
                best_params = _json_safe(search.best_params_)
                inner_rmse = float(-search.best_score_)
            else:
                fitted = estimator.fit(X_train, y_train)
                best_params = {}
                inner_rmse = float("nan")
            predicted = np.asarray(fitted.predict(X_validation), dtype=float)
            elapsed = float(time.perf_counter() - start)
            scores = regression_metrics(y_validation, predicted)
            metric_rows.append(
                {
                    "model": candidate.name,
                    "representation": candidate.representation,
                    "repeat": int(repeat),
                    "fold": int(fold),
                    "n_train": len(train_ids),
                    "n_validation": len(validation_ids),
                    "inner_best_rmse": inner_rmse,
                    **scores,
                    "runtime_seconds": elapsed,
                }
            )
            hyperparameter_rows.append(
                {
                    "model": candidate.name,
                    "representation": candidate.representation,
                    "repeat": int(repeat),
                    "fold": int(fold),
                    "inner_seed": inner_seed,
                    "best_params": best_params,
                }
            )
            for row, prediction in zip(
                outer_validation.to_dict(orient="records"), predicted
            ):
                prediction_rows.append(
                    {
                        "model": candidate.name,
                        "representation": candidate.representation,
                        "repeat": int(repeat),
                        "fold": int(fold),
                        IDENTITY_COLUMN: str(row[IDENTITY_COLUMN]),
                        SCAFFOLD_COLUMN: str(row[SCAFFOLD_COLUMN]),
                        "y_true": float(row[TARGET_COLUMN]),
                        "y_pred": float(prediction),
                    }
                )

    predictions = pd.DataFrame(prediction_rows)
    fold_metrics = pd.DataFrame(metric_rows)
    repeat_rows: list[dict[str, Any]] = []
    for (model, representation, repeat), group in predictions.groupby(
        ["model", "representation", "repeat"], sort=True
    ):
        if not smoke and group[IDENTITY_COLUMN].duplicated().any():
            raise RuntimeError("An identity received multiple OOF predictions in one repeat")
        scores = regression_metrics(group["y_true"], group["y_pred"])
        repeat_rows.append(
            {
                "model": model,
                "representation": representation,
                "repeat": int(repeat),
                "n_oof": len(group),
                **scores,
            }
        )
    repeat_metrics = pd.DataFrame(repeat_rows)
    summary_rows: list[dict[str, Any]] = []
    for (model, representation), group in repeat_metrics.groupby(
        ["model", "representation"], sort=True
    ):
        summary_rows.append(
            {
                "model": model,
                "representation": representation,
                "n_repeats": len(group),
                "rmse_mean": float(group["rmse"].mean()),
                "rmse_sd": float(group["rmse"].std(ddof=0)),
                "mae_mean": float(group["mae"].mean()),
                "mae_sd": float(group["mae"].std(ddof=0)),
                "r2_mean": float(group["r2"].mean()),
                "r2_sd": float(group["r2"].std(ddof=0)),
            }
        )
    model_summary = pd.DataFrame(summary_rows).sort_values(
        ["rmse_mean", "model"], kind="mergesort"
    ).reset_index(drop=True)
    selected = model_summary.iloc[0]
    selection = {
        "selection_data": "development repeated outer-fold OOF predictions only",
        "selection_metric": "mean per-repeat pooled RMSE",
        "direction": "minimize",
        "tie_breaker": "lexicographic model name",
        "selected_model": str(selected["model"]),
        "selected_representation": str(selected["representation"]),
        "selected_rmse_mean": float(selected["rmse_mean"]),
        "temporal_test_consulted": False,
    }
    predictions = predictions.sort_values(
        ["model", "repeat", "fold", IDENTITY_COLUMN], kind="mergesort"
    ).reset_index(drop=True)
    fold_metrics = fold_metrics.sort_values(
        ["model", "repeat", "fold"], kind="mergesort"
    ).reset_index(drop=True)
    repeat_metrics = repeat_metrics.sort_values(
        ["model", "repeat"], kind="mergesort"
    ).reset_index(drop=True)
    return ClassicalBenchmarkResult(
        predictions=predictions,
        fold_metrics=fold_metrics,
        repeat_metrics=repeat_metrics,
        model_summary=model_summary,
        selected_hyperparameters=hyperparameter_rows,
        selection=selection,
        feature_manifest=features.manifest,
    )


def write_classical_benchmark(
    result: ClassicalBenchmarkResult,
    output_dir: Path,
    *,
    input_manifest: Mapping[str, Any],
    candidate_manifest: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = {
        "oof_predictions.csv": result.predictions,
        "fold_metrics.csv": result.fold_metrics,
        "repeat_metrics.csv": result.repeat_metrics,
        "model_summary.csv": result.model_summary,
    }
    for filename, frame in frames.items():
        atomic_write_text(
            output_dir / filename,
            frame.to_csv(index=False, lineterminator="\n", float_format="%.12g"),
        )
    atomic_write_json(output_dir / "selected_hyperparameters.json", result.selected_hyperparameters)
    atomic_write_json(output_dir / "model_selection.json", result.selection)
    atomic_write_json(output_dir / "feature_manifest.json", result.feature_manifest)
    manifest_core = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "input_lineage": _json_safe(input_manifest),
        "candidates": _json_safe(candidate_manifest),
        "feature_manifest_sha256": sha256_json(result.feature_manifest),
        "selection": result.selection,
    }
    manifest = dict(manifest_core)
    manifest["run_sha256"] = sha256_json(manifest_core)
    atomic_write_json(output_dir / "run_manifest.json", manifest)
    atomic_write_text(output_dir / "RUN_COMPLETE", manifest["run_sha256"] + "\n")
    return manifest
