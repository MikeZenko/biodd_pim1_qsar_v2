"""One-time selected-model temporal evaluation and uncertainty diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import GridSearchCV

from pim1_qsar.benchmark.classical import (
    IDENTITY_COLUMN,
    SCAFFOLD_COLUMN,
    TARGET_COLUMN,
    _feature_matrix,
    default_candidate_specs,
    make_shared_inner_splits,
    regression_metrics,
)
from pim1_qsar.benchmark.gnn import (
    GRAPH_DESCRIPTOR_VARIANT,
    _chemprop_early_stop_epoch,
    _chemprop_refit_predict,
    _gine_early_stop_epoch,
    _gine_refit_predict,
    _precompute_graphs,
    model_name,
    GNNTrainingConfig,
)
from pim1_qsar.features import features_from_frame
from pim1_qsar.validation.bootstrap import (
    cluster_bootstrap_metrics,
    sample_cluster_weights,
)


EVALUATION_SCHEMA_VERSION = "temporal_evaluation_v2"


@dataclass(frozen=True)
class FinalPrediction:
    predictions: np.ndarray
    seed_standard_deviation: np.ndarray
    training_manifest: dict[str, Any]
    model_artifact: Path | None


def conformal_absolute_residual_quantile(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    coverage: float = 0.90,
) -> tuple[float, float]:
    """Return the finite-sample order statistic used by residual intervals.

    This numerical helper does not by itself make the surrounding OOF-score and
    final-refit workflow a formally valid conformal predictor.
    """

    observed = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if observed.shape != predicted.shape or observed.ndim != 1 or not len(observed):
        raise ValueError("Conformal inputs must be aligned non-empty vectors")
    if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise ValueError("Conformal inputs must be finite")
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be strictly between zero and one")
    n = len(observed)
    probability = min(1.0, math.ceil((n + 1) * coverage) / n)
    quantile = float(
        np.quantile(np.abs(observed - predicted), probability, method="higher")
    )
    return quantile, probability


def tanimoto_applicability_domain(
    development_ecfp: np.ndarray,
    temporal_ecfp: np.ndarray,
    development_ids: Sequence[str],
    *,
    k: int = 5,
    lower_development_quantile: float = 0.05,
) -> dict[str, Any]:
    """Training-defined mean-k-neighbor Tanimoto applicability domain."""

    train = np.asarray(development_ecfp, dtype=np.float32)
    test = np.asarray(temporal_ecfp, dtype=np.float32)
    if train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1]:
        raise ValueError("ECFP matrices must be two-dimensional and aligned")
    if len(train) <= k or len(development_ids) != len(train):
        raise ValueError("Applicability-domain training data are too small or misaligned")
    if not np.isin(train, [0, 1]).all() or not np.isin(test, [0, 1]).all():
        raise ValueError("Applicability-domain ECFP matrices must be binary")

    train_counts = train.sum(axis=1)
    train_intersections = train @ train.T
    train_unions = (
        train_counts[:, None] + train_counts[None, :] - train_intersections
    )
    train_similarity = np.divide(
        train_intersections,
        train_unions,
        out=np.zeros_like(train_intersections),
        where=train_unions > 0,
    )
    np.fill_diagonal(train_similarity, -np.inf)
    train_top_k = np.partition(train_similarity, -k, axis=1)[:, -k:]
    development_mean_similarity = train_top_k.mean(axis=1)
    threshold = float(
        np.quantile(
            development_mean_similarity,
            lower_development_quantile,
            method="linear",
        )
    )

    test_counts = test.sum(axis=1)
    test_intersections = test @ train.T
    test_unions = test_counts[:, None] + train_counts[None, :] - test_intersections
    test_similarity = np.divide(
        test_intersections,
        test_unions,
        out=np.zeros_like(test_intersections),
        where=test_unions > 0,
    )
    test_top_k = np.partition(test_similarity, -k, axis=1)[:, -k:]
    temporal_mean_similarity = test_top_k.mean(axis=1)
    nearest_indices = np.argmax(test_similarity, axis=1)
    return {
        "k": k,
        "development_lower_quantile": lower_development_quantile,
        "mean_similarity_threshold": threshold,
        "development_mean_similarity": development_mean_similarity,
        "temporal_mean_similarity": temporal_mean_similarity,
        "temporal_max_similarity": test_similarity[
            np.arange(len(test_similarity)), nearest_indices
        ],
        "nearest_development_identity": np.asarray(development_ids, dtype=object)[
            nearest_indices
        ],
        "inside_domain": temporal_mean_similarity >= threshold,
    }


def _fit_predict_classical(
    selected_model: str,
    development: pd.DataFrame,
    temporal_test: pd.DataFrame,
    *,
    model_dir: Path,
    n_jobs: int,
    seed: int = 6101,
) -> FinalPrediction:
    candidates = {candidate.name: candidate for candidate in default_candidate_specs()}
    if selected_model not in candidates:
        raise ValueError(f"Unknown classical selected model: {selected_model}")
    candidate = candidates[selected_model]
    development_features = features_from_frame(development)
    temporal_features = features_from_frame(temporal_test)
    X_development = _feature_matrix(development_features, candidate.representation)
    X_temporal = _feature_matrix(temporal_features, candidate.representation)
    y_development = development[TARGET_COLUMN].to_numpy(dtype=float)
    estimator = candidate.estimator_factory(seed)
    start_splits = make_shared_inner_splits(
        development,
        seed=seed,
        n_splits=3,
    )
    if candidate.tune:
        search = GridSearchCV(
            estimator,
            param_grid=dict(candidate.param_grid),
            scoring="neg_root_mean_squared_error",
            cv=start_splits,
            refit=True,
            n_jobs=n_jobs,
            error_score="raise",
        )
        search.fit(X_development, y_development)
        fitted = search.best_estimator_
        best_params = search.best_params_
        inner_rmse = float(-search.best_score_)
    else:
        fitted = estimator.fit(X_development, y_development)
        best_params = {}
        inner_rmse = None
    prediction = np.asarray(fitted.predict(X_temporal), dtype=float)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "selected_classical_model.joblib"
    temporary_path = model_path.with_suffix(".joblib.tmp")
    joblib.dump(fitted, temporary_path)
    temporary_path.replace(model_path)
    return FinalPrediction(
        predictions=prediction,
        seed_standard_deviation=np.zeros(len(prediction), dtype=float),
        training_manifest={
            "model_family": "classical",
            "selected_model": selected_model,
            "representation": candidate.representation,
            "final_tuning": "3-fold scaffold-disjoint development-only GridSearchCV",
            "best_params": best_params,
            "inner_best_rmse": inner_rmse,
            "seed": seed,
            "temporal_labels_used_for_fit_or_tuning": False,
        },
        model_artifact=model_path,
    )


def _parse_gnn_name(selected_model: str) -> tuple[str, str]:
    for backend in ("edge_aware_gine", "chemprop_dmpnn"):
        for variant in ("graph", "graph_plus_rdkit2d"):
            if model_name(backend, variant) == selected_model:
                return backend, variant
    raise ValueError(f"Unknown selected GNN model: {selected_model}")


def _fit_predict_gnn(
    selected_model: str,
    development: pd.DataFrame,
    temporal_test: pd.DataFrame,
    *,
    seeds: Sequence[int] = (2101, 2102, 2103),
) -> FinalPrediction:
    backend, variant = _parse_gnn_name(selected_model)
    use_descriptors = variant == GRAPH_DESCRIPTOR_VARIANT
    combined = pd.concat([development, temporal_test], ignore_index=True)
    features = features_from_frame(combined)
    descriptor_map = {
        identity: features.physicochemical[index].copy()
        for index, identity in enumerate(features.row_ids)
    }
    graphs = _precompute_graphs(combined) if backend == "edge_aware_gine" else {}
    base_config = GNNTrainingConfig()
    all_predictions: list[np.ndarray] = []
    epoch_records: list[dict[str, Any]] = []
    for seed in seeds:
        inner_splits = make_shared_inner_splits(development, seed=seed, n_splits=3)
        best_epochs: list[int] = []
        for inner_fold, (train_indices, validation_indices) in enumerate(inner_splits):
            train = development.iloc[train_indices].reset_index(drop=True)
            validation = development.iloc[validation_indices].reset_index(drop=True)
            inner_seed = int(seed) + inner_fold
            if backend == "edge_aware_gine":
                epoch = _gine_early_stop_epoch(
                    train,
                    validation,
                    graphs=graphs,
                    descriptor_map=descriptor_map,
                    use_descriptors=use_descriptors,
                    config=base_config,
                    seed=inner_seed,
                )
            else:
                epoch = _chemprop_early_stop_epoch(
                    train,
                    validation,
                    descriptor_map=descriptor_map,
                    use_descriptors=use_descriptors,
                    config=base_config,
                    seed=inner_seed,
                )
            best_epochs.append(epoch)
        refit_epochs = max(1, int(np.median(best_epochs)))
        if backend == "edge_aware_gine":
            prediction = _gine_refit_predict(
                development,
                temporal_test,
                graphs=graphs,
                descriptor_map=descriptor_map,
                use_descriptors=use_descriptors,
                config=base_config,
                seed=int(seed),
                epochs=refit_epochs,
            )
        else:
            prediction = _chemprop_refit_predict(
                development,
                temporal_test,
                descriptor_map=descriptor_map,
                use_descriptors=use_descriptors,
                config=base_config,
                seed=int(seed),
                epochs=refit_epochs,
            )
        all_predictions.append(prediction)
        epoch_records.append(
            {
                "seed": int(seed),
                "inner_best_epochs": best_epochs,
                "refit_epochs": refit_epochs,
            }
        )
    matrix = np.stack(all_predictions)
    return FinalPrediction(
        predictions=matrix.mean(axis=0),
        seed_standard_deviation=matrix.std(axis=0, ddof=0),
        training_manifest={
            "model_family": "graph",
            "selected_model": selected_model,
            "backend": backend,
            "variant": variant,
            "seed_models": epoch_records,
            "ensemble": "arithmetic mean of three development-only refits",
            "temporal_labels_used_for_fit_or_tuning": False,
        },
        model_artifact=None,
    )


def fit_selected_and_predict(
    selected_model: str,
    development: pd.DataFrame,
    temporal_test: pd.DataFrame,
    *,
    model_dir: Path,
    n_jobs: int = -1,
) -> FinalPrediction:
    classical_names = {candidate.name for candidate in default_candidate_specs()}
    if selected_model in classical_names:
        return _fit_predict_classical(
            selected_model,
            development,
            temporal_test,
            model_dir=model_dir,
            n_jobs=n_jobs,
        )
    return _fit_predict_gnn(selected_model, development, temporal_test)


def evaluate_temporal_predictions(
    development: pd.DataFrame,
    temporal_test: pd.DataFrame,
    selected_oof_repeat: pd.DataFrame,
    final_prediction: FinalPrediction,
    *,
    conformal_coverage: float = 0.90,
    n_bootstrap: int = 10000,
    bootstrap_seed: int = 3101,
) -> dict[str, Any]:
    """Calculate prespecified metrics, intervals, calibration, and AD results."""

    y_true = temporal_test[TARGET_COLUMN].to_numpy(dtype=float)
    y_pred = np.asarray(final_prediction.predictions, dtype=float)
    if y_pred.shape != y_true.shape or not np.isfinite(y_pred).all():
        raise ValueError("Final temporal predictions are invalid or misaligned")
    quantile, probability = conformal_absolute_residual_quantile(
        selected_oof_repeat["y_true"],
        selected_oof_repeat["y_pred"],
        coverage=conformal_coverage,
    )
    lower = y_pred - quantile
    upper = y_pred + quantile
    covered = (y_true >= lower) & (y_true <= upper)

    development_features = features_from_frame(development)
    temporal_features = features_from_frame(temporal_test)
    ad = tanimoto_applicability_domain(
        development_features.ecfp6,
        temporal_features.ecfp6,
        development_features.row_ids,
    )
    provenance_columns = [
        column
        for column in (
            IDENTITY_COLUMN,
            SCAFFOLD_COLUMN,
            "standardized_parent_smiles",
            TARGET_COLUMN,
            "first_record_year",
            "last_record_year",
            "measurement_count",
            "measurement_std",
            "event_ids",
        )
        if column in temporal_test.columns
    ]
    predictions = temporal_test[provenance_columns].copy()
    predictions = predictions.rename(columns={TARGET_COLUMN: "y_true"})
    predictions["y_pred"] = y_pred
    predictions["residual"] = y_pred - y_true
    predictions["absolute_error"] = np.abs(y_pred - y_true)
    predictions["seed_prediction_sd"] = final_prediction.seed_standard_deviation
    predictions["conformal_lower"] = lower
    predictions["conformal_upper"] = upper
    predictions["conformal_covered"] = covered
    predictions["mean_5nn_tanimoto"] = ad["temporal_mean_similarity"]
    predictions["max_tanimoto"] = ad["temporal_max_similarity"]
    predictions["nearest_development_identity"] = ad[
        "nearest_development_identity"
    ]
    predictions["inside_applicability_domain"] = ad["inside_domain"]
    development_scaffolds = set(development[SCAFFOLD_COLUMN].astype(str))
    predictions["scaffold_seen_in_development"] = predictions[
        SCAFFOLD_COLUMN
    ].astype(str).isin(development_scaffolds)

    bootstrap = cluster_bootstrap_metrics(
        y_true,
        y_pred,
        temporal_test[SCAFFOLD_COLUMN],
        n_bootstrap=n_bootstrap,
        random_state=bootstrap_seed,
    )
    rng = np.random.default_rng(bootstrap_seed + 1)
    coverage_replicates = np.empty(n_bootstrap, dtype=float)
    covered_float = covered.astype(float)
    for index in range(n_bootstrap):
        weights = sample_cluster_weights(temporal_test[SCAFFOLD_COLUMN], rng=rng)
        coverage_replicates[index] = float(
            np.sum(weights * covered_float) / np.sum(weights)
        )
    coverage_ci = np.quantile(coverage_replicates, [0.025, 0.975])

    calibration = predictions[["y_true", "y_pred"]].copy()
    calibration["bin"] = pd.qcut(
        calibration["y_pred"].rank(method="first"),
        q=5,
        labels=False,
    )
    calibration_bins = (
        calibration.groupby("bin", sort=True)
        .agg(
            n=("y_true", "size"),
            mean_prediction=("y_pred", "mean"),
            mean_observed=("y_true", "mean"),
            observed_sd=("y_true", "std"),
        )
        .reset_index()
    )
    slope, intercept = np.polyfit(y_pred, y_true, 1)
    absolute_error = np.abs(y_pred - y_true)
    pearson = pearsonr(predictions["mean_5nn_tanimoto"], absolute_error)
    spearman = spearmanr(predictions["mean_5nn_tanimoto"], absolute_error)

    ad_groups: list[dict[str, Any]] = []
    for inside, group in predictions.groupby("inside_applicability_domain", sort=False):
        scores = regression_metrics(group["y_true"], group["y_pred"])
        ad_groups.append(
            {
                "group": "inside" if bool(inside) else "outside",
                "n": len(group),
                "n_scaffolds": int(group[SCAFFOLD_COLUMN].nunique()),
                **scores,
            }
        )
    scaffold_novelty_groups: list[dict[str, Any]] = []
    for seen, group in predictions.groupby("scaffold_seen_in_development", sort=False):
        scaffold_novelty_groups.append(
            {
                "group": "seen_scaffold" if bool(seen) else "new_scaffold",
                "n": len(group),
                "n_scaffolds": int(group[SCAFFOLD_COLUMN].nunique()),
                **regression_metrics(group["y_true"], group["y_pred"]),
            }
        )
    non_extreme = predictions.loc[predictions["y_true"].le(10.0)]
    non_extreme_bootstrap = cluster_bootstrap_metrics(
        non_extreme["y_true"],
        non_extreme["y_pred"],
        non_extreme[SCAFFOLD_COLUMN],
        n_bootstrap=n_bootstrap,
        random_state=bootstrap_seed + 2,
    )
    diagnostics = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "primary_metrics": bootstrap.as_dict(),
        "conformal": {
            "method": "cross-validated absolute-residual interval",
            "calibration_source": "selected-model development OOF residuals, locked repeat 0",
            "validity_scope": (
                "descriptive under temporal distribution shift; this OOF construction "
                "is not claimed as a formal split-conformal coverage guarantee"
            ),
            "nominal_coverage": conformal_coverage,
            "finite_sample_probability": probability,
            "absolute_residual_quantile": quantile,
            "empirical_temporal_coverage": float(covered.mean()),
            "coverage_cluster_bootstrap_ci": [
                float(coverage_ci[0]),
                float(coverage_ci[1]),
            ],
            "mean_interval_width": float(2 * quantile),
        },
        "calibration": {
            "observed_on_prediction_slope": float(slope),
            "observed_on_prediction_intercept": float(intercept),
        },
        "applicability_domain": {
            "definition": "mean Tanimoto similarity to five development neighbors",
            "threshold_source": "development-only 5th percentile",
            "mean_similarity_threshold": float(ad["mean_similarity_threshold"]),
            "inside_count": int(ad["inside_domain"].sum()),
            "outside_count": int((~ad["inside_domain"]).sum()),
            "absolute_error_pearson_r": float(pearson.statistic),
            "absolute_error_pearson_p_descriptive": float(pearson.pvalue),
            "absolute_error_spearman_rho": float(spearman.statistic),
            "absolute_error_spearman_p_descriptive": float(spearman.pvalue),
            "group_metrics": ad_groups,
        },
        "scaffold_novelty": {
            "definition": "Bemis-Murcko scaffold present versus absent in development",
            "group_metrics": scaffold_novelty_groups,
        },
        "prespecified_high_tail_sensitivity": {
            "filter": "temporal compounds with observed pIC50 <= 10",
            "removed_compounds": int(len(predictions) - len(non_extreme)),
            "metrics": non_extreme_bootstrap.as_dict(),
        },
    }
    return {
        "predictions": predictions,
        "calibration_bins": calibration_bins,
        "diagnostics": diagnostics,
        "training_manifest": final_prediction.training_manifest,
    }
