"""Robustness diagnostics for immutable temporal predictions.

These helpers deliberately operate on stored predictions.  They must not fit a
model, change a prediction, or overwrite the confirmatory temporal evaluation.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from pim1_qsar.validation.bootstrap import cluster_bootstrap_metrics


IDENTITY_COLUMN = "standardized_parent_inchi_key"


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def _finite(values: pd.Series, *, label: str) -> np.ndarray:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains non-finite values")
    return array


def point_metrics(frame: pd.DataFrame) -> dict[str, float]:
    """Return the three primary regression metrics for a prediction frame."""

    observed = _finite(frame["y_true"], label="y_true")
    predicted = _finite(frame["y_pred"], label="y_pred")
    return {
        "RMSE": float(np.sqrt(mean_squared_error(observed, predicted))),
        "MAE": float(mean_absolute_error(observed, predicted)),
        "R2": float(r2_score(observed, predicted)),
    }


def _event_lookup(events: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        events,
        ["event_id", "document_chembl_id", "representative_assay_chembl_id"],
        label="strict events",
    )
    if events["event_id"].duplicated().any():
        raise ValueError("strict events contain duplicate event IDs")
    lookup = events.copy()
    lookup.index = lookup["event_id"].astype(str)
    return lookup


def _source_key(event_ids: Any, lookup: pd.DataFrame, column: str) -> tuple[str, int]:
    identifiers = [value for value in str(event_ids).split("|") if value]
    if not identifiers:
        raise ValueError("temporal prediction has no event IDs")
    missing = sorted(set(identifiers).difference(lookup.index))
    if missing:
        raise ValueError(f"temporal prediction references unknown event IDs: {missing[:3]}")
    values: set[str] = set()
    for event_id in identifiers:
        selected = lookup.loc[[event_id], column].astype(str).tolist()
        values.update(selected)
    if not values:
        raise ValueError(f"no {column} values found for temporal prediction")
    return "|".join(sorted(values)), len(values)


def build_audit_frame(
    predictions: pd.DataFrame,
    development: pd.DataFrame,
    strict_events: pd.DataFrame,
    broad_events: pd.DataFrame,
    *,
    cutoff_year: int,
) -> pd.DataFrame:
    """Attach source, chronology, and nearest-neighbor context to predictions."""

    _require_columns(
        predictions,
        [
            IDENTITY_COLUMN,
            "event_ids",
            "scaffold_smiles",
            "first_record_year",
            "y_true",
            "y_pred",
            "absolute_error",
            "nearest_development_identity",
            "max_tanimoto",
            "mean_5nn_tanimoto",
        ],
        label="temporal predictions",
    )
    _require_columns(development, [IDENTITY_COLUMN, "pIC50"], label="development")
    _require_columns(
        broad_events,
        [IDENTITY_COLUMN, "document_year"],
        label="broad measurement events",
    )
    if predictions.empty or predictions[IDENTITY_COLUMN].duplicated().any():
        raise ValueError("temporal predictions must contain unique non-empty identities")

    frame = predictions.copy()
    lookup = _event_lookup(strict_events)
    for output, source in (
        ("document_cluster", "document_chembl_id"),
        ("assay_cluster", "representative_assay_chembl_id"),
    ):
        values = frame["event_ids"].map(lambda value: _source_key(value, lookup, source))
        frame[output] = values.map(lambda value: value[0])
        frame[f"{output}_count"] = values.map(lambda value: value[1])

    broad_years = (
        broad_events.assign(
            document_year=pd.to_numeric(broad_events["document_year"], errors="coerce")
        )
        .groupby(IDENTITY_COLUMN, sort=True)["document_year"]
        .min()
    )
    frame["first_broad_record_year"] = frame[IDENTITY_COLUMN].map(broad_years)
    if frame["first_broad_record_year"].isna().any():
        raise ValueError("a temporal identity is absent from the broad event table")
    frame["broad_record_by_cutoff"] = frame["first_broad_record_year"].le(cutoff_year)

    development_labels = development.set_index(IDENTITY_COLUMN)["pIC50"]
    frame["nearest_development_pIC50"] = frame["nearest_development_identity"].map(
        development_labels
    )
    if frame["nearest_development_pIC50"].isna().any():
        raise ValueError("nearest development identity is absent from development labels")
    frame["nearest_neighbor_potency_gap"] = (
        frame["y_true"] - frame["nearest_development_pIC50"]
    ).abs()
    return frame


def metric_uncertainty_by_cluster(
    frame: pd.DataFrame,
    cluster_column: str,
    *,
    n_bootstrap: int,
    random_state: int,
) -> dict[str, Any]:
    """Bootstrap primary metrics by an explicitly named dependence cluster."""

    result = cluster_bootstrap_metrics(
        frame["y_true"],
        frame["y_pred"],
        frame[cluster_column].astype(str),
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )
    payload = result.as_dict()
    payload["cluster_column"] = cluster_column
    return payload


def _cluster_multiplicities(
    clusters: pd.Series,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    values = clusters.astype(str).to_numpy()
    unique, inverse = np.unique(values, return_inverse=True)
    draws = rng.integers(0, len(unique), size=len(unique))
    return np.bincount(draws, minlength=len(unique))[inverse]


def _linear_slope(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    total = float(weights.sum())
    if total <= 0:
        return float("nan")
    x_mean = float(np.sum(weights * x) / total)
    y_mean = float(np.sum(weights * y) / total)
    denominator = float(np.sum(weights * (x - x_mean) ** 2))
    if denominator <= 0:
        return float("nan")
    return float(np.sum(weights * (x - x_mean) * (y - y_mean)) / denominator)


def temporal_degradation_bootstrap(
    frame: pd.DataFrame,
    cluster_column: str,
    *,
    first_year: int,
    last_year: int,
    n_bootstrap: int,
    random_state: int,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Quantify endpoint RMSE change and a linear absolute-error time summary."""

    observed = _finite(frame["y_true"], label="y_true")
    predicted = _finite(frame["y_pred"], label="y_pred")
    years = _finite(frame["first_record_year"], label="first_record_year")
    absolute_error = np.abs(predicted - observed)
    first_mask = years == first_year
    last_mask = years == last_year
    if not first_mask.any() or not last_mask.any():
        raise ValueError("requested endpoint year is absent")

    def rmse(mask: np.ndarray, weights: np.ndarray) -> float:
        selected_weights = weights[mask]
        if selected_weights.sum() <= 0:
            return float("nan")
        squared = (predicted[mask] - observed[mask]) ** 2
        return float(np.sqrt(np.average(squared, weights=selected_weights)))

    unit = np.ones(len(frame), dtype=float)
    point_first = rmse(first_mask, unit)
    point_last = rmse(last_mask, unit)
    point_delta = point_last - point_first
    point_slope = _linear_slope(years, absolute_error, unit)

    rng = np.random.default_rng(random_state)
    delta = np.full(n_bootstrap, np.nan, dtype=float)
    slope = np.full(n_bootstrap, np.nan, dtype=float)
    for index in range(n_bootstrap):
        weights = _cluster_multiplicities(frame[cluster_column], rng=rng).astype(float)
        first_value = rmse(first_mask, weights)
        last_value = rmse(last_mask, weights)
        if np.isfinite(first_value) and np.isfinite(last_value):
            delta[index] = last_value - first_value
        slope[index] = _linear_slope(years, absolute_error, weights)

    alpha = (1.0 - confidence_level) / 2.0

    def summarize(values: np.ndarray) -> dict[str, Any]:
        valid = values[np.isfinite(values)]
        if not len(valid):
            raise RuntimeError("all bootstrap replicates were invalid")
        lower, upper = np.quantile(valid, [alpha, 1.0 - alpha])
        return {
            "ci_lower": float(lower),
            "ci_upper": float(upper),
            "valid_replicates": int(len(valid)),
            "invalid_replicates": int(n_bootstrap - len(valid)),
        }

    return {
        "cluster_column": cluster_column,
        "n_clusters": int(frame[cluster_column].astype(str).nunique()),
        "n_bootstrap": n_bootstrap,
        "confidence_level": confidence_level,
        "first_year": first_year,
        "last_year": last_year,
        "first_year_rmse": point_first,
        "last_year_rmse": point_last,
        "rmse_delta": {"point_estimate": point_delta, **summarize(delta)},
        "absolute_error_linear_slope_per_year": {
            "point_estimate": point_slope,
            **summarize(slope),
        },
    }


def group_centered_metrics(
    frame: pd.DataFrame,
    group_column: str,
    *,
    minimum_group_size: int = 3,
) -> dict[str, Any]:
    """Measure ordering after removing observed and predicted source intercepts."""

    if minimum_group_size < 2:
        raise ValueError("minimum_group_size must be at least 2")
    counts = frame[group_column].astype(str).value_counts()
    eligible = set(counts[counts >= minimum_group_size].index)
    selected = frame.loc[frame[group_column].astype(str).isin(eligible)].copy()
    if selected.empty:
        raise ValueError("no source groups meet the minimum size")
    groups = selected[group_column].astype(str)
    observed = _finite(selected["y_true"], label="y_true")
    predicted = _finite(selected["y_pred"], label="y_pred")
    observed_centered = observed - selected.groupby(groups)["y_true"].transform("mean")
    predicted_centered = predicted - selected.groupby(groups)["y_pred"].transform("mean")
    rho = spearmanr(observed_centered, predicted_centered).statistic
    return {
        "group_column": group_column,
        "minimum_group_size": minimum_group_size,
        "n": int(len(selected)),
        "n_groups": int(len(eligible)),
        "centered_R2": float(r2_score(observed_centered, predicted_centered)),
        "centered_spearman_rho": float(rho),
    }


def spearman_cluster_bootstrap(
    frame: pd.DataFrame,
    x_column: str,
    y_column: str,
    cluster_column: str,
    *,
    n_bootstrap: int,
    random_state: int,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Return a cluster-bootstrap interval for a Spearman association."""

    x = _finite(frame[x_column], label=x_column)
    y = _finite(frame[y_column], label=y_column)
    point = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(random_state)
    values = np.full(n_bootstrap, np.nan, dtype=float)
    indices = np.arange(len(frame))
    for bootstrap_index in range(n_bootstrap):
        multiplicities = _cluster_multiplicities(frame[cluster_column], rng=rng)
        sampled = np.repeat(indices, multiplicities)
        if len(sampled) > 1:
            values[bootstrap_index] = spearmanr(x[sampled], y[sampled]).statistic
    valid = values[np.isfinite(values)]
    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(valid, [alpha, 1.0 - alpha])
    return {
        "x": x_column,
        "y": y_column,
        "cluster_column": cluster_column,
        "n": int(len(frame)),
        "n_clusters": int(frame[cluster_column].astype(str).nunique()),
        "point_estimate": point,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "n_bootstrap": n_bootstrap,
        "valid_replicates": int(len(valid)),
        "invalid_replicates": int(n_bootstrap - len(valid)),
    }

