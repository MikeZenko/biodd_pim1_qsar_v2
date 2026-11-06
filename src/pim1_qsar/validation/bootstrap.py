"""Scaffold-cluster bootstrap intervals for sealed test predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


METRICS = ("R2", "RMSE", "MAE")


def _as_aligned_arrays(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    clusters: Sequence[Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    cluster_values = np.asarray([str(value) for value in clusters], dtype=object)
    if observed.ndim != 1 or predicted.ndim != 1 or cluster_values.ndim != 1:
        raise ValueError("y_true, y_pred, and clusters must be one-dimensional")
    if not (len(observed) == len(predicted) == len(cluster_values)):
        raise ValueError("y_true, y_pred, and clusters must have equal length")
    if len(observed) == 0:
        raise ValueError("bootstrap inputs cannot be empty")
    if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise ValueError("y_true and y_pred must contain only finite values")
    if any(value == "" for value in cluster_values):
        raise ValueError("clusters cannot contain empty labels")
    return observed, predicted, cluster_values


def _weighted_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    total_weight = float(np.sum(weights))
    if total_weight <= 0:
        raise ValueError("weights must have positive total mass")
    residual = predicted - observed
    mae = float(np.sum(weights * np.abs(residual)) / total_weight)
    mse = float(np.sum(weights * residual**2) / total_weight)
    mean_observed = float(np.sum(weights * observed) / total_weight)
    denominator = float(np.sum(weights * (observed - mean_observed) ** 2))
    r2 = float(1.0 - np.sum(weights * residual**2) / denominator) if denominator > 0 else np.nan
    return {"R2": r2, "RMSE": float(np.sqrt(mse)), "MAE": mae}


def sample_cluster_weights(
    clusters: Sequence[Any],
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw clusters with replacement and return row-level multiplicities.

    All rows in a scaffold cluster receive the same multiplicity, ensuring a
    scaffold is never partially resampled.
    """

    cluster_values = np.asarray([str(value) for value in clusters], dtype=object)
    if cluster_values.ndim != 1 or len(cluster_values) == 0:
        raise ValueError("clusters must be a non-empty one-dimensional sequence")
    unique_clusters, inverse = np.unique(cluster_values, return_inverse=True)
    draws = rng.integers(0, len(unique_clusters), size=len(unique_clusters))
    multiplicities = np.bincount(draws, minlength=len(unique_clusters))
    return multiplicities[inverse].astype(float)


@dataclass(frozen=True)
class ClusterBootstrapResult:
    """Serializable summary of scaffold-cluster bootstrap uncertainty."""

    point_estimates: dict[str, float]
    confidence_intervals: dict[str, tuple[float, float]]
    bootstrap_standard_errors: dict[str, float]
    valid_replicates: dict[str, int]
    invalid_replicates: dict[str, int]
    n_clusters: int
    n_bootstrap: int
    confidence_level: float
    seed: int
    replicates: dict[str, tuple[float, ...]] | None = None

    def as_dict(self, *, include_replicates: bool = False) -> dict[str, Any]:
        def json_float(value: float) -> float | None:
            numeric = float(value)
            return numeric if np.isfinite(numeric) else None

        payload: dict[str, Any] = {
            "point_estimates": {
                metric: json_float(value) for metric, value in self.point_estimates.items()
            },
            "confidence_intervals": {
                metric: [json_float(interval[0]), json_float(interval[1])]
                for metric, interval in self.confidence_intervals.items()
            },
            "bootstrap_standard_errors": {
                metric: json_float(value)
                for metric, value in self.bootstrap_standard_errors.items()
            },
            "valid_replicates": self.valid_replicates,
            "invalid_replicates": self.invalid_replicates,
            "n_clusters": self.n_clusters,
            "n_bootstrap": self.n_bootstrap,
            "confidence_level": self.confidence_level,
            "seed": self.seed,
        }
        if include_replicates and self.replicates is not None:
            payload["replicates"] = {
                metric: [json_float(value) for value in values]
                for metric, values in self.replicates.items()
            }
        return payload


def cluster_bootstrap_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    clusters: Sequence[Any],
    *,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    random_state: int = 3101,
    keep_replicates: bool = False,
) -> ClusterBootstrapResult:
    """Estimate metric uncertainty by resampling complete scaffold clusters."""

    observed, predicted, cluster_values = _as_aligned_arrays(y_true, y_pred, clusters)
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, int) or n_bootstrap < 1:
        raise ValueError("n_bootstrap must be a positive integer")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between 0 and 1")
    if isinstance(random_state, bool) or not isinstance(random_state, int):
        raise TypeError("random_state must be an integer")
    n_clusters = len(np.unique(cluster_values))
    if n_clusters < 2:
        raise ValueError("At least two scaffold clusters are required")

    unit_weights = np.ones(len(observed), dtype=float)
    point_estimates = _weighted_metrics(observed, predicted, unit_weights)
    rng = np.random.default_rng(random_state)
    values = {metric: np.empty(n_bootstrap, dtype=float) for metric in METRICS}
    for bootstrap_index in range(n_bootstrap):
        weights = sample_cluster_weights(cluster_values, rng=rng)
        metrics = _weighted_metrics(observed, predicted, weights)
        for metric in METRICS:
            values[metric][bootstrap_index] = metrics[metric]

    alpha = (1.0 - confidence_level) / 2.0
    confidence_intervals: dict[str, tuple[float, float]] = {}
    standard_errors: dict[str, float] = {}
    valid_replicates: dict[str, int] = {}
    invalid_replicates: dict[str, int] = {}
    retained: dict[str, tuple[float, ...]] | None = {} if keep_replicates else None
    for metric, metric_values in values.items():
        valid = metric_values[np.isfinite(metric_values)]
        valid_replicates[metric] = int(len(valid))
        invalid_replicates[metric] = int(n_bootstrap - len(valid))
        if len(valid):
            lower, upper = np.quantile(valid, [alpha, 1.0 - alpha])
            confidence_intervals[metric] = (float(lower), float(upper))
            standard_errors[metric] = float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0
        else:
            confidence_intervals[metric] = (np.nan, np.nan)
            standard_errors[metric] = np.nan
        if retained is not None:
            retained[metric] = tuple(float(value) for value in metric_values)

    return ClusterBootstrapResult(
        point_estimates=point_estimates,
        confidence_intervals=confidence_intervals,
        bootstrap_standard_errors=standard_errors,
        valid_replicates=valid_replicates,
        invalid_replicates=invalid_replicates,
        n_clusters=n_clusters,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        seed=random_state,
        replicates=retained,
    )
