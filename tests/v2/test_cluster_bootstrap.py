from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pim1_qsar.validation.bootstrap import (
    cluster_bootstrap_metrics,
    sample_cluster_weights,
)


def test_cluster_sampling_never_partially_resamples_a_scaffold():
    clusters = np.array(["a", "a", "b", "b", "b", "c"])
    weights = sample_cluster_weights(clusters, rng=np.random.default_rng(17))

    for cluster in np.unique(clusters):
        cluster_weights = weights[clusters == cluster]
        assert np.all(cluster_weights == cluster_weights[0])
    assert weights.sum() > 0


def test_point_estimates_use_all_compounds_once():
    observed = np.array([1.0, 2.0, 4.0, 8.0])
    predicted = np.array([1.5, 1.5, 5.0, 7.0])
    result = cluster_bootstrap_metrics(
        observed,
        predicted,
        ["a", "a", "b", "c"],
        n_bootstrap=100,
        random_state=11,
    )

    expected_mae = np.mean(np.abs(predicted - observed))
    expected_rmse = np.sqrt(np.mean((predicted - observed) ** 2))
    expected_r2 = 1.0 - np.sum((predicted - observed) ** 2) / np.sum(
        (observed - observed.mean()) ** 2
    )
    assert result.point_estimates["MAE"] == pytest.approx(expected_mae)
    assert result.point_estimates["RMSE"] == pytest.approx(expected_rmse)
    assert result.point_estimates["R2"] == pytest.approx(expected_r2)


def test_bootstrap_is_deterministic_for_a_fixed_seed():
    arguments = (
        [4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        [4.2, 4.8, 6.1, 6.7, 8.4, 8.8],
        ["a", "a", "b", "b", "c", "d"],
    )
    first = cluster_bootstrap_metrics(
        *arguments,
        n_bootstrap=250,
        random_state=23,
        keep_replicates=True,
    )
    second = cluster_bootstrap_metrics(
        *arguments,
        n_bootstrap=250,
        random_state=23,
        keep_replicates=True,
    )

    assert first.as_dict(include_replicates=True) == second.as_dict(include_replicates=True)


def test_invalid_r2_replicates_are_counted_not_silently_coerced():
    # When both sampled clusters are the same, observed variance is zero and
    # R2 is undefined. RMSE and MAE remain valid.
    result = cluster_bootstrap_metrics(
        [1.0, 1.0, 3.0, 3.0],
        [1.1, 0.9, 3.2, 2.8],
        ["low", "low", "high", "high"],
        n_bootstrap=400,
        random_state=31,
    )

    assert 0 < result.invalid_replicates["R2"] < result.n_bootstrap
    assert result.invalid_replicates["RMSE"] == 0
    assert result.invalid_replicates["MAE"] == 0
    assert np.isfinite(result.confidence_intervals["R2"]).all()


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"n_bootstrap": 0}, "positive integer"),
        ({"confidence_level": 1.0}, "strictly between"),
        ({"random_state": True}, "must be an integer"),
    ],
)
def test_invalid_bootstrap_configuration_is_rejected(kwargs, match):
    with pytest.raises((TypeError, ValueError), match=match):
        cluster_bootstrap_metrics(
            [1.0, 2.0],
            [1.1, 1.9],
            ["a", "b"],
            **kwargs,
        )


def test_at_least_two_clusters_are_required():
    with pytest.raises(ValueError, match="At least two"):
        cluster_bootstrap_metrics([1.0, 2.0], [1.0, 2.0], ["same", "same"])
