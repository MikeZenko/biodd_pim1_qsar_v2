from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "temporal_evaluation_v2"
    / "temporal_predictions.csv"
)


@pytest.fixture(scope="module")
def predictions() -> pd.DataFrame:
    frame = pd.read_csv(PREDICTIONS_PATH)
    assert len(frame) == 284
    assert frame["first_record_year"].between(2019, 2025).all()
    return frame


def _rmse(observed: pd.Series, predicted: pd.Series) -> float:
    residual = predicted.to_numpy(dtype=float) - observed.to_numpy(dtype=float)
    return float(np.sqrt(np.mean(np.square(residual))))


def _r2(observed: pd.Series, predicted: pd.Series) -> float:
    observed_array = observed.to_numpy(dtype=float)
    predicted_array = predicted.to_numpy(dtype=float)
    residual_sum_squares = np.square(observed_array - predicted_array).sum()
    total_sum_squares = np.square(observed_array - observed_array.mean()).sum()
    return float(1.0 - residual_sum_squares / total_sum_squares)


def _center_within_year(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column] - frame.groupby("first_record_year")[column].transform("mean")


def test_year_cohorts_explain_part_of_pooled_target_variance(
    predictions: pd.DataFrame,
) -> None:
    observed = predictions["y_true"]
    grand_mean = observed.mean()
    total_sum_squares = np.square(observed - grand_mean).sum()
    between_year_sum_squares = sum(
        len(cohort) * np.square(cohort["y_true"].mean() - grand_mean)
        for _, cohort in predictions.groupby("first_record_year")
    )

    between_year_variance_share = between_year_sum_squares / total_sum_squares
    assert between_year_variance_share == pytest.approx(
        0.31836132603735, abs=1e-12
    )


def test_within_year_centering_reveals_weaker_recent_cohort_performance(
    predictions: pd.DataFrame,
) -> None:
    centered_observed = _center_within_year(predictions, "y_true")
    centered_predicted = _center_within_year(predictions, "y_pred")

    assert _r2(centered_observed, centered_predicted) == pytest.approx(
        0.2659537914647251, abs=1e-12
    )

    post_2019 = predictions["first_record_year"].ge(2020)
    assert post_2019.sum() == 197
    assert _r2(
        centered_observed.loc[post_2019], centered_predicted.loc[post_2019]
    ) == pytest.approx(-0.04598709285038205, abs=1e-12)
    assert spearmanr(
        centered_observed.loc[post_2019], centered_predicted.loc[post_2019]
    ).statistic == pytest.approx(0.2855823239919925, abs=1e-12)


def test_annual_metrics_capture_temporal_degradation(
    predictions: pd.DataFrame,
) -> None:
    cohorts = {
        int(year): cohort
        for year, cohort in predictions.groupby("first_record_year", sort=True)
    }

    assert _rmse(cohorts[2019]["y_true"], cohorts[2019]["y_pred"]) == pytest.approx(
        0.7950723209001903, abs=1e-12
    )
    assert _rmse(cohorts[2024]["y_true"], cohorts[2024]["y_pred"]) == pytest.approx(
        1.2614732970004596, abs=1e-12
    )

    annual_spearman = {
        year: float(spearmanr(cohort["y_true"], cohort["y_pred"]).statistic)
        for year, cohort in cohorts.items()
    }
    assert annual_spearman == pytest.approx(
        {
            2019: 0.8421455565173297,
            2020: 0.33811299504472675,
            2021: 0.5265571403198899,
            2022: -0.29846153846153844,
            2023: 0.03214285714285714,
            2024: 0.3490885856229932,
            2025: -0.19999999999999998,
        },
        abs=1e-12,
    )


def test_similarity_domain_does_not_discriminate_absolute_error(
    predictions: pd.DataFrame,
) -> None:
    similarity_error_rho = spearmanr(
        predictions["mean_5nn_tanimoto"], predictions["absolute_error"]
    ).statistic
    assert similarity_error_rho == pytest.approx(0.04188514726611557, abs=1e-12)

    inside = predictions.loc[predictions["inside_applicability_domain"]]
    outside = predictions.loc[~predictions["inside_applicability_domain"]]
    assert (len(inside), len(outside)) == (130, 154)
    assert _rmse(inside["y_true"], inside["y_pred"]) == pytest.approx(
        0.953973203131952, abs=1e-12
    )
    assert _rmse(outside["y_true"], outside["y_pred"]) == pytest.approx(
        0.9877608004021766, abs=1e-12
    )


def test_residual_band_is_wide_and_under_covers(
    predictions: pd.DataFrame,
) -> None:
    widths = predictions["conformal_upper"] - predictions["conformal_lower"]
    # Bounds were serialized independently to 12 decimal places, so subtracting
    # them can differ by one unit at the final stored decimal.
    assert np.ptp(widths.to_numpy(dtype=float)) < 1e-10
    width = float(widths.iloc[0])
    assert width == pytest.approx(2.50408134926, abs=1e-11)

    independently_covered = (
        predictions["y_true"].ge(predictions["conformal_lower"])
        & predictions["y_true"].le(predictions["conformal_upper"])
    )
    assert int(independently_covered.sum()) == 231
    assert independently_covered.mean() == pytest.approx(
        0.8133802816901409, abs=1e-12
    )
    assert 10**width == pytest.approx(319.21357295181326, rel=1e-12)
