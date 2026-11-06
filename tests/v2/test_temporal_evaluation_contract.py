from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pim1_qsar.benchmark.evaluation import (  # noqa: E402
    conformal_absolute_residual_quantile,
    tanimoto_applicability_domain,
)


def test_conformal_quantile_uses_finite_sample_correction() -> None:
    observed = np.arange(9, dtype=float)
    predicted = observed + np.arange(9, dtype=float) / 10
    quantile, probability = conformal_absolute_residual_quantile(
        observed, predicted, coverage=0.8
    )
    assert probability == pytest.approx(8 / 9)
    assert quantile == pytest.approx(0.8)


def test_applicability_domain_excludes_self_neighbors() -> None:
    train = np.asarray(
        [
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
        ],
        dtype=np.uint8,
    )
    test = np.asarray([[1, 1, 0, 0]], dtype=np.uint8)
    result = tanimoto_applicability_domain(train, test, ["a", "b", "c", "d"], k=2)
    assert result["nearest_development_identity"].tolist() == ["b"]
    assert result["temporal_max_similarity"].tolist() == [1.0]
    assert np.isfinite(result["development_mean_similarity"]).all()


def test_conformal_rejects_invalid_coverage() -> None:
    with pytest.raises(ValueError, match="coverage"):
        conformal_absolute_residual_quantile([1.0], [1.0], coverage=1.0)
