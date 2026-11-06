from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pim1_qsar.benchmark.gnn import (  # noqa: E402
    GNNTrainingConfig,
    _fit_descriptor_transform,
    model_name,
)


def test_gnn_model_names_are_explicit_controlled_ablations() -> None:
    assert model_name("edge_aware_gine", "graph") == "edge_aware_gine_graph"
    assert (
        model_name("chemprop_dmpnn", "graph_plus_rdkit2d")
        == "chemprop_dmpnn_graph_plus_rdkit2d"
    )
    with pytest.raises(ValueError, match="Unsupported"):
        model_name("plain_gcn", "graph")


def test_descriptor_transform_is_fit_on_training_rows_only() -> None:
    training = np.asarray([[1.0, np.nan], [3.0, 4.0], [5.0, 8.0]])
    validation = np.asarray([[1000.0, np.nan]])
    scaled_train, scaled_validation = _fit_descriptor_transform(training, validation)

    np.testing.assert_allclose(scaled_train.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(scaled_train.std(axis=0), 1.0, atol=1e-12)
    # The extreme validation value must not update the training mean/scale.
    assert scaled_validation[0, 0] > 100
    assert np.isfinite(scaled_validation).all()


def test_smoke_configuration_is_serializable_and_bounded() -> None:
    config = GNNTrainingConfig(max_epochs=3, patience=2, hidden_dim=32, ffn_hidden_dim=32)
    assert config.as_dict()["max_epochs"] == 3
    with pytest.raises(ValueError, match="positive integer"):
        GNNTrainingConfig(torch_threads=0)
