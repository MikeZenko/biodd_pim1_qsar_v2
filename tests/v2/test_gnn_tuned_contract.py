from __future__ import annotations

import inspect
from pathlib import Path
import sys
import types

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pim1_qsar.benchmark.gnn_tuned import (  # noqa: E402
    FAMILIES,
    _chemprop_datasets_tuned,
    _chemprop_inner_score,
    _chemprop_refit_predict,
    _safe_training_batch_size,
    generate_candidates,
    load_search_protocol,
    pooled_rmse,
)


def test_candidate_design_is_deterministic_unique_and_complete() -> None:
    _, protocol = load_search_protocol(PROJECT_ROOT / "configs" / "gnn_search_v1.yaml")
    for family in FAMILIES:
        first = generate_candidates(family, protocol)
        second = generate_candidates(family, protocol)
        assert first == second
        assert len(first) == protocol.candidate_budget == 24
        assert len({candidate.candidate_id for candidate in first}) == len(first)
        assert {candidate.representation for candidate in first} == {
            "graph",
            "graph_plus_rdkit2d",
        }
        assert all(candidate.family == family for candidate in first)


def test_search_spans_requested_graph_choices() -> None:
    _, protocol = load_search_protocol(PROJECT_ROOT / "configs" / "gnn_search_v1.yaml")
    gine = generate_candidates("edge_aware_gine", protocol)
    assert len({candidate.depth for candidate in gine}) >= 3
    assert len({candidate.hidden_dim for candidate in gine}) >= 3
    assert {candidate.pooling for candidate in gine} == {"mean", "add"}
    assert {candidate.normalization for candidate in gine} == {"none", "batch", "layer"}
    assert {candidate.initialization for candidate in gine} == {
        "xavier_uniform",
        "kaiming_uniform",
    }
    dmpnn = generate_candidates("chemprop_dmpnn", protocol)
    assert {candidate.pooling for candidate in dmpnn} == {"mean", "sum", "norm"}
    assert {candidate.featurization for candidate in dmpnn} == {
        "chemprop_v1",
        "chemprop_v2",
        "chemprop_organic",
    }
    pretrained = generate_candidates("chemprop_chemeleon", protocol)
    assert {candidate.encoder_training for candidate in pretrained} == {"frozen"}
    assert all(candidate.initialization == "pretrained_chemeleon" for candidate in pretrained)


def test_pooled_rmse_uses_all_inner_observations() -> None:
    observed = pooled_rmse(
        [
            {"n_validation": 1, "rmse": 1.0},
            {"n_validation": 3, "rmse": 2.0},
        ]
    )
    assert observed == pytest.approx((13.0 / 4.0) ** 0.5)
    with pytest.raises(ValueError, match="empty"):
        pooled_rmse([])


def test_protocol_is_explicitly_post_hoc_and_repeated_seeded() -> None:
    raw, protocol = load_search_protocol(PROJECT_ROOT / "configs" / "gnn_search_v1.yaml")
    assert raw["status"] == "post_hoc_development_only"
    assert "already inspected" in raw["scientific_guardrail"]
    assert protocol.inner_folds == 3
    assert protocol.rerank_top == 3
    assert len(protocol.rerank_seeds) == 3
    assert len(protocol.refit_seeds) == 3


def test_training_batch_size_never_leaves_a_singleton() -> None:
    assert _safe_training_batch_size(641, 128) == 127
    assert _safe_training_batch_size(642, 128) == 128
    assert 513 % _safe_training_batch_size(513, 64) != 1


def test_chemprop_validation_uses_training_fold_scalers(monkeypatch) -> None:
    """Chemprop validation transforms expect targets/features pre-scaled."""

    datasets = []
    target_scaler = object()
    descriptor_scaler = object()

    class FakeDataset:
        def __init__(self, datapoints, featurizer):
            self.datapoints = datapoints
            self.featurizer = featurizer
            self.target_scaler_argument = None
            self.descriptor_scaler_argument = None
            datasets.append(self)

        def normalize_targets(self, scaler=None):
            self.target_scaler_argument = scaler
            return target_scaler if scaler is None else scaler

        def normalize_inputs(self, key, scaler=None):
            assert key == "X_d"
            self.descriptor_scaler_argument = scaler
            return descriptor_scaler if scaler is None else scaler

    fake_chemprop = types.ModuleType("chemprop")
    fake_chemprop.data = types.SimpleNamespace(MoleculeDataset=FakeDataset)
    monkeypatch.setitem(sys.modules, "chemprop", fake_chemprop)
    monkeypatch.setattr(
        "pim1_qsar.benchmark.gnn_tuned._chemprop_datapoints",
        lambda frame, descriptors: frame,
    )
    monkeypatch.setattr(
        "pim1_qsar.benchmark.gnn_tuned._chemprop_featurizer",
        lambda name: name,
    )
    _, protocol = load_search_protocol(PROJECT_ROOT / "configs" / "gnn_search_v1.yaml")
    candidate = next(
        item
        for item in generate_candidates("chemprop_dmpnn", protocol)
        if item.use_descriptors
    )

    observed = _chemprop_datasets_tuned(
        object(), object(), descriptors={}, candidate=candidate
    )

    train_dataset, validation_dataset = datasets
    assert observed == (
        train_dataset,
        validation_dataset,
        target_scaler,
        descriptor_scaler,
    )
    assert train_dataset.target_scaler_argument is None
    assert validation_dataset.target_scaler_argument is target_scaler
    assert train_dataset.descriptor_scaler_argument is None
    assert validation_dataset.descriptor_scaler_argument is descriptor_scaler


def test_chemprop_selection_monitors_rmse_not_the_mse_loss() -> None:
    source = inspect.getsource(_chemprop_inner_score)
    assert source.count('monitor="val/rmse"') == 2
    assert 'monitor="val_loss"' not in source
    assert "target_scaler.scale_[0]" in source


def test_chemprop_refit_resets_scaled_validation_before_prediction() -> None:
    source = inspect.getsource(_chemprop_refit_predict)
    reset_position = source.index("validation_dataset.reset()")
    loader_position = source.index("validation_loader =", reset_position)
    predict_position = source.index("trainer.predict", loader_position)
    assert reset_position < loader_position < predict_position
