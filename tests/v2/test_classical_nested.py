from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pim1_qsar.benchmark.classical import (  # noqa: E402
    CandidateSpec,
    TanimotoKNNRegressor,
    make_shared_inner_splits,
    run_classical_nested,
)


def _development() -> pd.DataFrame:
    rows = []
    smiles = [
        "CC",
        "CCC",
        "CCCC",
        "CCO",
        "CCN",
        "c1ccccc1",
        "c1ccncc1",
        "C1CCCCC1",
        "C1CCNCC1",
        "CC(=O)O",
        "CCOC",
        "CCS",
    ]
    for index, value in enumerate(smiles):
        rows.append(
            {
                "standardized_parent_inchi_key": f"id-{index:02d}",
                "standardized_parent_smiles": value,
                "scaffold_smiles": f"scaffold-{index:02d}",
                "pIC50": 5.0 + index / 10.0,
            }
        )
    return pd.DataFrame(rows)


def _assignments(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    identities = data["standardized_parent_inchi_key"].tolist()
    for fold in range(3):
        validation = {value for index, value in enumerate(identities) if index % 3 == fold}
        for identity in identities:
            rows.append(
                {
                    "repeat": 0,
                    "fold": fold,
                    "partition": "validation" if identity in validation else "train",
                    "standardized_parent_inchi_key": identity,
                }
            )
    return pd.DataFrame(rows)


def test_tanimoto_knn_uses_binary_similarity() -> None:
    model = TanimotoKNNRegressor(n_neighbors=1).fit(
        np.asarray([[1, 0, 0], [1, 1, 0], [0, 0, 1]], dtype=np.uint8),
        [1.0, 2.0, 9.0],
    )
    prediction = model.predict(np.asarray([[1, 1, 1]], dtype=np.uint8))
    assert prediction.tolist() == [2.0]


def test_inner_splits_are_scaffold_disjoint() -> None:
    data = _development()
    splits = make_shared_inner_splits(data, seed=9, n_splits=3)
    for train, validation in splits:
        assert set(data.iloc[train]["scaffold_smiles"]).isdisjoint(
            data.iloc[validation]["scaffold_smiles"]
        )


def test_mean_baseline_uses_only_outer_training_labels() -> None:
    data = _development()
    candidates = (
        CandidateSpec(
            "development_mean",
            "none",
            lambda seed: DummyRegressor(strategy="mean"),
            {},
            tune=False,
        ),
    )
    result = run_classical_nested(
        data,
        _assignments(data),
        candidates=candidates,
        inner_splits=3,
        smoke=True,
    )
    predictions = result.predictions
    fold = int(predictions.iloc[0]["fold"])
    assignment = _assignments(data)
    train_ids = assignment.loc[
        assignment["fold"].eq(fold) & assignment["partition"].eq("train"),
        "standardized_parent_inchi_key",
    ]
    expected = data.set_index("standardized_parent_inchi_key").loc[train_ids, "pIC50"].mean()
    assert np.allclose(predictions["y_pred"], expected)
    assert result.selection["temporal_test_consulted"] is False


def test_outer_scaffold_overlap_is_fatal() -> None:
    data = _development()
    data.loc[0, "scaffold_smiles"] = data.loc[1, "scaffold_smiles"]
    with pytest.raises(ValueError, match="scaffold overlap"):
        run_classical_nested(data, _assignments(data), smoke=True)
