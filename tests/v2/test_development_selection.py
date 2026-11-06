from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pim1_qsar.benchmark.selection import select_development_model  # noqa: E402


def _oof(model: str, offset: float) -> pd.DataFrame:
    rows = []
    for repeat in range(3):
        for index in range(6):
            rows.append(
                {
                    "model": model,
                    "repeat": repeat,
                    "fold": index % 3,
                    "standardized_parent_inchi_key": f"id-{index}",
                    "scaffold_smiles": f"scaffold-{index // 2}",
                    "y_true": float(index),
                    "y_pred": float(index) + offset,
                }
            )
    return pd.DataFrame(rows)


def test_selection_uses_development_rmse_and_cluster_bootstrap() -> None:
    result = select_development_model(
        _oof("classical", 0.5),
        _oof("gnn", 0.1),
        n_bootstrap=100,
    )
    assert result["selection"]["selected_model"] == "gnn"
    assert result["selection"]["temporal_test_consulted"] is False
    assert set(result["paired_bootstrap"]["candidate_model"]) == {"classical", "gnn"}


def test_missing_repeat_is_rejected() -> None:
    broken = _oof("gnn", 0.1).loc[lambda frame: frame["repeat"].ne(2)]
    with pytest.raises(ValueError, match="repeat inventory"):
        select_development_model(_oof("classical", 0.5), broken, n_bootstrap=10)


def test_identity_inventory_must_match_every_model() -> None:
    broken = _oof("gnn", 0.1).loc[
        lambda frame: ~(
            frame["repeat"].eq(1)
            & frame["standardized_parent_inchi_key"].eq("id-0")
        )
    ]
    with pytest.raises(ValueError, match="identity inventory"):
        select_development_model(_oof("classical", 0.5), broken, n_bootstrap=10)
