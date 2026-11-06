from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pim1_qsar.features import (  # noqa: E402
    MORGAN_BITS,
    PHYSICOCHEMICAL_DESCRIPTOR_NAMES,
    compute_molecular_features,
)


def test_feature_contract_is_aligned_and_deterministic() -> None:
    ids = ["a", "b"]
    smiles = ["CCO", "c1ccccc1"]
    first = compute_molecular_features(ids, smiles)
    second = compute_molecular_features(ids, smiles)

    assert first.ecfp6.shape == (2, MORGAN_BITS)
    assert first.physicochemical.shape == (
        2,
        len(PHYSICOCHEMICAL_DESCRIPTOR_NAMES),
    )
    np.testing.assert_array_equal(first.ecfp6, second.ecfp6)
    np.testing.assert_allclose(first.physicochemical, second.physicochemical)
    assert first.manifest == second.manifest
    assert first.canonical_smiles == ("CCO", "c1ccccc1")


def test_feature_generation_never_silently_drops_invalid_rows() -> None:
    with pytest.raises(ValueError, match="split alignment forbids row dropping"):
        compute_molecular_features(["ok", "bad"], ["CC", "not-a-smiles"])


def test_duplicate_row_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        compute_molecular_features(["same", "same"], ["CC", "CCC"])
