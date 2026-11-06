from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pim1_qsar.validation.splits import (
    make_repeated_scaffold_folds,
    repeated_folds_manifest,
    validate_repeated_scaffold_folds,
)


def _split_inputs():
    # Twenty scaffold clusters with three compounds each. Activity varies both
    # within and between scaffolds so stratified group shuffling has freedom.
    n_groups = 20
    compounds_per_group = 3
    groups = np.repeat([f"scaffold-{index:02d}" for index in range(n_groups)], compounds_per_group)
    ids = [f"compound-{index:03d}" for index in range(len(groups))]
    group_signal = np.repeat(np.linspace(4.0, 9.0, n_groups), compounds_per_group)
    within_group = np.tile([-0.15, 0.0, 0.15], n_groups)
    y = group_signal + within_group
    return ids, groups, y


def _folds(base_seed=101, n_repeats=3):
    ids, groups, y = _split_inputs()
    return make_repeated_scaffold_folds(
        ids,
        groups,
        y,
        n_splits=5,
        n_repeats=n_repeats,
        base_seed=base_seed,
        n_bins=5,
    )


def test_every_repeat_is_scaffold_disjoint_and_covers_each_sample_once():
    ids, _, _ = _split_inputs()
    folds = _folds()

    validate_repeated_scaffold_folds(
        folds,
        n_samples=len(ids),
        n_repeats=3,
        n_splits=5,
    )
    for repeat in range(3):
        repeat_folds = [assignment for assignment in folds if assignment.repeat == repeat]
        validation_indices = [
            index for assignment in repeat_folds for index in assignment.validation_indices
        ]
        assert sorted(validation_indices) == list(range(len(ids)))
        for assignment in repeat_folds:
            assert set(assignment.train_groups).isdisjoint(assignment.validation_groups)
            assert set(assignment.train_ids).isdisjoint(assignment.validation_ids)


def test_same_seed_replays_identical_assignments():
    first = _folds(base_seed=501)
    second = _folds(base_seed=501)
    assert first == second


def test_different_repeat_seeds_change_at_least_one_assignment():
    first = _folds(base_seed=601, n_repeats=1)
    second = _folds(base_seed=602, n_repeats=1)
    first_validation = [assignment.validation_indices for assignment in first]
    second_validation = [assignment.validation_indices for assignment in second]
    assert first_validation != second_validation


def test_persisted_manifest_is_content_addressed_and_deterministic():
    ids, _, _ = _split_inputs()
    folds = _folds(base_seed=701)
    first = repeated_folds_manifest(
        folds,
        n_samples=len(ids),
        n_repeats=3,
        n_splits=5,
        n_bins=5,
    )
    second = repeated_folds_manifest(
        _folds(base_seed=701),
        n_samples=len(ids),
        n_repeats=3,
        n_splits=5,
        n_bins=5,
    )
    changed = repeated_folds_manifest(
        _folds(base_seed=702),
        n_samples=len(ids),
        n_repeats=3,
        n_splits=5,
        n_bins=5,
    )

    assert first == second
    assert first["split_sha256"] != changed["split_sha256"]


def test_split_input_contract_is_enforced():
    ids, groups, y = _split_inputs()
    with pytest.raises(ValueError, match="equal length"):
        make_repeated_scaffold_folds(ids[:-1], groups, y)
    with pytest.raises(ValueError, match="unique"):
        make_repeated_scaffold_folds([ids[0]] * len(ids), groups, y)
    with pytest.raises(ValueError, match="at least n_splits"):
        make_repeated_scaffold_folds(ids, ["one"] * len(ids), y)
