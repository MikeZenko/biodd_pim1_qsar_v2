import warnings
import numpy as np
import pytest

from scaffold_utils import (
    get_scaffold,
    get_scaffold_groups,
    scaffold_split,
    get_scaffold_cv_splits,
    analyze_scaffold_distribution,
    get_scaffold_fallback_count,
    reset_scaffold_fallback_count,
)


class TestGetScaffold:
    def test_benzene_derivatives_share_scaffold(self):
        scaffolds = {get_scaffold(s) for s in [
            "c1ccccc1", "c1ccc(O)cc1", "c1ccc(N)cc1",
        ]}
        assert len(scaffolds) == 1

    def test_different_cores_produce_different_scaffolds(self):
        benzene = get_scaffold("c1ccccc1")
        naphthalene = get_scaffold("c1ccc2ccccc2c1")
        pyridine = get_scaffold("c1ccncc1")
        assert len({benzene, naphthalene, pyridine}) == 3

    def test_invalid_smiles_returns_original(self):
        bad = "NOT_A_SMILES"
        assert get_scaffold(bad) == bad

    def test_fallback_counter_increments_on_bad_smiles(self):
        reset_scaffold_fallback_count()
        get_scaffold("BAD1")
        get_scaffold("BAD2")
        assert get_scaffold_fallback_count() == 2
        reset_scaffold_fallback_count()

    def test_fallback_emits_warning(self):
        reset_scaffold_fallback_count()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            get_scaffold("INVALID_XYZ")
            assert any("singleton scaffold" in str(x.message) for x in w)
        reset_scaffold_fallback_count()


class TestGetScaffoldGroups:
    def test_same_scaffold_same_group(self):
        smiles = ["c1ccccc1", "c1ccc(O)cc1", "c1ccc(N)cc1"]
        groups, _ = get_scaffold_groups(smiles)
        assert groups[0] == groups[1] == groups[2]

    def test_different_scaffolds_different_groups(self):
        smiles = ["c1ccccc1", "c1ccc2ccccc2c1", "c1ccncc1"]
        groups, mapping = get_scaffold_groups(smiles)
        assert len(set(groups)) == 3
        assert len(mapping) == 3

    def test_groups_length_matches_input(self, diverse_smiles):
        groups, _ = get_scaffold_groups(diverse_smiles)
        assert len(groups) == len(diverse_smiles)


class TestScaffoldSplit:
    def test_zero_scaffold_overlap(self, diverse_smiles, diverse_features, diverse_targets):
        _, _, _, _, groups, tr_idx, te_idx = scaffold_split(
            diverse_features, diverse_targets, diverse_smiles,
            test_size=0.3, random_state=42,
        )
        train_scaffolds = set(groups[tr_idx])
        test_scaffolds = set(groups[te_idx])
        assert len(train_scaffolds & test_scaffolds) == 0

    def test_all_indices_covered(self, diverse_smiles, diverse_features, diverse_targets):
        _, _, _, _, _, tr_idx, te_idx = scaffold_split(
            diverse_features, diverse_targets, diverse_smiles,
        )
        all_idx = set(tr_idx) | set(te_idx)
        assert all_idx == set(range(len(diverse_smiles)))

    def test_approximate_split_ratio(self, diverse_smiles, diverse_features, diverse_targets):
        test_size = 0.3
        _, _, _, _, _, tr_idx, te_idx = scaffold_split(
            diverse_features, diverse_targets, diverse_smiles,
            test_size=test_size,
        )
        actual_ratio = len(te_idx) / len(diverse_smiles)
        assert 0.1 <= actual_ratio <= 0.6  # scaffold granularity makes exact match unlikely

    def test_works_with_dataframe_input(self, diverse_smiles, diverse_targets):
        import pandas as pd
        X_df = pd.DataFrame(
            np.random.RandomState(0).rand(len(diverse_smiles), 5),
            columns=[f"f{i}" for i in range(5)],
        )
        y_series = pd.Series(diverse_targets)
        X_tr, X_te, y_tr, y_te, _, _, _ = scaffold_split(X_df, y_series, diverse_smiles)
        assert isinstance(X_tr, pd.DataFrame)
        assert isinstance(y_tr, pd.Series)


class TestScaffoldCVSplits:
    def test_no_scaffold_leakage_across_folds(self, diverse_smiles, diverse_features, diverse_targets):
        n_splits = 3
        splits, groups = get_scaffold_cv_splits(
            diverse_features, diverse_targets, diverse_smiles, n_splits=n_splits,
        )
        assert len(splits) == n_splits
        for train_idx, val_idx in splits:
            train_scaffolds = set(groups[train_idx])
            val_scaffolds = set(groups[val_idx])
            assert len(train_scaffolds & val_scaffolds) == 0

    def test_all_samples_appear_in_validation_once(self, diverse_smiles, diverse_features, diverse_targets):
        n_splits = 3
        splits, _ = get_scaffold_cv_splits(
            diverse_features, diverse_targets, diverse_smiles, n_splits=n_splits,
        )
        all_val = np.concatenate([val for _, val in splits])
        assert set(all_val) == set(range(len(diverse_smiles)))


class TestAnalyzeScaffoldDistribution:
    def test_returns_expected_keys(self, diverse_smiles, diverse_targets):
        stats = analyze_scaffold_distribution(diverse_smiles, diverse_targets)
        for key in [
            "total_scaffolds", "singleton_scaffolds", "max_scaffold_size",
            "mean_scaffold_size", "median_scaffold_size",
            "active_compounds", "inactive_compounds", "active_ratio",
            "top_scaffolds",
        ]:
            assert key in stats, f"Missing key: {key}"

    def test_threshold_active_affects_counts(self, diverse_smiles, diverse_targets):
        stats_low = analyze_scaffold_distribution(diverse_smiles, diverse_targets, threshold_active=5.0)
        stats_high = analyze_scaffold_distribution(diverse_smiles, diverse_targets, threshold_active=8.0)
        assert stats_low["active_compounds"] >= stats_high["active_compounds"]

    def test_active_inactive_sum_to_total(self, diverse_smiles, diverse_targets):
        stats = analyze_scaffold_distribution(diverse_smiles, diverse_targets)
        total = stats["active_compounds"] + stats["inactive_compounds"]
        assert total == len(diverse_smiles)
