"""
Tests for the actual preprocessing pipeline (preprocessing_utils.preprocess_features)
and for Y-randomization invariants.
"""

import numpy as np
import pandas as pd
import pytest

from preprocessing_utils import preprocess_features


def _build_synthetic_dataset(n_train=80, n_test=20, n_morgan=64, n_physchem=10, seed=42):
    """Create a synthetic train/test split mimicking the pipeline's feature layout."""
    rng = np.random.RandomState(seed)
    morgan_cols = [f"Morgan_{i}" for i in range(n_morgan)]
    physchem_cols = [f"Phys_{i}" for i in range(n_physchem)]
    all_cols = morgan_cols + physchem_cols

    X_train = pd.DataFrame(rng.rand(n_train, len(all_cols)), columns=all_cols)
    X_test = pd.DataFrame(rng.rand(n_test, len(all_cols)), columns=all_cols)
    y_train = rng.uniform(4, 9, n_train)

    return X_train, X_test, y_train, morgan_cols, physchem_cols


# ---------------------------------------------------------------------------
# Tests that exercise the *actual* preprocess_features function
# ---------------------------------------------------------------------------

class TestPreprocessFeaturesContract:
    """Verify the function's input/output contract and error handling."""

    def test_requires_transformers_when_fit_new_false(self):
        X_train, X_test, y_train, morgan_cols, physchem_cols = _build_synthetic_dataset()
        with pytest.raises(ValueError, match="Must provide transformers"):
            preprocess_features(X_train, X_test, morgan_cols, physchem_cols,
                                y_train, fit_new=False, transformers=None)

    def test_returns_none_val_when_val_is_none(self):
        X_train, _, y_train, morgan_cols, physchem_cols = _build_synthetic_dataset()
        X_proc, X_val_proc, transformers = preprocess_features(
            X_train, None, morgan_cols, physchem_cols, y_train,
        )
        assert X_val_proc is None
        assert X_proc.shape[0] == len(X_train)

    def test_output_shapes_are_consistent(self):
        X_train, X_test, y_train, morgan_cols, physchem_cols = _build_synthetic_dataset()
        X_tr_proc, X_te_proc, _ = preprocess_features(
            X_train, X_test, morgan_cols, physchem_cols, y_train,
        )
        assert X_tr_proc.shape[1] == X_te_proc.shape[1]
        assert X_tr_proc.shape[0] == len(X_train)
        assert X_te_proc.shape[0] == len(X_test)


class TestPreprocessFeaturesNoLeakage:
    """Verify that transformers are fitted ONLY on training data."""

    def test_reusing_transformers_reproduces_output(self):
        """fit_new=True then fit_new=False on the same data should give identical results."""
        X_train, X_test, y_train, morgan_cols, physchem_cols = _build_synthetic_dataset()

        X_tr1, X_te1, transformers = preprocess_features(
            X_train, X_test, morgan_cols, physchem_cols, y_train,
        )
        X_tr2, X_te2, _ = preprocess_features(
            X_train, X_test, morgan_cols, physchem_cols, y_train,
            fit_new=False, transformers=transformers,
        )

        np.testing.assert_array_almost_equal(X_tr1, X_tr2)
        np.testing.assert_array_almost_equal(X_te1, X_te2)

    def test_pca_components_differ_when_trained_on_different_data(self):
        """PCA fitted inside preprocess_features should differ from PCA fitted on combined."""
        X_train, X_test, y_train, morgan_cols, physchem_cols = _build_synthetic_dataset()

        _, _, transformers_train = preprocess_features(
            X_train, X_test, morgan_cols, physchem_cols, y_train,
        )

        combined = pd.concat([X_train, X_test], ignore_index=True)
        y_combined = np.concatenate([y_train, np.zeros(len(X_test))])
        _, _, transformers_combined = preprocess_features(
            combined, None, morgan_cols, physchem_cols, y_combined,
        )

        pca_train = transformers_train['morgan_pca']
        pca_combined = transformers_combined['morgan_pca']

        components_differ = (pca_train.n_components_ != pca_combined.n_components_)
        mean_differs = not np.allclose(pca_train.mean_, pca_combined.mean_)
        n_common = min(pca_train.n_components_, pca_combined.n_components_)
        vals_differ = not np.allclose(
            pca_train.components_[:n_common],
            pca_combined.components_[:n_common],
        )
        assert components_differ or mean_differs or vals_differ

    def test_scaler_statistics_come_from_train_only(self):
        X_train, X_test, y_train, morgan_cols, physchem_cols = _build_synthetic_dataset()

        _, _, transformers = preprocess_features(
            X_train, X_test, morgan_cols, physchem_cols, y_train,
        )
        scaler = transformers['scaler']

        expected_mean = X_train[physchem_cols].mean().values
        np.testing.assert_array_almost_equal(scaler.mean_, expected_mean)

    def test_imputer_statistics_come_from_train_only(self):
        X_train, X_test, y_train, morgan_cols, physchem_cols = _build_synthetic_dataset()
        X_train.iloc[0, 0] = np.nan
        X_test.iloc[0, 0] = np.nan

        _, _, transformers = preprocess_features(
            X_train, X_test, morgan_cols, physchem_cols, y_train,
        )
        imputer = transformers['imputer']

        # Imputer mean for column 0 should match train (excluding NaN), not combined
        train_col_mean = X_train.iloc[1:, 0].mean()
        assert np.isclose(imputer.statistics_[0], train_col_mean)


class TestPreprocessFeaturesEdgeCases:
    def test_no_morgan_cols(self):
        X_train, X_test, y_train, _, physchem_cols = _build_synthetic_dataset()
        X_tr_proc, X_te_proc, _ = preprocess_features(
            X_train, X_test, [], physchem_cols, y_train,
        )
        assert X_tr_proc.shape[0] == len(X_train)
        assert X_te_proc.shape[0] == len(X_test)

    def test_no_physchem_cols(self):
        X_train, X_test, y_train, morgan_cols, _ = _build_synthetic_dataset()
        X_tr_proc, X_te_proc, _ = preprocess_features(
            X_train, X_test, morgan_cols, [], y_train,
        )
        assert X_tr_proc.shape[0] == len(X_train)
        assert X_te_proc.shape[0] == len(X_test)


class TestSkipPCA:
    """Verify the skip_pca path used for tree-based models."""

    def test_skip_pca_preserves_morgan_dimensionality(self):
        X_train, X_test, y_train, morgan_cols, physchem_cols = _build_synthetic_dataset()
        X_tr_proc, X_te_proc, transformers = preprocess_features(
            X_train, X_test, morgan_cols, physchem_cols, y_train,
            skip_pca=True,
        )
        # With skip_pca, Morgan columns pass through unreduced.
        # Total features = all Morgan cols + selected physchem cols.
        n_selected_physchem = transformers['selected_mask'].sum()
        assert X_tr_proc.shape[1] == len(morgan_cols) + n_selected_physchem
        assert X_te_proc.shape[1] == X_tr_proc.shape[1]

    def test_skip_pca_produces_more_features_than_pca(self):
        X_train, X_test, y_train, morgan_cols, physchem_cols = _build_synthetic_dataset()
        _, _, t_pca = preprocess_features(
            X_train, X_test, morgan_cols, physchem_cols, y_train,
            skip_pca=False,
        )
        X_tr_nopca, _, t_nopca = preprocess_features(
            X_train, X_test, morgan_cols, physchem_cols, y_train,
            skip_pca=True,
        )
        pca_n = t_pca['morgan_pca'].n_components_
        nopca_n = len(morgan_cols)
        assert nopca_n > pca_n

    def test_skip_pca_stores_none_for_morgan_pca(self):
        X_train, X_test, y_train, morgan_cols, physchem_cols = _build_synthetic_dataset()
        _, _, transformers = preprocess_features(
            X_train, X_test, morgan_cols, physchem_cols, y_train,
            skip_pca=True,
        )
        assert transformers['morgan_pca'] is None

    def test_skip_pca_reuse_transformers(self):
        X_train, X_test, y_train, morgan_cols, physchem_cols = _build_synthetic_dataset()
        X_tr1, X_te1, transformers = preprocess_features(
            X_train, X_test, morgan_cols, physchem_cols, y_train,
            skip_pca=True,
        )
        X_tr2, X_te2, _ = preprocess_features(
            X_train, X_test, morgan_cols, physchem_cols, y_train,
            fit_new=False, transformers=transformers,
        )
        np.testing.assert_array_almost_equal(X_tr1, X_tr2)
        np.testing.assert_array_almost_equal(X_te1, X_te2)


# ---------------------------------------------------------------------------
# Y-randomization tests
# ---------------------------------------------------------------------------

class TestYRandomization:
    def test_permutation_changes_labels(self):
        rng = np.random.RandomState(0)
        y = np.arange(100, dtype=float)
        y_shuffled = rng.permutation(y)
        assert not np.array_equal(y, y_shuffled)

    def test_permutation_preserves_distribution(self):
        rng = np.random.RandomState(0)
        y = rng.uniform(4, 9, size=200)
        y_shuffled = rng.permutation(y)
        assert np.isclose(y.mean(), y_shuffled.mean(), atol=1e-10)
        assert np.isclose(y.std(), y_shuffled.std(), atol=1e-10)
        assert set(y) == set(y_shuffled)

    def test_permutation_produces_poor_model(self):
        """A model trained on shuffled labels should have near-zero R^2."""
        from sklearn.linear_model import Ridge
        from sklearn.metrics import r2_score

        rng = np.random.RandomState(42)
        X = rng.rand(200, 20)
        y_real = X[:, 0] * 2 + X[:, 1] * 3 + rng.normal(0, 0.1, 200)

        model_real = Ridge().fit(X[:150], y_real[:150])
        r2_real = r2_score(y_real[150:], model_real.predict(X[150:]))

        scores = []
        for i in range(10):
            y_shuf = np.random.RandomState(i).permutation(y_real[:150])
            model_shuf = Ridge().fit(X[:150], y_shuf)
            scores.append(r2_score(y_real[150:], model_shuf.predict(X[150:])))

        assert r2_real > 0.5, "Sanity check: real model should have reasonable R²"
        assert np.mean(scores) < 0.1, "Y-randomized models should have near-zero R²"
