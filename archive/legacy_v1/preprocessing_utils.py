"""
Feature preprocessing utilities for QSAR modeling.

Provides a single preprocessing pipeline that handles imputation, PCA
dimensionality reduction for Morgan fingerprints, scaling and mutual-
information feature selection for physicochemical descriptors — all fitted
exclusively on training data to prevent data leakage.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def preprocess_features(X_train, X_val, morgan_cols, physchem_cols, y_train,
                        fit_new=True, transformers=None, skip_pca=False,
                        random_state=42):
    """
    Preprocess features with transformers fitted ONLY on training data.
    This prevents data leakage during cross-validation.

    Parameters
    ----------
    X_train : DataFrame
        Training features.
    X_val : DataFrame or None
        Validation features (can be None for final fitting).
    morgan_cols : list
        Morgan fingerprint column names.
    physchem_cols : list
        Physicochemical descriptor column names.
    y_train : array-like
        Training target values (used for MI feature selection).
    fit_new : bool
        If True, fit new transformers; if False, use *transformers*.
    transformers : dict or None
        Pre-fitted transformers (required when *fit_new* is False).
    skip_pca : bool
        If True, pass Morgan fingerprints through without PCA reduction.
        Tree-based models (RF, GBR, DT, AdaBoost) work on raw binary
        features and do not benefit from PCA; applying PCA can actually
        hurt their performance by destroying the sparse binary structure
        they split on.
    random_state : int
        Seed used by mutual-information feature selection.  Exposing this
        parameter lets repeated-split analyses vary the full stochastic
        pipeline reproducibly.

    Returns
    -------
    X_train_processed : ndarray
    X_val_processed : ndarray or None
    transformers : dict
    """
    if not fit_new and transformers is None:
        raise ValueError("Must provide transformers if fit_new=False")

    if fit_new:
        transformers = {}

    # Step 1: Handle missing values — fit on train only
    if fit_new:
        imputer = SimpleImputer(strategy='mean')
        X_train_imputed = pd.DataFrame(
            imputer.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index,
        )
        transformers['imputer'] = imputer
    else:
        imputer = transformers['imputer']
        X_train_imputed = pd.DataFrame(
            imputer.transform(X_train),
            columns=X_train.columns,
            index=X_train.index,
        )

    if X_val is not None:
        X_val_imputed = pd.DataFrame(
            imputer.transform(X_val),
            columns=X_val.columns,
            index=X_val.index,
        )

    # Step 2: Morgan fingerprints — PCA or pass-through
    if len(morgan_cols) > 0:
        if skip_pca:
            X_train_morgan = X_train_imputed[morgan_cols].values
            transformers['morgan_pca'] = None
            if X_val is not None:
                X_val_morgan = X_val_imputed[morgan_cols].values
        else:
            if fit_new:
                morgan_pca = PCA(n_components=0.95)
                X_train_morgan = morgan_pca.fit_transform(X_train_imputed[morgan_cols])
                transformers['morgan_pca'] = morgan_pca
            else:
                morgan_pca = transformers['morgan_pca']
                if morgan_pca is None:
                    X_train_morgan = X_train_imputed[morgan_cols].values
                else:
                    X_train_morgan = morgan_pca.transform(X_train_imputed[morgan_cols])

            if X_val is not None:
                if transformers.get('morgan_pca') is None:
                    X_val_morgan = X_val_imputed[morgan_cols].values
                else:
                    X_val_morgan = transformers['morgan_pca'].transform(X_val_imputed[morgan_cols])
    else:
        X_train_morgan = np.empty((len(X_train), 0))
        if X_val is not None:
            X_val_morgan = np.empty((len(X_val), 0))

    # Step 3: Scale + MI feature-select physicochemical descriptors — fit on train only
    if len(physchem_cols) > 0:
        if fit_new:
            scaler = StandardScaler()
            X_train_physchem_scaled = scaler.fit_transform(X_train_imputed[physchem_cols])
            transformers['scaler'] = scaler

            mi_scores = mutual_info_regression(X_train_physchem_scaled, y_train,
                                               random_state=random_state)
            mi_threshold = np.percentile(mi_scores, 70)
            selected_mask = mi_scores >= mi_threshold
            transformers['selected_mask'] = selected_mask
        else:
            scaler = transformers['scaler']
            X_train_physchem_scaled = scaler.transform(X_train_imputed[physchem_cols])
            selected_mask = transformers['selected_mask']

        X_train_physchem = X_train_physchem_scaled[:, selected_mask]

        if X_val is not None:
            X_val_physchem_scaled = scaler.transform(X_val_imputed[physchem_cols])
            X_val_physchem = X_val_physchem_scaled[:, selected_mask]
    else:
        X_train_physchem = np.empty((len(X_train), 0))
        if X_val is not None:
            X_val_physchem = np.empty((len(X_val), 0))

    # Step 4: Combine
    X_train_processed = np.hstack([X_train_morgan, X_train_physchem])

    if X_val is not None:
        X_val_processed = np.hstack([X_val_morgan, X_val_physchem])
        return X_train_processed, X_val_processed, transformers
    else:
        return X_train_processed, None, transformers
