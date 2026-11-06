#Comparing different regression models for pancreatic cancer drug discovery
#Updated: Implements scaffold-based splitting with EFFICIENT hyperparameter tuning
#Uses RandomizedSearchCV with fixed compute budget for scientific rigor

#1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from sklearn.model_selection import (train_test_split, RandomizedSearchCV,
                                      cross_val_score, StratifiedKFold, GroupKFold)
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import os
import sys
import pickle
import time
import joblib
import json
from pathlib import Path
from scipy.stats import uniform, randint, loguniform  # loguniform deprecated in SciPy >=1.12; migrate to scipy.stats.log_uniform when removing legacy support

# Import various regression models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              AdaBoostRegressor, VotingRegressor, StackingRegressor)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# Import scaffold utilities for scaffold-based splitting
from scaffold_utils import scaffold_split, get_scaffold_groups, analyze_scaffold_distribution

# Import shared preprocessing (extracted so tests can exercise the actual code)
from preprocessing_utils import preprocess_features


print("Step 5: Comparing Different Regression Models")
print("Using SCAFFOLD-BASED splitting + RandomizedSearchCV (budgeted tuning)")

# Parse optional target filters
parser = argparse.ArgumentParser(description="Compare regression models for a specific target.")
parser.add_argument("--target-id", help="ChEMBL target ID to filter (e.g., CHEMBL2147)")
parser.add_argument("--target-name", help="Override target name for labeling outputs")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed for splitting, model fitting, and randomized searches (default: 42)")
parser.add_argument("--force", action="store_true",
                    help="Ignore cached models and fold results, then recompute this run")
args = parser.parse_args()

RANDOM_SEED = args.seed
STACKING_PROTOCOL_VERSION = "group_oof_v1"

# Resolve paths relative to script location
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

#2. Load the dataset prepared in Step 3
input_file = DATA_DIR / 'pancreatic_cancer_bioactivity_data_3class_pIC50_morgan_fp.csv'
smiles_file = DATA_DIR / 'bioactivity_data_preprocessed.csv'

if not os.path.exists(input_file):
    print(f"ERROR: Input file {input_file} not found! Run Step 3 first.")
    sys.exit(1)

try:
    df = pd.read_csv(input_file)
    print(f"Loaded dataset with {df.shape[0]} compounds and {df.shape[1]} features")
except Exception as e:
    print(f"ERROR loading dataset: {e}")
    sys.exit(1)

# Filter by target if target metadata is available
target_id = None
target_name = None
if 'target_chembl_id' in df.columns:
    unique_targets = df['target_chembl_id'].dropna().unique()
    if args.target_id:
        df = df[df['target_chembl_id'] == args.target_id].copy()
        if df.empty:
            print(f"ERROR: No data found for target {args.target_id}")
            sys.exit(1)
        target_id = args.target_id
        target_name = args.target_name or df['target_name'].iloc[0] if 'target_name' in df.columns else args.target_id
    elif len(unique_targets) > 1:
        print("ERROR: Multiple targets present in dataset. Specify --target-id to train a target-specific model.")
        print(f"Available targets: {', '.join(unique_targets)}")
        sys.exit(1)
    else:
        target_id = unique_targets[0] if len(unique_targets) == 1 else None
        target_name = df['target_name'].iloc[0] if 'target_name' in df.columns else target_id
elif args.target_id:
    print("WARNING: --target-id provided but dataset has no target_chembl_id column. Ignoring filter.")

if target_id:
    print(f"Training target-specific model for {target_id} ({target_name})")

# Check minimum sample size for reliable QSAR modeling
MIN_SAMPLES = 300  # Minimum compounds for reliable QSAR (conservative threshold)
if len(df) < MIN_SAMPLES:
    print(f"\nWARNING: Only {len(df)} compounds available for {target_id}.")
    print(f"         Minimum recommended: {MIN_SAMPLES} compounds.")
    print(f"         Results may not be statistically reliable.")
    print(f"         Consider using this target for exploratory analysis only.\n")
    if len(df) < 100:
        print(f"ERROR: Dataset too small ({len(df)} < 100). Exiting.")
        sys.exit(1)

# Configure output directories
models_dir = PROJECT_ROOT / 'models'
plots_dir = PROJECT_ROOT / 'plots'
results_dir = PROJECT_ROOT / 'results'
if target_id:
    models_dir = models_dir / target_id
    plots_dir = plots_dir / target_id
    results_dir = results_dir / target_id

models_dir.mkdir(parents=True, exist_ok=True)
plots_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)

# Create fold results directory for audit trail
fold_results_dir = results_dir / 'fold_results'
fold_results_dir.mkdir(parents=True, exist_ok=True)

# Load SMILES data for scaffold-based splitting
if 'canonical_smiles' in df.columns:
    smiles_list = df['canonical_smiles'].tolist()
    print(f"Loaded {len(smiles_list)} SMILES for scaffold extraction from dataset")
elif os.path.exists(smiles_file):
    smiles_df = pd.read_csv(smiles_file)
    smiles_list = smiles_df['canonical_smiles'].tolist()
    smiles_list = smiles_list[:len(df)]
    print(f"Loaded {len(smiles_list)} SMILES for scaffold extraction from {smiles_file}")
else:
    print(f"WARNING: SMILES data not found!")
    print("Falling back to random splitting (not recommended)")
    smiles_list = None

#3. Prepare input and output variables
non_feature_cols = ['target_name', 'target_chembl_id', 'canonical_smiles', 'molecule_chembl_id']
X = df.drop(columns=[col for col in non_feature_cols if col in df.columns], errors='ignore')
X = X.drop('pIC50', axis=1)
Y = df.pIC50

print(f"Input features: {X.shape[1]} molecular descriptors")
print(f"Output variable: pIC50 with range {Y.min():.2f} to {Y.max():.2f}")

# Separate Morgan fingerprints from physicochemical descriptors
morgan_cols = [col for col in X.columns if col.startswith('Morgan_')]
physchem_cols = [col for col in X.columns if not col.startswith('Morgan_')]
print(f"Morgan fingerprint bits: {len(morgan_cols)}")
print(f"Physicochemical descriptors: {len(physchem_cols)}")

#4. Scaffold-based data splitting (80/20)
print("\n" + "="*60)
print("SCAFFOLD-BASED DATA SPLITTING")
print("="*60)

if smiles_list is not None and len(smiles_list) == len(X):
    X_dev, X_holdout, Y_dev, Y_holdout, scaffold_groups_all, dev_idx, holdout_idx = scaffold_split(
        X, Y, smiles_list, test_size=0.2, random_state=RANDOM_SEED
    )
    scaffold_groups_dev = scaffold_groups_all[dev_idx]
    use_scaffold_cv = True
else:
    print("WARNING: Using random split (scaffold data not available)")
    n_bins = min(5, len(np.unique(Y)))
    Y_bins = pd.qcut(Y, q=n_bins, labels=False, duplicates='drop')
    X_dev, X_holdout, Y_dev, Y_holdout, Y_bins_dev, Y_bins_holdout = train_test_split(
        X, Y, Y_bins, test_size=0.2, random_state=RANDOM_SEED, stratify=Y_bins
    )
    Y_bins_dev = np.array(Y_bins_dev)
    scaffold_groups_dev = None
    use_scaffold_cv = False

print(f"Development set: {X_dev.shape[0]} compounds, Holdout set: {X_holdout.shape[0]} compounds")
if use_scaffold_cv:
    print("  Note: GroupShuffleSplit applies test_size to scaffold groups, not compounds.")

# NOTE: Feature preprocessing (imputation, PCA, scaling, feature selection) is now done
# INSIDE the CV loop to prevent data leakage. Final transformers for holdout evaluation
# are fitted after CV completes.
print("\nPreprocessing will be done inside CV loop to prevent data leakage...")
print(f"  Morgan fingerprint columns: {len(morgan_cols)}")
print(f"  Physicochemical descriptor columns: {len(physchem_cols)}")

#5. Define models with RANDOMIZED parameter distributions
# Fixed compute budget: n_iter=20 for all models (statistically efficient)
N_ITER_BUDGET = 20  # Standard budget for RandomizedSearchCV (20-50 is typical)

print("\n" + "="*60)
print(f"MODEL COMPARISON WITH BUDGETED TUNING (n_iter={N_ITER_BUDGET})")
print("="*60)

# Tree-based models receive raw (unreduced) Morgan fingerprints — PCA destroys
# the sparse binary structure that decision-tree splits exploit.  Linear and
# distance-based models still benefit from PCA dimensionality reduction.
TREE_BASED_MODELS = {
    'RandomForestRegressor', 'GradientBoostingRegressor',
    'AdaBoostRegressor', 'DecisionTreeRegressor',
}

# Parameter distributions for RandomizedSearchCV
models_with_params = {
    'LinearRegression': {
        'model': LinearRegression(),
        'params': None,  # No hyperparameters
        'skip_pca': False,
    },
    'Ridge': {
        'model': Ridge(random_state=RANDOM_SEED),
        'params': {
            'alpha': loguniform(1e-3, 1e3)
        },
        'skip_pca': False,
    },
    'Lasso': {
        'model': Lasso(random_state=RANDOM_SEED, max_iter=5000),
        'params': {
            'alpha': loguniform(1e-5, 1e1)
        },
        'skip_pca': False,
    },
    'ElasticNet': {
        'model': ElasticNet(random_state=RANDOM_SEED, max_iter=5000),
        'params': {
            'alpha': loguniform(1e-5, 1e1),
            'l1_ratio': uniform(0.1, 0.9)
        },
        'skip_pca': False,
    },
    # SVR SKIPPED - too slow for this sample size even with linear kernel
    # Ridge and ElasticNet provide similar regularized linear models
    'KNeighborsRegressor': {
        'model': KNeighborsRegressor(n_jobs=1),  # Single-threaded
        'params': {
            'n_neighbors': randint(3, 20),
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        },
        'skip_pca': False,
    },
    'RandomForestRegressor': {
        'model': RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=1),  # SINGLE-THREADED model
        'params': {
            'n_estimators': randint(50, 150),
            'max_depth': [None, 10, 20],
            'max_features': ['sqrt', 0.3],
            'min_samples_leaf': [1, 2, 4],
            'min_samples_split': [2, 5, 10]
        },
        'skip_pca': True,
    },
    'GradientBoostingRegressor': {
        'model': GradientBoostingRegressor(random_state=RANDOM_SEED),  # Single-threaded
        'params': {
            'n_estimators': randint(30, 80),
            'learning_rate': loguniform(0.01, 0.2),
            'max_depth': randint(2, 5),
            'subsample': uniform(0.6, 0.4),
            'min_samples_leaf': [2, 4]
        },
        'skip_pca': True,
    },
    'AdaBoostRegressor': {
        'model': AdaBoostRegressor(random_state=RANDOM_SEED),
        'params': {
            'n_estimators': randint(50, 200),
            'learning_rate': loguniform(0.01, 2.0)
        },
        'skip_pca': True,
    },
    'DecisionTreeRegressor': {
        'model': DecisionTreeRegressor(random_state=RANDOM_SEED),
        'params': {
            'max_depth': [None, 5, 10, 20, 40],
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10)
        },
        'skip_pca': True,
    }
}

# Initialize results storage
nested_cv_results = []
train_results_list = []
test_results_list = []
all_fold_scores = {}  # Store fold-by-fold scores for statistical comparison
final_transformers_by_model = {}

# CV configuration
OUTER_FOLDS = 5
INNER_FOLDS = 3

if use_scaffold_cv:
    print(f"\nUsing SCAFFOLD-AWARE cross-validation")
    print(f"  Outer folds: {OUTER_FOLDS} (GroupKFold)")
    print(f"  Inner folds: {INNER_FOLDS} (GroupKFold)")
    outer_cv = GroupKFold(n_splits=OUTER_FOLDS)
    inner_cv = GroupKFold(n_splits=INNER_FOLDS)
else:
    print(f"\nUsing stratified cross-validation (no scaffold data)")
    outer_cv = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    inner_cv = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_SEED)

def save_fold_result(model_name, fold_idx, results_dict, fold_dir):
    """Save individual fold results to disk for audit trail."""
    filename = fold_dir / f'{model_name}_fold{fold_idx+1}.json'
    payload = dict(results_dict)
    payload['random_seed'] = RANDOM_SEED
    with open(filename, 'w') as f:
        json.dump(payload, f, indent=2)


def has_valid_fold_cache(fold_path):
    """Only reuse a fold cache created with the requested split seed."""
    if not fold_path.exists():
        return False
    try:
        with open(fold_path, 'r') as f:
            cached = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return cached.get('random_seed') == RANDOM_SEED

# Train and evaluate each model with nested cross-validation
for name, model_info in models_with_params.items():
    skip_pca = model_info.get('skip_pca', False)
    print(f"\n{'='*50}")
    print(f"Training {name}{'  [skip_pca]' if skip_pca else ''}...")
    print(f"{'='*50}")
    start_time = time.time()

    # CHECK FOR RESUME
    model_filename = models_dir / f'{name.lower()}_tuned.pkl'
    fold_files_exist = True
    for i in range(OUTER_FOLDS):
        if not has_valid_fold_cache(fold_results_dir / f'{name}_fold{i+1}.json'):
            fold_files_exist = False
            break

    if not args.force and os.path.exists(model_filename) and fold_files_exist:
        print(f"  Model {name} already trained. Loading results...")
        try:
            with open(model_filename, 'rb') as f:
                best_model = pickle.load(f)
            nested_scores = []
            for i in range(OUTER_FOLDS):
                with open(fold_results_dir / f'{name}_fold{i+1}.json', 'r') as f:
                    nested_scores.append(json.load(f)['r2'])

            cv_mean = np.mean(nested_scores)
            cv_std = np.std(nested_scores)
            all_fold_scores[name] = nested_scores

            # Preprocess data for predictions
            X_dev_proc, X_holdout_proc, cached_transformers = preprocess_features(
                X_dev, X_holdout, morgan_cols, physchem_cols, Y_dev.values,
                skip_pca=skip_pca,
                random_state=RANDOM_SEED,
            )
            final_transformers_by_model[name] = cached_transformers

            y_train_pred = best_model.predict(X_dev_proc)
            y_test_pred = best_model.predict(X_holdout_proc)

            train_r2 = r2_score(Y_dev, y_train_pred)
            test_r2 = r2_score(Y_holdout, y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(Y_dev, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(Y_holdout, y_test_pred))
            train_mae = mean_absolute_error(Y_dev, y_train_pred)
            test_mae = mean_absolute_error(Y_holdout, y_test_pred)

            print(f"  Loaded: CV R²={cv_mean:.4f}, Test R²={test_r2:.4f}")

            nested_cv_results.append({'Model': name, 'Nested_CV_R_Squared_Mean': cv_mean, 'Nested_CV_R_Squared_Std': cv_std, 'Train_R_Squared': train_r2, 'Test_R_Squared': test_r2, 'Time_Taken': 0.0})
            train_results_list.append({'Model': name, 'R-Squared': train_r2, 'RMSE': train_rmse, 'MAE': train_mae, 'Time Taken': 0.0})
            test_results_list.append({'Model': name, 'R-Squared': test_r2, 'RMSE': test_rmse, 'MAE': test_mae, 'Time Taken': 0.0})
            continue
        except Exception as e:
            print(f"  Error loading cache: {e}. Retraining...")

    model = model_info['model']
    params = model_info['params']

    nested_scores = []
    fold_details = []

    # Create outer CV iterator - use RAW X_dev (preprocessing done inside loop)
    if use_scaffold_cv:
        outer_cv_iter = list(outer_cv.split(X_dev, Y_dev, groups=scaffold_groups_dev))
    else:
        outer_cv_iter = list(outer_cv.split(X_dev, Y_bins_dev))

    if params:  # Models with hyperparameters
        for fold_idx, (train_idx, val_idx) in enumerate(outer_cv_iter):
            fold_start = time.time()
            print(f"  Fold {fold_idx+1}/{OUTER_FOLDS}...")

            # Get raw data for this fold
            X_train_raw = X_dev.iloc[train_idx]
            X_val_raw = X_dev.iloc[val_idx]
            y_train_outer = Y_dev.iloc[train_idx].values
            y_val_outer = Y_dev.iloc[val_idx].values

            # CRITICAL FIX: Preprocess INSIDE fold - fit on train only
            X_train_outer, X_val_outer, fold_transformers = preprocess_features(
                X_train_raw, X_val_raw, morgan_cols, physchem_cols, y_train_outer,
                skip_pca=skip_pca,
                random_state=RANDOM_SEED,
            )

            # Inner CV for hyperparameter tuning
            if use_scaffold_cv:
                inner_groups = scaffold_groups_dev[train_idx]
                inner_cv_splits = list(inner_cv.split(X_train_outer, y_train_outer, groups=inner_groups))
            else:
                inner_cv_splits = list(inner_cv.split(X_train_outer, Y_bins_dev[train_idx]))

            # RandomizedSearchCV with parallel search, single-threaded model
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=params,
                n_iter=N_ITER_BUDGET,
                cv=inner_cv_splits,
                scoring='r2',
                n_jobs=-1,  # PARALLELIZE the search
                random_state=RANDOM_SEED + fold_idx,
                verbose=0
            )

            search.fit(X_train_outer, y_train_outer)

            # Evaluate on validation fold
            y_pred = search.predict(X_val_outer)
            score = r2_score(y_val_outer, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val_outer, y_pred))
            mae = mean_absolute_error(y_val_outer, y_pred)

            nested_scores.append(score)
            fold_time = time.time() - fold_start

            fold_result = {
                'fold': fold_idx + 1,
                'r2': float(score),
                'rmse': float(rmse),
                'mae': float(mae),
                'best_params': {k: str(v) for k, v in search.best_params_.items()},
                'best_inner_cv_score': float(search.best_score_),
                'time_seconds': float(fold_time)
            }
            fold_details.append(fold_result)

            # Save fold result to disk
            save_fold_result(name, fold_idx, fold_result, fold_results_dir)

            print(f"    R²={score:.4f}, RMSE={rmse:.4f}, Time={fold_time:.1f}s")
            print(f"    Best params: {search.best_params_}")

        cv_mean = np.mean(nested_scores)
        cv_std = np.std(nested_scores)
        print(f"  Nested CV R²: {cv_mean:.4f} ± {cv_std:.4f}")

    else:  # LinearRegression - no hyperparameters
        for fold_idx, (train_idx, val_idx) in enumerate(outer_cv_iter):
            fold_start = time.time()
            print(f"  Fold {fold_idx+1}/{OUTER_FOLDS}...")

            # Get raw data for this fold
            X_train_raw = X_dev.iloc[train_idx]
            X_val_raw = X_dev.iloc[val_idx]
            y_train_outer = Y_dev.iloc[train_idx].values
            y_val_outer = Y_dev.iloc[val_idx].values

            # CRITICAL FIX: Preprocess INSIDE fold - fit on train only
            X_train_outer, X_val_outer, fold_transformers = preprocess_features(
                X_train_raw, X_val_raw, morgan_cols, physchem_cols, y_train_outer,
                skip_pca=skip_pca,
                random_state=RANDOM_SEED,
            )

            model.fit(X_train_outer, y_train_outer)
            y_pred = model.predict(X_val_outer)

            score = r2_score(y_val_outer, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val_outer, y_pred))
            mae = mean_absolute_error(y_val_outer, y_pred)

            nested_scores.append(score)
            fold_time = time.time() - fold_start

            fold_result = {
                'fold': fold_idx + 1,
                'r2': float(score),
                'rmse': float(rmse),
                'mae': float(mae),
                'best_params': {},
                'time_seconds': float(fold_time)
            }
            fold_details.append(fold_result)
            save_fold_result(name, fold_idx, fold_result, fold_results_dir)

            print(f"    R²={score:.4f}, RMSE={rmse:.4f}, Time={fold_time:.1f}s")

        cv_mean = np.mean(nested_scores)
        cv_std = np.std(nested_scores)
        print(f"  CV R²: {cv_mean:.4f} ± {cv_std:.4f}")

    # Store fold-by-fold scores for statistical comparison
    all_fold_scores[name] = nested_scores

    # Final model training on all development data
    print(f"  Training final model on all development data...")

    # CRITICAL: Preprocess full dev set and holdout with dev-fitted transformers
    X_dev_processed, X_holdout_processed, final_transformers = preprocess_features(
        X_dev, X_holdout, morgan_cols, physchem_cols, Y_dev.values,
        skip_pca=skip_pca,
        random_state=RANDOM_SEED,
    )

    if params:
        # Final search on full development set
        if use_scaffold_cv:
            final_cv_splits = list(inner_cv.split(X_dev_processed, Y_dev, groups=scaffold_groups_dev))
        else:
            final_cv_splits = INNER_FOLDS

        final_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=params,
            n_iter=N_ITER_BUDGET,
            cv=final_cv_splits,
            scoring='r2',
            n_jobs=-1,
            random_state=RANDOM_SEED,
            verbose=0
        )
        final_search.fit(X_dev_processed, Y_dev)
        best_model = final_search.best_estimator_
        best_params = final_search.best_params_
        print(f"  Final best params: {best_params}")
    else:
        best_model = model
        best_model.fit(X_dev_processed, Y_dev)
        best_params = {}

    # Predictions and metrics
    y_train_pred = best_model.predict(X_dev_processed)
    y_test_pred = best_model.predict(X_holdout_processed)

    train_r2 = r2_score(Y_dev, y_train_pred)
    test_r2 = r2_score(Y_holdout, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(Y_dev, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(Y_holdout, y_test_pred))
    train_mae = mean_absolute_error(Y_dev, y_train_pred)
    test_mae = mean_absolute_error(Y_holdout, y_test_pred)

    time_taken = time.time() - start_time

    # Save model
    model_filename = models_dir / f'{name.lower()}_tuned.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"  Model saved to {model_filename}")
    print(f"  Train R²={train_r2:.4f}, Test R²={test_r2:.4f}, Total time={time_taken:.1f}s")

    # Save transformers for Steps 6/7.  These must reflect the skip_pca=False (PCA)
    # representation used by the best model and the Ridge interpretability analysis, so
    # only persist them from non-tree models: tree models set morgan_pca=None (skip_pca=True)
    # and would otherwise overwrite morgan_transformer.pkl with a null PCA.
    if not skip_pca:
        if 'imputer' in final_transformers:
            joblib.dump(final_transformers['imputer'], models_dir / 'imputer.pkl')
        if final_transformers.get('morgan_pca') is not None:
            joblib.dump(final_transformers['morgan_pca'], models_dir / 'morgan_transformer.pkl')
        if 'scaler' in final_transformers:
            joblib.dump(final_transformers['scaler'], models_dir / 'physchem_scaler.pkl')
        if 'selected_mask' in final_transformers:
            joblib.dump(final_transformers['selected_mask'], models_dir / 'selected_features_mask.pkl')
    final_transformers_by_model[name] = final_transformers

    # Store results
    nested_cv_results.append({
        'Model': name,
        'Nested_CV_R_Squared_Mean': cv_mean,
        'Nested_CV_R_Squared_Std': cv_std,
        'Train_R_Squared': train_r2,
        'Test_R_Squared': test_r2,
        'Time_Taken': time_taken
    })

    train_results_list.append({
        'Model': name, 'R-Squared': train_r2, 'RMSE': train_rmse,
        'MAE': train_mae, 'Time Taken': time_taken
    })

    test_results_list.append({
        'Model': name, 'R-Squared': test_r2, 'RMSE': test_rmse,
        'MAE': test_mae, 'Time Taken': time_taken
    })

# ============================================================
# ENSEMBLE MODELS WITH NESTED CV
# ============================================================
print("\n" + "="*60)
print("TRAINING ENSEMBLE MODELS WITH NESTED CROSS-VALIDATION")
print("="*60)

def get_base_models_for_ensemble():
    """
    Create fresh instances of base models for ensemble.
    
    NOTE: These use deliberately simpler hyperparameters than the individually
    tuned models to reduce overfitting risk in the ensemble. Ensembles benefit
    from diverse, slightly under-fit base learners rather than fully optimized ones.
    """
    return [
        ('Ridge', Ridge(alpha=10.0, random_state=RANDOM_SEED)),
        ('ElasticNet', ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=RANDOM_SEED, max_iter=5000)),
        ('RandomForest', RandomForestRegressor(n_estimators=50, max_depth=10, random_state=RANDOM_SEED, n_jobs=1)),
        ('GradientBoosting', GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=RANDOM_SEED)),
        ('KNN', KNeighborsRegressor(n_neighbors=7, weights='distance'))
    ]

def build_ensemble(ensemble_name, X_fit, y_fit, groups=None):
    """Build an ensemble with a group-disjoint stacker when scaffold groups exist."""
    base_models = get_base_models_for_ensemble()
    if ensemble_name == 'VotingEnsemble':
        return VotingRegressor(estimators=base_models, n_jobs=-1)

    if ensemble_name != 'StackingEnsemble':
        raise ValueError(f"Unknown ensemble: {ensemble_name}")

    if groups is not None:
        unique_groups = np.unique(groups)
        if len(unique_groups) < INNER_FOLDS:
            raise ValueError(
                f"Need at least {INNER_FOLDS} scaffold groups for group-aware stacking; "
                f"found {len(unique_groups)}."
            )
        stack_cv = list(
            GroupKFold(n_splits=INNER_FOLDS).split(X_fit, y_fit, groups=groups)
        )
    else:
        # The random-split fallback retains ordinary CV because no scaffold labels exist.
        stack_cv = INNER_FOLDS

    return StackingRegressor(
        estimators=base_models,
        final_estimator=Ridge(alpha=1.0, random_state=RANDOM_SEED),
        cv=stack_cv,
        n_jobs=-1,
    )


ENSEMBLE_NAMES = ('VotingEnsemble', 'StackingEnsemble')


def has_valid_ensemble_fold_cache(ensemble_name, fold_path):
    """Reject caches with another seed or legacy stacking OOF folds."""
    if not has_valid_fold_cache(fold_path):
        return False
    with open(fold_path, 'r') as f:
        cached = json.load(f)
    if ensemble_name == 'StackingEnsemble':
        return cached.get('stacking_protocol') == STACKING_PROTOCOL_VERSION
    return True

for ensemble_name in ENSEMBLE_NAMES:
    print(f"\n{'='*50}")
    print(f"Training {ensemble_name}...")
    print(f"{'='*50}")
    start_time = time.time()

    # CHECK FOR RESUME
    model_filename = models_dir / f'{ensemble_name.lower()}.pkl'
    fold_files_exist = True
    for i in range(OUTER_FOLDS):
        fold_path = fold_results_dir / f'{ensemble_name}_fold{i+1}.json'
        if not has_valid_ensemble_fold_cache(ensemble_name, fold_path):
            fold_files_exist = False
            break

    if not args.force and os.path.exists(model_filename) and fold_files_exist:
        print(f"  Model {ensemble_name} already trained. Loading results...")
        try:
            with open(model_filename, 'rb') as f:
                final_ensemble = pickle.load(f)
            ensemble_cv_scores = []
            for i in range(OUTER_FOLDS):
                with open(fold_results_dir / f'{ensemble_name}_fold{i+1}.json', 'r') as f:
                    ensemble_cv_scores.append(json.load(f)['r2'])

            cv_mean = np.mean(ensemble_cv_scores)
            cv_std = np.std(ensemble_cv_scores)
            all_fold_scores[ensemble_name] = ensemble_cv_scores

            # Preprocess data for predictions
            X_dev_proc, X_holdout_proc, cached_transformers = preprocess_features(
                X_dev, X_holdout, morgan_cols, physchem_cols, Y_dev.values,
                random_state=RANDOM_SEED,
            )
            final_transformers_by_model[ensemble_name] = cached_transformers

            train_pred = final_ensemble.predict(X_dev_proc)
            test_pred = final_ensemble.predict(X_holdout_proc)

            train_r2 = r2_score(Y_dev, train_pred)
            test_r2 = r2_score(Y_holdout, test_pred)
            train_rmse = np.sqrt(mean_squared_error(Y_dev, train_pred))
            test_rmse = np.sqrt(mean_squared_error(Y_holdout, test_pred))
            train_mae = mean_absolute_error(Y_dev, train_pred)
            test_mae = mean_absolute_error(Y_holdout, test_pred)

            print(f"  Loaded: CV R²={cv_mean:.4f}, Test R²={test_r2:.4f}")

            nested_cv_results.append({'Model': ensemble_name, 'Nested_CV_R_Squared_Mean': cv_mean, 'Nested_CV_R_Squared_Std': cv_std, 'Train_R_Squared': train_r2, 'Test_R_Squared': test_r2, 'Time_Taken': 0.0})
            train_results_list.append({'Model': ensemble_name, 'R-Squared': train_r2, 'RMSE': train_rmse, 'MAE': train_mae, 'Time Taken': 0.0})
            test_results_list.append({'Model': ensemble_name, 'R-Squared': test_r2, 'RMSE': test_rmse, 'MAE': test_mae, 'Time Taken': 0.0})
            continue
        except Exception as e:
            print(f"  Error loading cache: {e}. Retraining...")

    ensemble_cv_scores = []
    fold_details = []

    # Use RAW X_dev for CV iterator
    if use_scaffold_cv:
        outer_cv_iter = list(outer_cv.split(X_dev, Y_dev, groups=scaffold_groups_dev))
    else:
        outer_cv_iter = list(outer_cv.split(X_dev, Y_bins_dev))

    for fold_idx, (train_idx, val_idx) in enumerate(outer_cv_iter):
        # RESUME: Check if fold result exists
        fold_json_path = fold_results_dir / f'{ensemble_name}_fold{fold_idx+1}.json'
        if not args.force and has_valid_ensemble_fold_cache(ensemble_name, fold_json_path):
            print(f"  Fold {fold_idx+1} already computed. Loading...")
            try:
                with open(fold_json_path, 'r') as f:
                    fold_res = json.load(f)
                    ensemble_cv_scores.append(fold_res['r2'])
                continue
            except Exception:
                pass

        fold_start = time.time()
        print(f"  Fold {fold_idx+1}/{OUTER_FOLDS}...")

        # Get raw data for this fold
        X_train_raw = X_dev.iloc[train_idx]
        X_val_raw = X_dev.iloc[val_idx]
        y_train_outer = Y_dev.iloc[train_idx].values
        y_val_outer = Y_dev.iloc[val_idx].values

        # CRITICAL FIX: Preprocess INSIDE fold - fit on train only
        X_train_outer, X_val_outer, fold_transformers = preprocess_features(
            X_train_raw, X_val_raw, morgan_cols, physchem_cols, y_train_outer,
            random_state=RANDOM_SEED,
        )

        outer_groups = scaffold_groups_dev[train_idx] if use_scaffold_cv else None
        ensemble = build_ensemble(
            ensemble_name, X_train_outer, y_train_outer, groups=outer_groups
        )

        try:
            ensemble.fit(X_train_outer, y_train_outer)
            y_pred = ensemble.predict(X_val_outer)
            score = r2_score(y_val_outer, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val_outer, y_pred))
            mae = mean_absolute_error(y_val_outer, y_pred)

            ensemble_cv_scores.append(score)
            fold_time = time.time() - fold_start

            fold_result = {
                'fold': fold_idx + 1,
                'r2': float(score),
                'rmse': float(rmse),
                'mae': float(mae),
                'time_seconds': float(fold_time),
            }
            if ensemble_name == 'StackingEnsemble':
                fold_result['stacking_protocol'] = STACKING_PROTOCOL_VERSION
            fold_details.append(fold_result)
            save_fold_result(ensemble_name, fold_idx, fold_result, fold_results_dir)

            print(f"    R²={score:.4f}, RMSE={rmse:.4f}, Time={fold_time:.1f}s")
        except Exception as e:
            print(f"    Fold {fold_idx+1} failed: {e}")
            continue

    if len(ensemble_cv_scores) > 0:
        cv_mean = np.mean(ensemble_cv_scores)
        cv_std = np.std(ensemble_cv_scores)
        all_fold_scores[ensemble_name] = ensemble_cv_scores
        print(f"  Nested CV R²: {cv_mean:.4f} ± {cv_std:.4f}")
    else:
        cv_mean = np.nan
        cv_std = np.nan
        print(f"  WARNING: {ensemble_name} nested CV failed")

    # Final ensemble
    print(f"  Training final {ensemble_name} on all development data...")

    # CRITICAL: Preprocess full dev set and holdout with dev-fitted transformers
    X_dev_processed, X_holdout_processed, final_transformers = preprocess_features(
        X_dev, X_holdout, morgan_cols, physchem_cols, Y_dev.values,
        random_state=RANDOM_SEED,
    )

    final_groups = scaffold_groups_dev if use_scaffold_cv else None
    final_ensemble = build_ensemble(
        ensemble_name, X_dev_processed, Y_dev.values, groups=final_groups
    )
    final_ensemble.fit(X_dev_processed, Y_dev)

    train_pred = final_ensemble.predict(X_dev_processed)
    test_pred = final_ensemble.predict(X_holdout_processed)

    train_r2 = r2_score(Y_dev, train_pred)
    test_r2 = r2_score(Y_holdout, test_pred)
    train_rmse = np.sqrt(mean_squared_error(Y_dev, train_pred))
    test_rmse = np.sqrt(mean_squared_error(Y_holdout, test_pred))
    train_mae = mean_absolute_error(Y_dev, train_pred)
    test_mae = mean_absolute_error(Y_holdout, test_pred)

    time_taken = time.time() - start_time

    # Save with underscore for compatibility with Step 6
    if ensemble_name == 'StackingEnsemble':
        filename_str = 'stacking_ensemble.pkl'
    elif ensemble_name == 'VotingEnsemble':
        filename_str = 'voting_ensemble.pkl'
    else:
        filename_str = f'{ensemble_name.lower()}.pkl'
        
    with open(models_dir / filename_str, 'wb') as f:
        pickle.dump(final_ensemble, f)
    # Also save as lower() for resume logic consistency in this script
    if filename_str != f'{ensemble_name.lower()}.pkl':
        with open(models_dir / f'{ensemble_name.lower()}.pkl', 'wb') as f:
            pickle.dump(final_ensemble, f)
    final_transformers_by_model[ensemble_name] = final_transformers
         
    print(f"  Train R²={train_r2:.4f}, Test R²={test_r2:.4f}, Total time={time_taken:.1f}s")

    nested_cv_results.append({
        'Model': ensemble_name,
        'Nested_CV_R_Squared_Mean': cv_mean,
        'Nested_CV_R_Squared_Std': cv_std,
        'Train_R_Squared': train_r2,
        'Test_R_Squared': test_r2,
        'Time_Taken': time_taken
    })

    train_results_list.append({
        'Model': ensemble_name, 'R-Squared': train_r2, 'RMSE': train_rmse,
        'MAE': train_mae, 'Time Taken': time_taken
    })

    test_results_list.append({
        'Model': ensemble_name, 'R-Squared': test_r2, 'RMSE': test_rmse,
        'MAE': test_mae, 'Time Taken': time_taken
    })

# ============================================================
# SAVE RESULTS
# ============================================================
nested_cv_df = pd.DataFrame(nested_cv_results)
train_results = pd.DataFrame(train_results_list)
test_results = pd.DataFrame(test_results_list)

# Lock the downstream model choice before inspecting holdout performance: the
# highest mean outer-fold score is the pre-specified selection criterion.
nested_cv_df = nested_cv_df.sort_values(
    ['Nested_CV_R_Squared_Mean', 'Nested_CV_R_Squared_Std'],
    ascending=[False, True],
).reset_index(drop=True)
train_results = train_results.sort_values('R-Squared', ascending=False)
test_results = test_results.sort_values('R-Squared', ascending=False)

selected_model_name = nested_cv_df.iloc[0]['Model']
selected_model_cv_mean = float(nested_cv_df.iloc[0]['Nested_CV_R_Squared_Mean'])
selected_model_cv_std = float(nested_cv_df.iloc[0]['Nested_CV_R_Squared_Std'])
selected_model_holdout = test_results.loc[
    test_results['Model'] == selected_model_name, 'R-Squared'
].iloc[0]
selection_metadata = {
    'selected_model': selected_model_name,
    'selection_rule': 'highest mean outer-fold R2 from inner-tuned GroupKFold; holdout metrics were not used for selection',
    'nested_cv_r2_mean': selected_model_cv_mean,
    'nested_cv_r2_std': selected_model_cv_std,
    'holdout_r2_after_selection': float(selected_model_holdout),
    'random_seed': RANDOM_SEED,
    'stacking_protocol': STACKING_PROTOCOL_VERSION,
    'uses_raw_morgan_fingerprints': bool(selected_model_name in TREE_BASED_MODELS),
}

nested_cv_df.to_csv(results_dir / 'nested_cv_performance.csv', index=False)
train_results.to_csv(results_dir / 'model_performance_training.csv', index=False)
test_results.to_csv(results_dir / 'model_performance_test.csv', index=False)
with open(results_dir / 'model_selection.json', 'w') as f:
    json.dump(selection_metadata, f, indent=2)

# Preserve the exact development-fitted preprocessing state associated with the
# model selected above. Downstream evaluation, AD, and interpretability scripts
# use this bundle instead of assuming whichever individual model saved transformers last.
selected_transformers = final_transformers_by_model.get(selected_model_name)
if selected_transformers is None:
    print(f"WARNING: no final transformer bundle was retained for {selected_model_name}.")
else:
    selected_transformers = dict(selected_transformers)
    if selected_transformers.get('morgan_pca') is None:
        comparator_transformers = next(
            (
                transformers for transformers in final_transformers_by_model.values()
                if transformers.get('morgan_pca') is not None
            ),
            None,
        )
        if comparator_transformers is not None:
            selected_transformers['linear_comparator_morgan_pca'] = comparator_transformers['morgan_pca']
    joblib.dump(selected_transformers, models_dir / 'selected_model_transformers.pkl')
    selected_transformer_manifest = {
        'selected_model': selected_model_name,
        'random_seed': RANDOM_SEED,
        'uses_raw_morgan_fingerprints': bool(selected_model_name in TREE_BASED_MODELS),
        'stacking_protocol': STACKING_PROTOCOL_VERSION,
    }
    with open(models_dir / 'selected_model_transformers.json', 'w') as f:
        json.dump(selected_transformer_manifest, f, indent=2)

# Save fold-by-fold scores for statistical comparison
with open(results_dir / 'all_fold_scores.json', 'w') as f:
    json.dump(all_fold_scores, f, indent=2)

print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
print(f"\nCompared 11 regression models (9 individual + 2 ensemble)")
print("\nModel selected before holdout evaluation:")
print(f"  {selected_model_name}: nested CV R²={selected_model_cv_mean:.4f}±{selected_model_cv_std:.4f}")
print(f"  Holdout R² (reported after selection)={selected_model_holdout:.4f}")
print("\nTop 5 models by nested CV R²:")
for _, row in nested_cv_df.head(5).iterrows():
    test_row = test_results[test_results['Model'] == row['Model']]
    test_score = test_row['R-Squared'].iloc[0] if len(test_row) else np.nan
    print(f"  {row['Model']}: CV R²={row['Nested_CV_R_Squared_Mean']:.4f}±{row['Nested_CV_R_Squared_Std']:.4f}, Test R²={test_score:.4f}")

# ============================================================
# VISUALIZATION
# ============================================================
plt.figure(figsize=(15, 12))

plt.subplot(2, 2, 1)
cv_data = nested_cv_df.copy()
cv_data = cv_data.rename(columns={
    'Train_R_Squared': 'Training',
    'Test_R_Squared': 'Test',
    'Nested_CV_R_Squared_Mean': 'Cross-Validation'
})
cv_data = pd.melt(cv_data, id_vars=['Model'],
                 value_vars=['Training', 'Test', 'Cross-Validation'],
                 var_name='Dataset', value_name='R-Squared')
sns.barplot(data=cv_data, x='R-Squared', y='Model', hue='Dataset')
plt.title('R-Squared Across Datasets')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
plt.grid(True, linestyle='--', alpha=0.6)

plt.subplot(2, 2, 2)
sns.barplot(data=test_results, x='RMSE', y='Model')
plt.title('RMSE on Test Set')
plt.grid(True, linestyle='--', alpha=0.6)

plt.subplot(2, 2, 3)
sns.barplot(data=test_results, x='MAE', y='Model')
plt.title('MAE on Test Set')
plt.grid(True, linestyle='--', alpha=0.6)

plt.subplot(2, 2, 4)
sns.barplot(data=test_results, x='Time Taken', y='Model')
plt.title('Computation Time (seconds)')
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig(plots_dir / 'model_comparison.png', dpi=300)
plt.savefig(plots_dir / 'model_comparison.pdf')
plt.close()

# Selected-model scatter plots.  This follows the nested-CV rule above rather
# than choosing the highest observed holdout score.
best_model_name = selected_model_name
if 'ensemble' in best_model_name.lower():
    best_model_path = models_dir / f'{best_model_name.lower()}.pkl'
else:
    best_model_path = models_dir / f'{best_model_name.lower()}_tuned.pkl'

best_model_skip_pca = best_model_name in TREE_BASED_MODELS

if os.path.exists(best_model_path):
    with open(best_model_path, 'rb') as f:
        best_model = pickle.load(f)
    # Preprocess data for predictions
    X_dev_proc_final, X_holdout_proc_final, _ = preprocess_features(
        X_dev, X_holdout, morgan_cols, physchem_cols, Y_dev.values,
        skip_pca=best_model_skip_pca,
        random_state=RANDOM_SEED,
    )
    y_train_pred = best_model.predict(X_dev_proc_final)
    y_test_pred = best_model.predict(X_holdout_proc_final)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.regplot(x=Y_dev, y=y_train_pred, scatter_kws={'alpha':0.4})
    plt.xlabel('Experimental pIC50', fontsize='large', fontweight='bold')
    plt.ylabel('Predicted pIC50', fontsize='large', fontweight='bold')
    plt.title(f'{best_model_name} - Training Set')
    plt.plot([Y_dev.min(), Y_dev.max()], [Y_dev.min(), Y_dev.max()], 'k--', lw=2)
    r2 = r2_score(Y_dev, y_train_pred)
    rmse = np.sqrt(mean_squared_error(Y_dev, y_train_pred))
    plt.annotate(f'R² = {r2:.4f}\nRMSE = {rmse:.4f}',
                xy=(0.05, 0.95), xycoords='axes fraction',
                ha='left', va='top', fontsize=12, bbox=dict(boxstyle='round', alpha=0.1))

    plt.subplot(1, 2, 2)
    sns.regplot(x=Y_holdout, y=y_test_pred, scatter_kws={'alpha':0.4})
    plt.xlabel('Experimental pIC50', fontsize='large', fontweight='bold')
    plt.ylabel('Predicted pIC50', fontsize='large', fontweight='bold')
    plt.title(f'{best_model_name} - Test Set')
    plt.plot([Y_holdout.min(), Y_holdout.max()], [Y_holdout.min(), Y_holdout.max()], 'k--', lw=2)
    r2 = r2_score(Y_holdout, y_test_pred)
    rmse = np.sqrt(mean_squared_error(Y_holdout, y_test_pred))
    plt.annotate(f'R² = {r2:.4f}\nRMSE = {rmse:.4f}',
                xy=(0.05, 0.95), xycoords='axes fraction',
                ha='left', va='top', fontsize=12, bbox=dict(boxstyle='round', alpha=0.1))

    plt.tight_layout()
    plt.savefig(plots_dir / 'best_model_predictions.png', dpi=300)
    plt.savefig(plots_dir / 'best_model_predictions.pdf')
    plt.close()

# ============================================================
# Y-RANDOMIZATION VALIDATION
# ============================================================
print("\n" + "="*60)
print("Y-RANDOMIZATION VALIDATION TEST")
print("="*60)

from sklearn.base import clone

def y_randomization_test(model, X_train_raw, y_train, X_test_raw, y_test,
                         morgan_cols, physchem_cols, n_iterations=30,
                         skip_pca=False, random_state=42):
    """Re-runs full preprocessing (incl. MI feature selection) with shuffled y each iteration."""
    rng = np.random.RandomState(random_state)  # seeded so Y-randomization is reproducible
    random_scores_test = []
    for i in range(n_iterations):
        y_train_shuffled = rng.permutation(y_train)
        try:
            X_train_proc, X_test_proc, _ = preprocess_features(
                X_train_raw, X_test_raw, morgan_cols, physchem_cols, y_train_shuffled,
                skip_pca=skip_pca,
                random_state=random_state,
            )
            model_clone = clone(model)
            model_clone.fit(X_train_proc, y_train_shuffled)
            test_pred = model_clone.predict(X_test_proc)
            test_score = r2_score(y_test, test_pred)
            random_scores_test.append(test_score)
        except Exception:
            continue
        if (i + 1) % 10 == 0:
            print(f"  Completed {i+1}/{n_iterations} iterations")
    return random_scores_test

if os.path.exists(best_model_path):
    print(f"\nTesting {best_model_name}...")
    random_test = y_randomization_test(
        best_model, X_dev, Y_dev, X_holdout, Y_holdout,
        morgan_cols, physchem_cols, n_iterations=30,
        skip_pca=best_model_skip_pca,
        random_state=RANDOM_SEED,
    )

    y_rand_test_mean = np.mean(random_test)
    y_rand_test_std = np.std(random_test)
    original_r2 = selected_model_holdout

    print(f"\nY-Randomization Results:")
    print(f"  Original model R² (test): {original_r2:.4f}")
    print(f"  Randomized test R²: {y_rand_test_mean:.4f} ± {y_rand_test_std:.4f}")

    if y_rand_test_mean < 0.1 and original_r2 > 0.4:
        print("  PASSED: Model learns real patterns")
        y_rand_status = "PASSED"
    else:
        print("  WARNING: Check model validity")
        y_rand_status = "WARNING"

    y_rand_results = pd.DataFrame({
        'Metric': ['Original_R2', 'YRand_Test_Mean', 'YRand_Test_Std', 'Status'],
        'Value': [original_r2, y_rand_test_mean, y_rand_test_std, y_rand_status]
    })
    y_rand_results.to_csv(results_dir / 'y_randomization_results.csv', index=False)

# ============================================================
# STATISTICAL COMPARISON OF ALGORITHMS
# ============================================================
print("\n" + "="*60)
print("STATISTICAL COMPARISON OF ALGORITHMS")
print("="*60)

from scipy.stats import wilcoxon

best_model_name_stat = selected_model_name
best_scores = np.array(all_fold_scores.get(best_model_name_stat, []))

stat_comparison_rows = []

if len(best_scores) >= 5:
    print(f"\nWilcoxon signed-rank test: {best_model_name_stat} vs. each model")
    print(f"(Using {len(best_scores)}-fold CV R² scores)")
    print(f"{'Model':30s} {'Mean R²':>10s} {'p-value':>10s} {'Effect Size':>12s} {'Significant':>12s}")
    print("-" * 80)

    for other_name, other_scores_list in all_fold_scores.items():
        if other_name == best_model_name_stat:
            continue
        other_scores = np.array(other_scores_list)
        if len(other_scores) != len(best_scores):
            continue

        diff = best_scores - other_scores
        mean_diff = np.mean(diff)

        try:
            stat_val, p_val = wilcoxon(best_scores, other_scores)
        except ValueError:
            p_val = 1.0
            stat_val = np.nan

        # Corrected resampled paired t-test (Nadeau & Bengio, 2003)
        n_folds = len(best_scores)
        test_ratio = 1.0 / (n_folds - 1)  # approximate test/train ratio for each fold
        var_diff = np.var(diff, ddof=1)
        t_corr = mean_diff / np.sqrt((1.0 / n_folds + test_ratio) * var_diff) if var_diff > 0 else 0.0

        from scipy.stats import t as t_dist
        p_corr = 2 * (1 - t_dist.cdf(abs(t_corr), df=n_folds - 1))

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(best_scores, ddof=1) + np.var(other_scores, ddof=1)) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0

        sig = "Yes" if p_val < 0.05 else "No"
        print(f"  {other_name:28s} {np.mean(other_scores):>10.4f} {p_val:>10.4f} {cohens_d:>12.3f} {sig:>12s}")

        stat_comparison_rows.append({
            'Model': other_name,
            'CV_R2_Mean': float(np.mean(other_scores)),
            'Best_Model_R2_Mean': float(np.mean(best_scores)),
            'Mean_Difference': float(mean_diff),
            'Wilcoxon_Statistic': float(stat_val) if not np.isnan(stat_val) else None,
            'Wilcoxon_p_value': float(p_val),
            'Corrected_t_statistic': float(t_corr),
            'Corrected_p_value': float(p_corr),
            'Cohens_d': float(cohens_d),
            'Significant_p05': sig
        })

    print(f"\nNote: With only {len(best_scores)} folds, statistical power is limited.")
    print("  Wilcoxon signed-rank test has low power at n=5; interpret with caution.")

    stat_df = pd.DataFrame(stat_comparison_rows)
    stat_df.to_csv(results_dir / 'statistical_comparison.csv', index=False)
    print(f"  Saved to {results_dir / 'statistical_comparison.csv'}")
else:
    print("  Insufficient fold scores for statistical comparison.")

# ============================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================
print("\n" + "="*60)
print("BOOTSTRAP CONFIDENCE INTERVALS FOR BEST MODEL")
print("="*60)

if os.path.exists(best_model_path):
    y_test_arr = Y_holdout.values
    y_pred_arr = best_model.predict(X_holdout_proc_final) if 'X_holdout_proc_final' in dir() else best_model.predict(
        preprocess_features(X_dev, X_holdout, morgan_cols, physchem_cols, Y_dev.values,
                            skip_pca=best_model_skip_pca,
                            random_state=RANDOM_SEED)[1]
    )

    n_bootstrap = 1000
    n_test = len(y_test_arr)
    rng = np.random.RandomState(RANDOM_SEED)

    boot_r2 = []
    boot_rmse = []
    boot_mae = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n_test, size=n_test, replace=True)
        if len(np.unique(y_test_arr[idx])) < 2:
            continue
        boot_r2.append(r2_score(y_test_arr[idx], y_pred_arr[idx]))
        boot_rmse.append(np.sqrt(mean_squared_error(y_test_arr[idx], y_pred_arr[idx])))
        boot_mae.append(mean_absolute_error(y_test_arr[idx], y_pred_arr[idx]))

    boot_r2 = np.array(boot_r2)
    boot_rmse = np.array(boot_rmse)
    boot_mae = np.array(boot_mae)

    ci_results = pd.DataFrame({
        'Metric': ['R2', 'RMSE', 'MAE'],
        'Point_Estimate': [
            r2_score(y_test_arr, y_pred_arr),
            np.sqrt(mean_squared_error(y_test_arr, y_pred_arr)),
            mean_absolute_error(y_test_arr, y_pred_arr)
        ],
        'CI_Lower_2.5': [
            np.percentile(boot_r2, 2.5),
            np.percentile(boot_rmse, 2.5),
            np.percentile(boot_mae, 2.5)
        ],
        'CI_Upper_97.5': [
            np.percentile(boot_r2, 97.5),
            np.percentile(boot_rmse, 97.5),
            np.percentile(boot_mae, 97.5)
        ],
        'Bootstrap_Std': [
            np.std(boot_r2),
            np.std(boot_rmse),
            np.std(boot_mae)
        ]
    })

    print(f"\n{best_model_name_stat} - 95% Bootstrap Confidence Intervals ({n_bootstrap} iterations):")
    for _, row in ci_results.iterrows():
        print(f"  {row['Metric']:6s}: {row['Point_Estimate']:.4f}  [{row['CI_Lower_2.5']:.4f}, {row['CI_Upper_97.5']:.4f}]")

    ci_results.to_csv(results_dir / 'confidence_intervals.csv', index=False)
    print(f"  Saved to {results_dir / 'confidence_intervals.csv'}")

# Final summary
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"Target: {target_name} ({target_id})")
print(f"Selected model (nested-CV rule): {selected_model_name}")
print(f"Nested CV R²: {selected_model_cv_mean:.4f} ± {selected_model_cv_std:.4f}")
print(f"Holdout R² after selection: {selected_model_holdout:.4f}")
print(f"\nResults saved to: {results_dir}")
print(f"Models saved to: {models_dir}")
print(f"Plots saved to: {plots_dir}")
print(f"Fold-by-fold results saved to: {fold_results_dir}")
print("\nScript completed successfully!")
