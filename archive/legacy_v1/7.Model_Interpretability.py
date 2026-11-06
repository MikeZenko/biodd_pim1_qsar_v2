"""
Step 7: Model Interpretability Analysis

Computes permutation importance for the best model and extracts learned
coefficients from linear models (Ridge, ElasticNet) to identify which
molecular descriptors most influence pIC50 predictions.

PCA components from Morgan fingerprints are reported separately since they
lack direct chemical interpretation. Physicochemical descriptors are
reported with their original names.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib
import os
import sys
import argparse
import json
from pathlib import Path
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score

from scaffold_utils import scaffold_split

print("Step 7: Model Interpretability Analysis")
print("=" * 60)

parser = argparse.ArgumentParser(description="Model interpretability for a specific target.")
parser.add_argument("--target-id", help="ChEMBL target ID to filter (e.g., CHEMBL2147)")
parser.add_argument("--target-name", help="Override target name for labeling outputs")
parser.add_argument("--seed", type=int, default=42,
                    help="Seed used for the scaffold holdout split and permutation test (default: 42)")
args = parser.parse_args()

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

models_dir = PROJECT_ROOT / 'models'
plots_dir = PROJECT_ROOT / 'plots'
results_dir = PROJECT_ROOT / 'results'
if args.target_id:
    models_dir = models_dir / args.target_id
    plots_dir = plots_dir / args.target_id
    results_dir = results_dir / args.target_id

plots_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. Load data, model, and transformers
# ============================================================

print("\n1. Loading data, model, and transformers...")

required_files = [
    models_dir / 'selected_model_transformers.pkl',
    DATA_DIR / 'pancreatic_cancer_bioactivity_data_3class_pIC50_morgan_fp.csv'
]
missing = [f for f in required_files if not f.exists()]
if missing:
    print(f"ERROR: Missing files: {missing}")
    print("Run Steps 3-5 first.")
    sys.exit(1)

selected_transformers = joblib.load(models_dir / 'selected_model_transformers.pkl')
imputer = selected_transformers['imputer']
morgan_pca = selected_transformers['morgan_pca']
physchem_scaler = selected_transformers['scaler']
selected_mask = selected_transformers['selected_mask']

df = pd.read_csv(DATA_DIR / 'pancreatic_cancer_bioactivity_data_3class_pIC50_morgan_fp.csv')
if args.target_id and 'target_chembl_id' in df.columns:
    df = df[df['target_chembl_id'] == args.target_id].copy()
    if df.empty:
        print(f"ERROR: No data for target {args.target_id}")
        sys.exit(1)

target_id = args.target_id or (df['target_chembl_id'].iloc[0] if 'target_chembl_id' in df.columns else 'unknown')
target_name = args.target_name or (df['target_name'].iloc[0] if 'target_name' in df.columns else target_id)

if 'canonical_smiles' in df.columns:
    smiles_list = df['canonical_smiles'].tolist()
else:
    smiles_df = pd.read_csv(DATA_DIR / 'bioactivity_data_preprocessed.csv')
    smiles_list = smiles_df['canonical_smiles'].tolist()[:len(df)]

non_feature_cols = ['target_name', 'target_chembl_id', 'canonical_smiles', 'molecule_chembl_id']
X = df.drop(columns=[c for c in non_feature_cols if c in df.columns], errors='ignore').drop('pIC50', axis=1)
Y = df['pIC50']

morgan_cols = [c for c in X.columns if c.startswith('Morgan_')]
physchem_cols = [c for c in X.columns if not c.startswith('Morgan_')]

print(f"  Dataset: {len(df)} compounds")
print(f"  Morgan columns: {len(morgan_cols)}, Physchem columns: {len(physchem_cols)}")

# Scaffold split
X_train, X_test, Y_train, Y_test, scaffold_groups, train_idx, test_idx = scaffold_split(
    X, Y, smiles_list, test_size=0.2, random_state=args.seed
)

# Determine the representation from the development-only model-selection record.
selection_metadata = {}
selection_path = results_dir / 'model_selection.json'
if selection_path.exists():
    with open(selection_path, 'r') as f:
        selection_metadata = json.load(f)
    selected_model_name = selection_metadata['selected_model']
    selection_seed = selection_metadata.get('random_seed')
    if selection_seed is not None and selection_seed != args.seed:
        print(f"WARNING: selected model was fit with seed {selection_seed}, but interpretability uses seed {args.seed}.")
elif (results_dir / 'nested_cv_performance.csv').exists():
    nested_perf = pd.read_csv(results_dir / 'nested_cv_performance.csv')
    selected_model_name = nested_perf.sort_values(
        ['Nested_CV_R_Squared_Mean', 'Nested_CV_R_Squared_Std'],
        ascending=[False, True],
    ).iloc[0]['Model']
    print("WARNING: model_selection.json is absent; selecting from nested-CV results.")
else:
    selected_model_name = None

uses_raw_morgan = bool(selection_metadata.get('uses_raw_morgan_fingerprints', False))

# Process features using training-fitted transformers.
X_train_imp = pd.DataFrame(imputer.transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)

if uses_raw_morgan:
    X_train_morgan = X_train_imp[morgan_cols].values
    X_test_morgan = X_test_imp[morgan_cols].values
else:
    X_train_morgan = morgan_pca.transform(X_train_imp[morgan_cols])
    X_test_morgan = morgan_pca.transform(X_test_imp[morgan_cols])

X_train_physchem = physchem_scaler.transform(X_train_imp[physchem_cols])[:, selected_mask]
X_test_physchem = physchem_scaler.transform(X_test_imp[physchem_cols])[:, selected_mask]

X_train_proc = np.hstack([X_train_morgan, X_train_physchem])
X_test_proc = np.hstack([X_test_morgan, X_test_physchem])

n_pca = X_train_morgan.shape[1]
n_physchem = X_train_physchem.shape[1]
selected_physchem_names = [physchem_cols[i] for i, m in enumerate(selected_mask) if m]

if uses_raw_morgan:
    morgan_feature_names = morgan_cols
    morgan_feature_type = 'Morgan_Bit'
else:
    morgan_feature_names = [f"PCA_{i+1}" for i in range(n_pca)]
    morgan_feature_type = 'PCA_Morgan'
feature_names = morgan_feature_names + selected_physchem_names

# Ridge and ElasticNet were always trained on the PCA representation, even when
# a raw-Morgan tree model is selected for permutation importance. Keep their
# coefficient names separate from the selected model's feature names.
linear_comparator_pca = selected_transformers.get('linear_comparator_morgan_pca', morgan_pca)
if linear_comparator_pca is None:
    raise ValueError("The selected transformer bundle lacks a PCA state for the Ridge comparator.")
n_pca_comparator = linear_comparator_pca.components_.shape[0]
linear_feature_names = [f"PCA_{i+1}" for i in range(n_pca_comparator)] + selected_physchem_names
linear_feature_types = ['PCA_Morgan'] * n_pca_comparator + ['Physicochemical'] * n_physchem

representation_label = 'raw Morgan bits' if uses_raw_morgan else 'PCA-reduced Morgan features'
print(f"  Processed features: {X_train_proc.shape[1]} ({n_pca} {representation_label} + {n_physchem} physchem)")

# Load the model selected using nested CV rather than the model with the
# highest observed holdout score.
best_model = None
best_model_name = selected_model_name
if best_model_name is not None:
    if 'ensemble' in best_model_name.lower():
        candidate = models_dir / f'{best_model_name.lower()}.pkl'
    else:
        candidate = models_dir / f'{best_model_name.lower()}_tuned.pkl'
    if candidate.exists():
        with open(candidate, 'rb') as f:
            best_model = pickle.load(f)
        print(f"  Loaded nested-CV-selected model: {best_model_name}")

if best_model is None:
    print("ERROR: No trained model found. Run Step 5 first.")
    sys.exit(1)

# ============================================================
# 2. Permutation Importance on Test Set
# ============================================================

print("\n2. Computing permutation importance on test set...")

perm_result = permutation_importance(
    best_model, X_test_proc, Y_test, n_repeats=30,
    random_state=args.seed, scoring='r2', n_jobs=-1
)

perm_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance_Mean': perm_result.importances_mean,
    'Importance_Std': perm_result.importances_std,
    'Feature_Type': [morgan_feature_type] * n_pca + ['Physicochemical'] * n_physchem
}).sort_values('Importance_Mean', ascending=False)

print("\nTop 15 features by permutation importance:")
for _, row in perm_df.head(15).iterrows():
    print(f"  {row['Feature']:25s}  {row['Importance_Mean']:.4f} ± {row['Importance_Std']:.4f}  ({row['Feature_Type']})")

# ============================================================
# 3. Physicochemical Descriptor Importance (Interpretable)
# ============================================================

print("\n3. Physicochemical descriptor importance (directly interpretable)...")

physchem_perm = perm_df[perm_df['Feature_Type'] == 'Physicochemical'].copy()
physchem_perm = physchem_perm.sort_values('Importance_Mean', ascending=False)

print("\nPhysicochemical descriptors ranked by permutation importance:")
for _, row in physchem_perm.iterrows():
    print(f"  {row['Feature']:25s}  {row['Importance_Mean']:.4f} ± {row['Importance_Std']:.4f}")

# ============================================================
# 4. Ridge and ElasticNet Coefficient Analysis
# ============================================================

print("\n4. Linear model coefficient analysis...")

linear_results = {}
for model_name in ['Ridge', 'ElasticNet']:
    model_path = models_dir / f'{model_name.lower()}_tuned.pkl'
    if not model_path.exists():
        continue

    with open(model_path, 'rb') as f:
        linear_model = pickle.load(f)

    coeffs = linear_model.coef_
    if len(coeffs) != len(linear_feature_names):
        raise ValueError(
            f"{model_name} has {len(coeffs)} coefficients, but the PCA comparator "
            f"representation has {len(linear_feature_names)} named features."
        )
    coeff_df = pd.DataFrame({
        'Feature': linear_feature_names,
        'Coefficient': coeffs,
        'Abs_Coefficient': np.abs(coeffs),
        'Feature_Type': linear_feature_types,
    }).sort_values('Abs_Coefficient', ascending=False)

    linear_results[model_name] = coeff_df

    print(f"\n{model_name} - Top 10 features by |coefficient|:")
    for _, row in coeff_df.head(10).iterrows():
        sign = "+" if row['Coefficient'] > 0 else "-"
        print(f"  {row['Feature']:25s}  {sign}{row['Abs_Coefficient']:.4f}  ({row['Feature_Type']})")

    physchem_coeffs = coeff_df[coeff_df['Feature_Type'] == 'Physicochemical'].copy()
    print(f"\n{model_name} - Physicochemical descriptor coefficients:")
    for _, row in physchem_coeffs.iterrows():
        direction = "(+)" if row['Coefficient'] > 0 else "(-)"
        print(f"  {row['Feature']:25s}  {row['Coefficient']:+.4f}  {direction}")

# ============================================================
# 5. PCA Component Variance Explained
# ============================================================

print("\n5. PCA component analysis...")

if uses_raw_morgan:
    var_ratios = np.array([])
    cumulative = np.array([])
    print("  PCA was not used for the selected model; raw Morgan-bit importance is reported instead.")
else:
    var_ratios = morgan_pca.explained_variance_ratio_
    cumulative = np.cumsum(var_ratios)
    print(f"  Total PCA components retained: {len(var_ratios)}")
    print(f"  Variance explained by top 5 PCs: {cumulative[min(4, len(cumulative)-1)]*100:.1f}%")
    print(f"  Variance explained by top 10 PCs: {cumulative[min(9, len(cumulative)-1)]*100:.1f}%")
    print(f"  Total variance explained: {cumulative[-1]*100:.1f}%")

# ============================================================
# 6. Generate Plots
# ============================================================

print("\n6. Generating plots...")

# Plot 1: Top 20 features by permutation importance
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

top_n = min(20, len(perm_df))
top_perm = perm_df.head(top_n).iloc[::-1]
feature_colors = {
    'PCA_Morgan': '#2196F3',
    'Morgan_Bit': '#607D8B',
    'Physicochemical': '#FF9800',
}
colors = [feature_colors[t] for t in top_perm['Feature_Type']]

axes[0].barh(range(top_n), top_perm['Importance_Mean'], xerr=top_perm['Importance_Std'],
             color=colors, alpha=0.8, capsize=3)
axes[0].set_yticks(range(top_n))
axes[0].set_yticklabels(top_perm['Feature'])
axes[0].set_xlabel('Permutation Importance (decrease in R²)')
axes[0].set_title(f'Top {top_n} Features - {best_model_name}')
axes[0].axvline(x=0, color='black', linewidth=0.5)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#2196F3', label='PCA (Morgan FP)'),
                   Patch(facecolor='#607D8B', label='Morgan bit'),
                   Patch(facecolor='#FF9800', label='Physicochemical')]
axes[0].legend(handles=legend_elements, loc='lower right')

# Plot 2: Physicochemical descriptors only
physchem_plot = physchem_perm.iloc[::-1]
axes[1].barh(range(len(physchem_plot)), physchem_plot['Importance_Mean'],
             xerr=physchem_plot['Importance_Std'], color='#FF9800', alpha=0.8, capsize=3)
axes[1].set_yticks(range(len(physchem_plot)))
axes[1].set_yticklabels(physchem_plot['Feature'])
axes[1].set_xlabel('Permutation Importance (decrease in R²)')
axes[1].set_title('Physicochemical Descriptors Only')
axes[1].axvline(x=0, color='black', linewidth=0.5)

plt.tight_layout()
plt.savefig(plots_dir / 'feature_importance_permutation.png', dpi=300, bbox_inches='tight')
plt.savefig(plots_dir / 'feature_importance_permutation.pdf', bbox_inches='tight')
plt.close()

# Plot 3: Ridge coefficient plot for physicochemical descriptors
if 'Ridge' in linear_results:
    ridge_physchem = linear_results['Ridge'][linear_results['Ridge']['Feature_Type'] == 'Physicochemical'].copy()
    ridge_physchem = ridge_physchem.sort_values('Coefficient')

    fig, ax = plt.subplots(figsize=(10, 7))
    colors_ridge = ['#4CAF50' if c > 0 else '#F44336' for c in ridge_physchem['Coefficient']]
    ax.barh(range(len(ridge_physchem)), ridge_physchem['Coefficient'],
            color=colors_ridge, alpha=0.8)
    ax.set_yticks(range(len(ridge_physchem)))
    ax.set_yticklabels(ridge_physchem['Feature'])
    ax.set_xlabel('Ridge Coefficient (standardized features)')
    ax.set_title('Ridge Regression Coefficients - Physicochemical Descriptors')
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)

    legend_elements_r = [Patch(facecolor='#4CAF50', label='Positive (increases pIC50)'),
                         Patch(facecolor='#F44336', label='Negative (decreases pIC50)')]
    ax.legend(handles=legend_elements_r, loc='lower right')

    plt.tight_layout()
    plt.savefig(plots_dir / 'ridge_coefficients_physchem.png', dpi=300, bbox_inches='tight')
    plt.savefig(plots_dir / 'ridge_coefficients_physchem.pdf', bbox_inches='tight')
    plt.close()

# Plot 4: PCA variance explained, when PCA is part of the selected model.
if not uses_raw_morgan:
    fig, ax = plt.subplots(figsize=(10, 5))
    n_show = min(30, len(var_ratios))
    ax.bar(range(1, n_show + 1), var_ratios[:n_show] * 100, alpha=0.7, label='Individual')
    ax.plot(range(1, n_show + 1), cumulative[:n_show] * 100, 'r-o', markersize=4, label='Cumulative')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance Explained (%)')
    ax.set_title('PCA Components from Morgan Fingerprints')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / 'pca_variance_explained.png', dpi=300, bbox_inches='tight')
    plt.savefig(plots_dir / 'pca_variance_explained.pdf', bbox_inches='tight')
    plt.close()

print("  Plots saved to plots directory")

# ============================================================
# 7. Save Results
# ============================================================

print("\n7. Saving results...")

perm_df.to_csv(results_dir / 'feature_importance.csv', index=False)

physchem_perm.to_csv(results_dir / 'physchem_importance.csv', index=False)

for model_name, coeff_df in linear_results.items():
    coeff_df.to_csv(results_dir / f'{model_name.lower()}_coefficients.csv', index=False)

if not uses_raw_morgan:
    pca_info = pd.DataFrame({
        'Component': [f'PCA_{i+1}' for i in range(len(var_ratios))],
        'Variance_Explained': var_ratios,
        'Cumulative_Variance': cumulative
    })
    pca_info.to_csv(results_dir / 'pca_component_info.csv', index=False)

summary = {
    'best_model': best_model_name,
    'selection_rule': selection_metadata.get('selection_rule', 'highest mean outer-fold nested-CV R2'),
    'seed': args.seed,
    'permutation_model': best_model_name,
    'ridge_coefficients_are_auxiliary_comparator': True,
    'n_features_total': len(feature_names),
    'n_pca_components': n_pca,
    'n_pca_components_for_linear_comparator': n_pca_comparator,
    'n_physchem_selected': n_physchem,
    'physchem_names': selected_physchem_names,
    'top5_features': perm_df.head(5)['Feature'].tolist(),
    'top5_importance': perm_df.head(5)['Importance_Mean'].tolist(),
    'top_physchem': physchem_perm.head(5)['Feature'].tolist() if len(physchem_perm) >= 5 else physchem_perm['Feature'].tolist(),
    'morgan_representation': representation_label,
    'total_pca_variance_explained': float(cumulative[-1]) if len(cumulative) else None
}
with open(results_dir / 'interpretability_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("  Results saved to results directory")

# ============================================================
# 8. Summary
# ============================================================

print("\n" + "=" * 60)
print("MODEL INTERPRETABILITY SUMMARY")
print("=" * 60)
print(f"Target: {target_name} ({target_id})")
print(f"Model: {best_model_name}")
print(f"\nTop 5 most important features (permutation importance):")
for _, row in perm_df.head(5).iterrows():
    print(f"  {row['Feature']:25s}  {row['Importance_Mean']:.4f}")
print(f"\nTop physicochemical descriptors:")
for _, row in physchem_perm.head(5).iterrows():
    print(f"  {row['Feature']:25s}  {row['Importance_Mean']:.4f}")
if 'Ridge' in linear_results:
    ridge_top = linear_results['Ridge'].head(3)
    print(f"\nRidge top coefficients:")
    for _, row in ridge_top.iterrows():
        print(f"  {row['Feature']:25s}  {row['Coefficient']:+.4f}")

print(f"\nOutput files:")
print(f"  {results_dir / 'feature_importance.csv'}")
print(f"  {results_dir / 'physchem_importance.csv'}")
print(f"  {plots_dir / 'feature_importance_permutation.png'}")
if 'Ridge' in linear_results:
    print(f"  {plots_dir / 'ridge_coefficients_physchem.png'}")

print("\nStep 7 completed!")
