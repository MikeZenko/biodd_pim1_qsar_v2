"""
Step 6: Applicability Domain Analysis

This script analyzes the applicability domain (AD) of trained models to identify
which predictions are reliable. Compounds outside the AD may have unreliable
predictions because they are structurally different from training compounds.

Reference: OECD QSAR validation principles require AD definition for model deployment.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pickle
import joblib
import os
import sys
import argparse
import json
from pathlib import Path

print("Step 6: Applicability Domain Analysis")
print("="*60)

# Parse optional target filters
parser = argparse.ArgumentParser(description="Applicability domain analysis for a specific target.")
parser.add_argument("--target-id", help="ChEMBL target ID to filter (e.g., CHEMBL2147)")
parser.add_argument("--target-name", help="Override target name for labeling outputs")
parser.add_argument("--seed", type=int, default=42,
                    help="Seed used for the scaffold holdout split (default: 42)")
args = parser.parse_args()

# Resolve paths relative to script location
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

# ============================================================
# 1. Load trained model and data
# ============================================================

print("\n1. Loading model and data...")

# Configure output directories (relative to project root)
models_dir = PROJECT_ROOT / 'models'
plots_dir = PROJECT_ROOT / 'plots'
results_dir = PROJECT_ROOT / 'results'
if args.target_id:
    models_dir = models_dir / args.target_id
    plots_dir = plots_dir / args.target_id
    results_dir = results_dir / args.target_id

plots_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)

# Check for required files
required_files = [
    models_dir / 'selected_model_transformers.pkl',
    DATA_DIR / 'pancreatic_cancer_bioactivity_data_3class_pIC50_morgan_fp.csv'
]

missing_files = [f for f in required_files if not os.path.exists(f)]
if missing_files:
    print(f"ERROR: Missing required files: {missing_files}")
    print("Please run scripts 3, 4, and 5 first.")
    sys.exit(1)

# Load the model selected using development-only nested CV.  Older result
# directories without selection metadata retain a clearly labelled fallback.
best_model = None
best_model_name = None
selection_metadata = {}
selection_path = results_dir / 'model_selection.json'
if selection_path.exists():
    with open(selection_path, 'r') as f:
        selection_metadata = json.load(f)
    best_model_name = selection_metadata['selected_model']
    selection_seed = selection_metadata.get('random_seed')
    if selection_seed is not None and selection_seed != args.seed:
        print(f"WARNING: selected model was fit with seed {selection_seed}, but AD uses seed {args.seed}.")
elif (results_dir / 'nested_cv_performance.csv').exists():
    nested_perf = pd.read_csv(results_dir / 'nested_cv_performance.csv')
    best_model_name = nested_perf.sort_values(
        ['Nested_CV_R_Squared_Mean', 'Nested_CV_R_Squared_Std'],
        ascending=[False, True],
    ).iloc[0]['Model']
    print("WARNING: model_selection.json is absent; selecting from nested-CV results.")

if best_model_name is not None:
    if 'ensemble' in best_model_name.lower():
        candidate = models_dir / f'{best_model_name.lower()}.pkl'
    else:
        candidate = models_dir / f'{best_model_name.lower()}_tuned.pkl'
    if candidate.exists():
        with open(candidate, 'rb') as f:
            best_model = pickle.load(f)
        print(f"Loaded nested-CV-selected model: {best_model_name}")

if best_model is None:
    for model_file in ['stacking_ensemble.pkl', 'voting_ensemble.pkl', 'random_forest_model.pkl']:
        candidate = models_dir / model_file
        if os.path.exists(candidate):
            with open(candidate, 'rb') as f:
                best_model = pickle.load(f)
            best_model_name = model_file.replace('.pkl', '')
            print(f"Loaded model from {candidate}")
            break

if best_model is None:
    print("ERROR: No trained model found. Run Step 5 first.")
    sys.exit(1)

# Load the preprocessing state saved for the selected model rather than a generic
# transformer file last written by some other candidate.
selected_transformers = joblib.load(models_dir / 'selected_model_transformers.pkl')
imputer = selected_transformers['imputer']
morgan_transformer = selected_transformers['morgan_pca']
physchem_scaler = selected_transformers['scaler']
selected_mask = selected_transformers['selected_mask']

# Load data
df = pd.read_csv(DATA_DIR / 'pancreatic_cancer_bioactivity_data_3class_pIC50_morgan_fp.csv')
if args.target_id and 'target_chembl_id' in df.columns:
    df = df[df['target_chembl_id'] == args.target_id].copy()
    if df.empty:
        print(f"ERROR: No data found for target {args.target_id}")
        sys.exit(1)

if 'canonical_smiles' in df.columns:
    smiles_list = df['canonical_smiles'].tolist()
else:
    smiles_df = pd.read_csv(DATA_DIR / 'bioactivity_data_preprocessed.csv')
    smiles_list = smiles_df['canonical_smiles'].tolist()
    smiles_list = smiles_list[:len(df)]

print(f"Loaded {len(df)} compounds")

# Prepare features
non_feature_cols = ['target_name', 'target_chembl_id', 'canonical_smiles', 'molecule_chembl_id']
X = df.drop(columns=[col for col in non_feature_cols if col in df.columns], errors='ignore')
X = X.drop('pIC50', axis=1)
Y = df['pIC50']

morgan_cols = [col for col in X.columns if col.startswith('Morgan_')]
physchem_cols = [col for col in X.columns if not col.startswith('Morgan_')]

# ============================================================
# 2. Define Applicability Domain Class
# ============================================================

class ApplicabilityDomain:
    """
    Defines the applicability domain of a QSAR model using k-NN distance approach.
    
    Compounds with average distance to k nearest training neighbors exceeding
    a threshold are considered outside the applicability domain.
    """
    
    def __init__(self, k=5, threshold_percentile=95):
        """
        Parameters:
        -----------
        k : int
            Number of nearest neighbors to consider
        threshold_percentile : float
            Percentile of training distances to use as threshold
        """
        self.k = k
        self.threshold_percentile = threshold_percentile
        self.nn = NearestNeighbors(n_neighbors=k + 1, algorithm='auto')
        self.threshold = None
        self.training_distances = None
        self.reliability_p50 = None
        self.reliability_p90 = None
        
    def fit(self, X_train):
        """Fit the applicability domain model on training data."""
        print(f"  Fitting AD model with k={self.k} neighbors...")
        self.nn.fit(X_train)
        
        # Calculate distances for training set, excluding self (distance=0)
        distances, _ = self.nn.kneighbors(X_train)
        self.training_distances = distances[:, 1:].mean(axis=1)
        
        # Set threshold at specified percentile
        self.threshold = np.percentile(self.training_distances, self.threshold_percentile)
        self.reliability_p50 = np.percentile(self.training_distances, 50)
        self.reliability_p90 = np.percentile(self.training_distances, 90)
        
        print(f"  Training distance range: {self.training_distances.min():.4f} - {self.training_distances.max():.4f}")
        print(f"  AD threshold ({self.threshold_percentile}th percentile): {self.threshold:.4f}")
        
        return self
    
    def predict_in_domain(self, X_test):
        """
        Check if test compounds are within applicability domain.
        
        Returns:
        --------
        in_domain : array of bool
            True if compound is within AD
        distances : array of float
            Mean distance to k nearest training neighbors
        confidence : array of float
            Confidence score (1.0 = very confident, 0.0 = outside AD)
        """
        distances, _ = self.nn.kneighbors(X_test)
        mean_distances = distances[:, :self.k].mean(axis=1)
        
        in_domain = mean_distances < self.threshold
        
        # Calculate confidence score (1 at distance=0, 0 at distance=threshold)
        confidence = np.clip(1 - (mean_distances / self.threshold), 0, 1)
        
        return in_domain, mean_distances, confidence
    
    def get_reliability_category(self, distances):
        """
        Categorize predictions by reliability based on distance to training set.
        
        Categories:
        - High: distance < 50th percentile of training
        - Medium: 50th <= distance < 90th percentile
        - Low: 90th <= distance < threshold
        - Outside: distance >= threshold
        """
        categories = []
        for d in distances:
            if d < self.reliability_p50:
                categories.append('High')
            elif d < self.reliability_p90:
                categories.append('Medium')
            elif d < self.threshold:
                categories.append('Low')
            else:
                categories.append('Outside')
        
        return np.array(categories)


# ============================================================
# 3. Perform Applicability Domain Analysis
# ============================================================

print("\n2. Performing Applicability Domain Analysis...")

# Use scaffold-based split for proper evaluation
from scaffold_utils import scaffold_split

X_train, X_test, Y_train, Y_test, scaffold_groups, train_idx, test_idx = scaffold_split(
    X, Y, smiles_list, test_size=0.2, random_state=args.seed
)

# Process features
print("\n3. Processing features...")

# IMPORTANT: Use transform() not fit_transform() to use the same transformations
# that were applied during model training. Re-fitting would produce different
# PCA components and scaling, leading to inconsistent feature representation.

# Step 1: Impute missing values (using training-fitted imputer)
X_train_imputed = pd.DataFrame(imputer.transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)

# Morgan fingerprints - use the representation selected for the fitted model.
uses_raw_morgan = bool(selection_metadata.get('uses_raw_morgan_fingerprints', False))
if uses_raw_morgan:
    X_train_morgan = X_train_imputed[morgan_cols].values
    X_test_morgan = X_test_imputed[morgan_cols].values
else:
    X_train_morgan = morgan_transformer.transform(X_train_imputed[morgan_cols])
    X_test_morgan = morgan_transformer.transform(X_test_imputed[morgan_cols])

# Physicochemical descriptors - use loaded scaler (already fit during training)
X_train_physchem = physchem_scaler.transform(X_train_imputed[physchem_cols])[:, selected_mask]
X_test_physchem = physchem_scaler.transform(X_test_imputed[physchem_cols])[:, selected_mask]

# Combine features
X_train_processed = np.hstack([X_train_morgan, X_train_physchem])
X_test_processed = np.hstack([X_test_morgan, X_test_physchem])

representation_name = "raw 2048-bit Morgan fingerprints" if uses_raw_morgan else "PCA-reduced Morgan fingerprints"
print(f"Processed features: {X_train_processed.shape[1]} ({representation_name})")

# ============================================================
# 4. Fit Applicability Domain Model
# ============================================================

print("\n4. Fitting Applicability Domain model...")

ad = ApplicabilityDomain(k=5, threshold_percentile=95)
ad.fit(X_train_processed)

# Evaluate test set
in_domain, distances, confidence = ad.predict_in_domain(X_test_processed)
reliability = ad.get_reliability_category(distances)

print(f"\nTest Set AD Analysis:")
print(f"  Total compounds: {len(X_test)}")
print(f"  In-domain: {in_domain.sum()} ({in_domain.mean()*100:.1f}%)")
print(f"  Outside-domain: {(~in_domain).sum()} ({(~in_domain).mean()*100:.1f}%)")

print(f"\nReliability Categories:")
for cat in ['High', 'Medium', 'Low', 'Outside']:
    count = (reliability == cat).sum()
    pct = count / len(reliability) * 100
    print(f"  {cat}: {count} ({pct:.1f}%)")

# ============================================================
# 5. Compare Performance In-Domain vs Out-of-Domain
# ============================================================

print("\n5. Comparing model performance by domain...")

# Make predictions
y_pred = best_model.predict(X_test_processed)

# Overall performance
r2_overall = r2_score(Y_test, y_pred)
rmse_overall = np.sqrt(mean_squared_error(Y_test, y_pred))
mae_overall = mean_absolute_error(Y_test, y_pred)

print(f"\nOverall Performance:")
print(f"  R²: {r2_overall:.4f}")
print(f"  RMSE: {rmse_overall:.4f}")
print(f"  MAE: {mae_overall:.4f}")

# In-domain performance
if in_domain.sum() > 10:
    r2_in = r2_score(Y_test.values[in_domain], y_pred[in_domain])
    rmse_in = np.sqrt(mean_squared_error(Y_test.values[in_domain], y_pred[in_domain]))
    mae_in = mean_absolute_error(Y_test.values[in_domain], y_pred[in_domain])
    
    print(f"\nIn-Domain Performance ({in_domain.sum()} compounds):")
    print(f"  R²: {r2_in:.4f}")
    print(f"  RMSE: {rmse_in:.4f}")
    print(f"  MAE: {mae_in:.4f}")

# Out-of-domain performance
if (~in_domain).sum() > 10:
    r2_out = r2_score(Y_test.values[~in_domain], y_pred[~in_domain])
    rmse_out = np.sqrt(mean_squared_error(Y_test.values[~in_domain], y_pred[~in_domain]))
    mae_out = mean_absolute_error(Y_test.values[~in_domain], y_pred[~in_domain])
    
    print(f"\nOut-of-Domain Performance ({(~in_domain).sum()} compounds):")
    print(f"  R²: {r2_out:.4f}")
    print(f"  RMSE: {rmse_out:.4f}")
    print(f"  MAE: {mae_out:.4f}")

# Performance by reliability category
print("\nPerformance by Reliability Category:")
for cat in ['High', 'Medium', 'Low', 'Outside']:
    mask = reliability == cat
    if mask.sum() > 5:
        r2_cat = r2_score(Y_test.values[mask], y_pred[mask])
        rmse_cat = np.sqrt(mean_squared_error(Y_test.values[mask], y_pred[mask]))
        print(f"  {cat}: R²={r2_cat:.4f}, RMSE={rmse_cat:.4f} (n={mask.sum()})")


def bootstrap_metric_ci(y_true, y_hat, metric, n_bootstrap=1000, random_state=42):
    """Return a percentile CI for a metric, or NaNs when a subset is too small."""
    y_true = np.asarray(y_true)
    y_hat = np.asarray(y_hat)
    if len(y_true) < 3:
        return (np.nan, np.nan)
    rng = np.random.RandomState(random_state)
    values = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        if metric is r2_score and len(np.unique(y_true[idx])) < 2:
            continue
        values.append(metric(y_true[idx], y_hat[idx]))
    if not values:
        return (np.nan, np.nan)
    return (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))


rmse_metric = lambda actual, predicted: np.sqrt(mean_squared_error(actual, predicted))
r2_in_ci = bootstrap_metric_ci(Y_test.values[in_domain], y_pred[in_domain], r2_score, random_state=args.seed)
r2_out_ci = bootstrap_metric_ci(Y_test.values[~in_domain], y_pred[~in_domain], r2_score, random_state=args.seed)
rmse_in_ci = bootstrap_metric_ci(Y_test.values[in_domain], y_pred[in_domain], rmse_metric, random_state=args.seed)
rmse_out_ci = bootstrap_metric_ci(Y_test.values[~in_domain], y_pred[~in_domain], rmse_metric, random_state=args.seed)
print("\nBootstrap 95% CIs by domain (compound-level; interpret the small outside-domain subset cautiously):")
print(f"  In-domain R² [{r2_in_ci[0]:.3f}, {r2_in_ci[1]:.3f}], RMSE [{rmse_in_ci[0]:.3f}, {rmse_in_ci[1]:.3f}]")
print(f"  Outside R² [{r2_out_ci[0]:.3f}, {r2_out_ci[1]:.3f}], RMSE [{rmse_out_ci[0]:.3f}, {rmse_out_ci[1]:.3f}]")

# ============================================================
# 6. Generate Visualizations
# ============================================================

print("\n6. Generating visualizations...")

plots_dir.mkdir(parents=True, exist_ok=True)

# Plot 1: Distance distribution
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(ad.training_distances, bins=50, alpha=0.7, label='Training')
plt.axvline(x=ad.threshold, color='r', linestyle='--', label=f'AD Threshold ({ad.threshold:.3f})')
plt.xlabel('Distance to k-NN')
plt.ylabel('Frequency')
plt.title('Training Set Distance Distribution')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(distances[in_domain], bins=30, alpha=0.7, label='In-domain', color='green')
plt.hist(distances[~in_domain], bins=30, alpha=0.7, label='Outside AD', color='red')
plt.axvline(x=ad.threshold, color='r', linestyle='--', label=f'Threshold')
plt.xlabel('Distance to k-NN')
plt.ylabel('Frequency')
plt.title('Test Set Distance Distribution')
plt.legend()

plt.tight_layout()
plt.savefig(plots_dir / 'applicability_domain_distances.png', dpi=300)
plt.savefig(plots_dir / 'applicability_domain_distances.pdf')
plt.close()

# Plot 2: Prediction accuracy vs confidence
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
errors = np.abs(Y_test.values - y_pred)
plt.scatter(confidence, errors, alpha=0.5, c=in_domain, cmap='RdYlGn')
plt.xlabel('Confidence Score')
plt.ylabel('Absolute Prediction Error')
plt.title('Prediction Error vs. Confidence')
plt.colorbar(label='In Domain')

plt.subplot(1, 2, 2)
colors = {'High': 'green', 'Medium': 'blue', 'Low': 'orange', 'Outside': 'red'}
for cat in ['High', 'Medium', 'Low', 'Outside']:
    mask = reliability == cat
    if mask.sum() > 0:
        plt.scatter(Y_test.values[mask], y_pred[mask], alpha=0.5, 
                   c=colors[cat], label=f'{cat} ({mask.sum()})')

plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=2)
plt.xlabel('Experimental pIC50')
plt.ylabel('Predicted pIC50')
plt.title('Predictions by Reliability Category')
plt.legend()

plt.tight_layout()
plt.savefig(plots_dir / 'applicability_domain_predictions.png', dpi=300)
plt.savefig(plots_dir / 'applicability_domain_predictions.pdf')
plt.close()

# Plot 3: Chemical space visualization (PCA)
plt.figure(figsize=(10, 8))

pca = PCA(n_components=2)
X_all_pca = pca.fit_transform(np.vstack([X_train_processed, X_test_processed]))
X_train_pca = X_all_pca[:len(X_train_processed)]
X_test_pca = X_all_pca[len(X_train_processed):]

plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], alpha=0.3, label='Training', c='gray', s=20)
plt.scatter(X_test_pca[in_domain, 0], X_test_pca[in_domain, 1], 
           alpha=0.7, label='Test (In-domain)', c='green', s=40)
plt.scatter(X_test_pca[~in_domain, 0], X_test_pca[~in_domain, 1], 
           alpha=0.7, label='Test (Outside)', c='red', s=40, marker='x')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.title('Chemical Space Coverage')
plt.legend()
plt.tight_layout()
plt.savefig(plots_dir / 'applicability_domain_chemical_space.png', dpi=300)
plt.savefig(plots_dir / 'applicability_domain_chemical_space.pdf')
plt.close()

print("Plots saved to plots/ directory")

# ============================================================
# 6b. Error Analysis: Error vs Distance & Per-Scaffold Summaries
# ============================================================

print("\n6b. Error analysis...")

errors = np.abs(Y_test.values - y_pred)

# Plot: prediction error vs distance-to-training-set
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].scatter(distances, errors, alpha=0.4, s=20, c='steelblue')
z = np.polyfit(distances, errors, 1)
p = np.poly1d(z)
x_line = np.linspace(distances.min(), distances.max(), 100)
axes[0].plot(x_line, p(x_line), 'r-', linewidth=2, label=f'Trend (slope={z[0]:.3f})')
axes[0].axvline(x=ad.threshold, color='orange', linestyle='--', label=f'AD threshold')
axes[0].set_xlabel('Distance to k-NN (Training Set)')
axes[0].set_ylabel('Absolute Prediction Error (pIC50)')
axes[0].set_title('Prediction Error vs. Distance to Training Set')
axes[0].legend()
axes[0].grid(alpha=0.3)

from scipy.stats import pearsonr
corr, pval = pearsonr(distances, errors)
print(f"  Correlation(distance, |error|): r={corr:.3f}, p={pval:.4f}")

# Error distribution histogram
axes[1].hist(errors, bins=40, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
axes[1].axvline(x=np.median(errors), color='red', linestyle='--', label=f'Median={np.median(errors):.3f}')
axes[1].axvline(x=np.mean(errors), color='orange', linestyle='--', label=f'Mean={np.mean(errors):.3f}')
axes[1].set_xlabel('Absolute Prediction Error (pIC50)')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Prediction Error Distribution')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

# Error by reliability category
cat_order = ['High', 'Medium', 'Low', 'Outside']
cat_errors = {cat: errors[reliability == cat] for cat in cat_order if (reliability == cat).sum() > 0}
positions = range(len(cat_errors))
bp = axes[2].boxplot([cat_errors[c] for c in cat_errors.keys()], tick_labels=list(cat_errors.keys()),
                     patch_artist=True)
cat_colors = {'High': '#4CAF50', 'Medium': '#2196F3', 'Low': '#FF9800', 'Outside': '#F44336'}
for patch, cat in zip(bp['boxes'], cat_errors.keys()):
    patch.set_facecolor(cat_colors.get(cat, 'gray'))
    patch.set_alpha(0.7)
axes[2].set_xlabel('Reliability Category')
axes[2].set_ylabel('Absolute Prediction Error (pIC50)')
axes[2].set_title('Error by Reliability Category')
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(plots_dir / 'error_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig(plots_dir / 'error_analysis.pdf', bbox_inches='tight')
plt.close()

# Per-scaffold error summaries
print("  Computing per-scaffold error summaries...")
test_scaffolds_arr = scaffold_groups[test_idx]
scaffold_error_records = []
unique_scaffolds = np.unique(test_scaffolds_arr)
for sc in unique_scaffolds:
    sc_mask = test_scaffolds_arr == sc
    if sc_mask.sum() == 0:
        continue
    sc_errors = errors[sc_mask]
    sc_distances = distances[sc_mask]
    scaffold_error_records.append({
        'Scaffold_Group': int(sc),
        'N_Compounds': int(sc_mask.sum()),
        'Mean_Error': float(np.mean(sc_errors)),
        'Median_Error': float(np.median(sc_errors)),
        'Std_Error': float(np.std(sc_errors)),
        'Max_Error': float(np.max(sc_errors)),
        'Mean_Distance': float(np.mean(sc_distances)),
        'Mean_pIC50': float(np.mean(Y_test.values[sc_mask])),
    })

scaffold_error_df = pd.DataFrame(scaffold_error_records).sort_values('Mean_Error', ascending=False)
scaffold_error_df.to_csv(results_dir / 'per_scaffold_errors.csv', index=False)

print(f"  {len(unique_scaffolds)} unique scaffolds in test set")
print(f"  Top 5 scaffolds by mean error:")
for _, row in scaffold_error_df.head(5).iterrows():
    print(f"    Scaffold {int(row['Scaffold_Group'])}: n={int(row['N_Compounds'])}, "
          f"mean_error={row['Mean_Error']:.3f}, mean_dist={row['Mean_Distance']:.3f}")

# Error analysis summary CSV
error_summary = pd.DataFrame({
    'Metric': ['Error_Distance_Correlation', 'Error_Distance_p_value',
               'Median_Error', 'Mean_Error', 'Error_Q25', 'Error_Q75',
               'Pct_Error_Below_0.5', 'Pct_Error_Below_1.0'],
    'Value': [float(corr), float(pval),
              float(np.median(errors)), float(np.mean(errors)),
              float(np.percentile(errors, 25)), float(np.percentile(errors, 75)),
              float(np.mean(errors < 0.5) * 100), float(np.mean(errors < 1.0) * 100)]
})
error_summary.to_csv(results_dir / 'error_analysis_summary.csv', index=False)
print("  Error analysis results saved")

# ============================================================
# 7. Save Results
# ============================================================

print("\n7. Saving results...")

results_dir.mkdir(parents=True, exist_ok=True)

# Save AD summary
ad_summary = pd.DataFrame({
    'Metric': [
        'AD_Threshold', 'k_Neighbors', 'Threshold_Percentile',
        'Reliability_High_Upper_Distance_P50', 'Reliability_Medium_Upper_Distance_P90',
        'Feature_Space',
        'Test_In_Domain_Count', 'Test_In_Domain_Percent',
        'Test_Outside_Count', 'Test_Outside_Percent',
        'R2_Overall', 'R2_In_Domain', 'R2_Outside',
        'RMSE_Overall', 'RMSE_In_Domain', 'RMSE_Outside',
        'R2_In_Domain_CI_Lower', 'R2_In_Domain_CI_Upper',
        'R2_Outside_CI_Lower', 'R2_Outside_CI_Upper',
        'RMSE_In_Domain_CI_Lower', 'RMSE_In_Domain_CI_Upper',
        'RMSE_Outside_CI_Lower', 'RMSE_Outside_CI_Upper'
    ],
    'Value': [
        ad.threshold, ad.k, ad.threshold_percentile,
        ad.reliability_p50, ad.reliability_p90, representation_name,
        in_domain.sum(), in_domain.mean() * 100,
        (~in_domain).sum(), (~in_domain).mean() * 100,
        r2_overall, r2_in if in_domain.sum() > 10 else np.nan,
        r2_out if (~in_domain).sum() > 10 else np.nan,
        rmse_overall, rmse_in if in_domain.sum() > 10 else np.nan,
        rmse_out if (~in_domain).sum() > 10 else np.nan,
        r2_in_ci[0], r2_in_ci[1], r2_out_ci[0], r2_out_ci[1],
        rmse_in_ci[0], rmse_in_ci[1], rmse_out_ci[0], rmse_out_ci[1]
    ]
})
ad_summary.to_csv(results_dir / 'applicability_domain_summary.csv', index=False)

# Save per-compound AD results
ad_results = pd.DataFrame({
    'Index': Y_test.index,
    'Experimental_pIC50': Y_test.values,
    'Predicted_pIC50': y_pred,
    'Prediction_Error': np.abs(Y_test.values - y_pred),
    'Distance_to_Training': distances,
    'In_Domain': in_domain,
    'Confidence': confidence,
    'Reliability': reliability,
    'Selected_Model': best_model_name,
    'Seed': args.seed,
    'Feature_Space': representation_name,
})
ad_results.to_csv(results_dir / 'applicability_domain_per_compound.csv', index=False)

print("Results saved to results/ directory")

# ============================================================
# 8. Summary
# ============================================================

print("\n" + "="*60)
print("APPLICABILITY DOMAIN ANALYSIS SUMMARY")
print("="*60)
print(f"AD Method: k-NN distance (k={ad.k})")
print(f"AD Threshold: {ad.threshold:.4f} ({ad.threshold_percentile}th percentile)")
print(f"Reliability bins: High < {ad.reliability_p50:.4f}; Medium < {ad.reliability_p90:.4f}; Low < {ad.threshold:.4f}; Outside ≥ {ad.threshold:.4f}")
print(f"\nTest Set Results:")
print(f"  In-domain compounds: {in_domain.sum()}/{len(in_domain)} ({in_domain.mean()*100:.1f}%)")
print(f"  Overall R²: {r2_overall:.4f}")
if in_domain.sum() > 10:
    print(f"  In-domain R²: {r2_in:.4f} (improvement: {r2_in - r2_overall:+.4f})")
print(f"\nRecommendation:")
print("  For new predictions, check if compound is within AD before trusting the prediction.")
print("  Compounds outside AD should be flagged as 'uncertain predictions'.")

print("\nStep 6 completed!")
