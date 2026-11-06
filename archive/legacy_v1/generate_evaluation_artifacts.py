"""Materialize reproducible holdout predictions and split metadata for the manuscript.

This script only applies saved transformers and the model chosen from nested CV.  It does
not refit a model or preprocessing component, which makes the generated figure inputs
traceable to one completed evaluation run.
"""

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from scaffold_utils import scaffold_split


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "pancreatic_cancer_bioactivity_data_3class_pIC50_morgan_fp.csv"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_path(models_dir, model_name):
    suffix = f"{model_name.lower()}.pkl" if "ensemble" in model_name.lower() else f"{model_name.lower()}_tuned.pkl"
    return models_dir / suffix


def main():
    parser = argparse.ArgumentParser(description="Write holdout prediction and split artifacts from a saved run.")
    parser.add_argument("--target-id", default="CHEMBL2147")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    models_dir = PROJECT_ROOT / "models" / args.target_id
    results_dir = PROJECT_ROOT / "results" / args.target_id
    selection_path = results_dir / "model_selection.json"
    test_metrics_path = results_dir / "model_performance_test.csv"
    if not selection_path.exists() or not test_metrics_path.exists():
        raise FileNotFoundError("Run 5.Comparing_Regressors.py before generating evaluation artifacts.")

    with selection_path.open() as handle:
        selection = json.load(handle)
    if selection.get("random_seed") != args.seed:
        raise ValueError(
            f"Requested seed {args.seed} does not match the selected-model seed {selection.get('random_seed')}"
        )
    model_name = selection["selected_model"]
    selected_model_path = model_path(models_dir, model_name)
    if not selected_model_path.exists():
        raise FileNotFoundError(f"Selected model is missing: {selected_model_path}")

    df = pd.read_csv(DATA_PATH)
    if "target_chembl_id" in df.columns:
        df = df[df["target_chembl_id"] == args.target_id].copy()
    if df.empty:
        raise ValueError(f"No rows found for {args.target_id}")

    non_feature_cols = ["target_name", "target_chembl_id", "canonical_smiles", "molecule_chembl_id"]
    X = df.drop(columns=[col for col in non_feature_cols if col in df.columns], errors="ignore").drop("pIC50", axis=1)
    y = df["pIC50"]
    smiles = df["canonical_smiles"].tolist()
    X_dev, X_holdout, y_dev, y_holdout, groups, dev_idx, holdout_idx = scaffold_split(
        X, y, smiles, test_size=0.2, random_state=args.seed
    )

    transformer_bundle_path = models_dir / "selected_model_transformers.pkl"
    if not transformer_bundle_path.exists():
        raise FileNotFoundError(
            "Selected-model preprocessing bundle is missing. Run reconcile_selected_model.py "
            "after Step 5 before generating manuscript artifacts."
        )
    transformers = joblib.load(transformer_bundle_path)
    imputer = transformers["imputer"]
    scaler = transformers["scaler"]
    selected_mask = transformers["selected_mask"]
    morgan_cols = [col for col in X.columns if col.startswith("Morgan_")]
    physchem_cols = [col for col in X.columns if not col.startswith("Morgan_")]
    uses_raw = bool(selection.get("uses_raw_morgan_fingerprints", False))

    x_dev_imputed = pd.DataFrame(imputer.transform(X_dev), columns=X.columns, index=X_dev.index)
    x_holdout_imputed = pd.DataFrame(imputer.transform(X_holdout), columns=X.columns, index=X_holdout.index)
    if uses_raw:
        x_dev_morgan = x_dev_imputed[morgan_cols].values
        x_holdout_morgan = x_holdout_imputed[morgan_cols].values
        representation = "raw 2048-bit Morgan fingerprints plus standardized selected physicochemical descriptors"
    else:
        pca = transformers["morgan_pca"]
        if pca is None:
            raise ValueError("Selected model expects PCA-reduced Morgan features, but its bundle has no PCA transformer.")
        x_dev_morgan = pca.transform(x_dev_imputed[morgan_cols])
        x_holdout_morgan = pca.transform(x_holdout_imputed[morgan_cols])
        representation = "PCA-reduced Morgan fingerprints plus standardized selected physicochemical descriptors"
    x_dev_physchem = scaler.transform(x_dev_imputed[physchem_cols])[:, selected_mask]
    x_holdout_physchem = scaler.transform(x_holdout_imputed[physchem_cols])[:, selected_mask]
    x_dev_processed = np.hstack([x_dev_morgan, x_dev_physchem])
    x_holdout_processed = np.hstack([x_holdout_morgan, x_holdout_physchem])

    with selected_model_path.open("rb") as handle:
        model = pickle.load(handle)
    predictions = model.predict(x_holdout_processed)
    observed = y_holdout.values
    r2 = float(r2_score(observed, predictions))
    rmse = float(np.sqrt(mean_squared_error(observed, predictions)))
    mae = float(mean_absolute_error(observed, predictions))

    metrics = pd.read_csv(test_metrics_path)
    selected_metrics = metrics.loc[metrics["Model"] == model_name]
    if len(selected_metrics) != 1:
        raise ValueError(f"Could not find one test-metric row for {model_name}")
    expected = selected_metrics.iloc[0]
    for name, observed_value, expected_value in (
        ("R2", r2, float(expected["R-Squared"])),
        ("RMSE", rmse, float(expected["RMSE"])),
        ("MAE", mae, float(expected["MAE"])),
    ):
        if not np.isclose(observed_value, expected_value, atol=1e-8):
            raise RuntimeError(
                f"Replayed {name} ({observed_value:.8f}) does not match stored evaluation ({expected_value:.8f})."
            )

    holdout_rows = df.iloc[holdout_idx].copy()
    holdout_rows = holdout_rows[[col for col in ["molecule_chembl_id", "canonical_smiles"] if col in holdout_rows.columns]]
    holdout_rows["Scaffold_Group"] = groups[holdout_idx]
    holdout_rows["Experimental_pIC50"] = observed
    holdout_rows["Predicted_pIC50"] = predictions
    holdout_rows["Residual"] = predictions - observed
    holdout_rows["Absolute_Error"] = np.abs(predictions - observed)
    holdout_rows["Selected_Model"] = model_name
    holdout_rows["Seed"] = args.seed
    holdout_rows.to_csv(results_dir / "holdout_predictions.csv", index=False)

    def partition_summary(values, partition_indices):
        values = np.asarray(values)
        return {
            "compounds": int(len(values)),
            "scaffold_groups": int(len(np.unique(groups[partition_indices]))),
            "median_pIC50": float(np.median(values)),
            "minimum_pIC50": float(np.min(values)),
            "maximum_pIC50": float(np.max(values)),
            "pIC50_gt_10": int(np.sum(values > 10)),
        }

    manifest = {
        "target_id": args.target_id,
        "seed": args.seed,
        "selected_model": model_name,
        "selection_rule": selection["selection_rule"],
        "model_representation": representation,
        "data_sha256": sha256(DATA_PATH),
        "selected_model_sha256": sha256(selected_model_path),
        "selected_model_transformers_sha256": sha256(transformer_bundle_path),
        "total_compounds": int(len(df)),
        "total_scaffold_groups": int(len(np.unique(groups))),
        "development": partition_summary(y_dev.values, dev_idx),
        "holdout": partition_summary(y_holdout.values, holdout_idx),
        "holdout_metrics": {"R2": r2, "RMSE": rmse, "MAE": mae},
    }
    with (results_dir / "split_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(f"Wrote {results_dir / 'holdout_predictions.csv'}")
    print(f"Wrote {results_dir / 'split_manifest.json'}")


if __name__ == "__main__":
    main()
