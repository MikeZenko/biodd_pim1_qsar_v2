"""Recreate and verify the selected model's development-fitted preprocessing state.

This utility refuses to write a transformer bundle unless replayed predictions reproduce
the selected-model holdout metrics saved by Step 5.  It is used after an evaluation run
and before downstream manuscript artifacts are created.
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

from preprocessing_utils import preprocess_features
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
    parser = argparse.ArgumentParser(description="Validate and save selected-model preprocessing state.")
    parser.add_argument("--target-id", default="CHEMBL2147")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    models_dir = PROJECT_ROOT / "models" / args.target_id
    results_dir = PROJECT_ROOT / "results" / args.target_id
    with (results_dir / "model_selection.json").open(encoding="utf-8") as handle:
        selection = json.load(handle)
    if selection.get("random_seed") != args.seed:
        raise ValueError("Requested seed does not match the selected-model record.")
    model_name = selection["selected_model"]
    uses_raw = bool(selection.get("uses_raw_morgan_fingerprints", False))

    df = pd.read_csv(DATA_PATH)
    if "target_chembl_id" in df.columns:
        df = df.loc[df["target_chembl_id"] == args.target_id].copy()
    non_feature_cols = ["target_name", "target_chembl_id", "canonical_smiles", "molecule_chembl_id"]
    X = df.drop(columns=[column for column in non_feature_cols if column in df.columns], errors="ignore").drop("pIC50", axis=1)
    y = df["pIC50"]
    X_dev, X_holdout, y_dev, y_holdout, _, _, _ = scaffold_split(
        X, y, df["canonical_smiles"].tolist(), test_size=0.2, random_state=args.seed
    )
    morgan_cols = [column for column in X.columns if column.startswith("Morgan_")]
    physchem_cols = [column for column in X.columns if not column.startswith("Morgan_")]
    X_dev_processed, X_holdout_processed, transformers = preprocess_features(
        X_dev,
        X_holdout,
        morgan_cols,
        physchem_cols,
        y_dev.values,
        skip_pca=uses_raw,
        random_state=args.seed,
    )

    with model_path(models_dir, model_name).open("rb") as handle:
        model = pickle.load(handle)
    predictions = model.predict(X_holdout_processed)
    replayed = {
        "R-Squared": float(r2_score(y_holdout, predictions)),
        "RMSE": float(np.sqrt(mean_squared_error(y_holdout, predictions))),
        "MAE": float(mean_absolute_error(y_holdout, predictions)),
    }
    expected_rows = pd.read_csv(results_dir / "model_performance_test.csv")
    expected = expected_rows.loc[expected_rows["Model"] == model_name]
    if len(expected) != 1:
        raise ValueError("Expected one selected-model holdout row.")
    expected = expected.iloc[0]
    for metric, value in replayed.items():
        if not np.isclose(value, float(expected[metric]), atol=1e-8, rtol=0):
            raise RuntimeError(
                f"Recreated preprocessing gives {metric}={value:.8f}, but Step 5 recorded "
                f"{float(expected[metric]):.8f}. Refusing to write a mismatched transformer bundle."
            )

    _, replayed_holdout, _ = preprocess_features(
        X_dev,
        X_holdout,
        morgan_cols,
        physchem_cols,
        y_dev.values,
        fit_new=False,
        transformers=transformers,
        skip_pca=uses_raw,
        random_state=args.seed,
    )
    if not np.allclose(X_holdout_processed, replayed_holdout, atol=1e-12, rtol=0):
        raise RuntimeError("Saved-transformer replay differs from the fitted preprocessing output.")

    bundle_path = models_dir / "selected_model_transformers.pkl"
    joblib.dump(transformers, bundle_path)
    manifest = {
        "target_id": args.target_id,
        "seed": args.seed,
        "selected_model": model_name,
        "uses_raw_morgan_fingerprints": uses_raw,
        "feature_count": int(X_dev_processed.shape[1]),
        "replayed_holdout_metrics": replayed,
        "data_sha256": sha256(DATA_PATH),
        "model_sha256": sha256(model_path(models_dir, model_name)),
    }
    with (models_dir / "selected_model_transformers.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(f"Wrote {bundle_path}")
    print(f"Wrote {models_dir / 'selected_model_transformers.json'}")


if __name__ == "__main__":
    main()
