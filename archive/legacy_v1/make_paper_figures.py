"""Generate manuscript figures from a single completed evaluation run.

The script consumes saved result artifacts rather than refitting models. It fails when
the selected model, seed, or reported metrics disagree, preventing stale figure reuse.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_squared_error, r2_score

from scaffold_utils import get_scaffold_groups


PROJECT_ROOT = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_summary(path):
    frame = pd.read_csv(path)
    return dict(zip(frame["Metric"], frame["Value"]))


def require(path):
    if not path.exists():
        raise FileNotFoundError(f"Required manuscript artifact is missing: {path}")
    return path


def save_figure(fig, out_dir, stem):
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def asymmetric_errorbar(values, lower_bounds, upper_bounds, label):
    values = np.asarray(values, dtype=float)
    lower_bounds = np.asarray(lower_bounds, dtype=float)
    upper_bounds = np.asarray(upper_bounds, dtype=float)
    lower = values - lower_bounds
    upper = upper_bounds - values
    if not np.all(np.isfinite(np.concatenate([values, lower_bounds, upper_bounds]))):
        raise ValueError(f"{label} confidence intervals contain non-finite values.")
    if np.any(lower < -1e-10) or np.any(upper < -1e-10):
        raise ValueError(f"{label} confidence intervals do not bracket their point estimates.")
    return np.maximum(np.vstack([lower, upper]), 0)


def main():
    parser = argparse.ArgumentParser(description="Generate reproducible figures for the PIM1 paper.")
    parser.add_argument("--target-id", default="CHEMBL2147")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="figs")
    parser.add_argument("--figures", default="1,2,3,4,5", help="Comma-separated figure numbers")
    args = parser.parse_args()

    results_dir = PROJECT_ROOT / "results" / args.target_id
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    modeling_path = require(PROJECT_ROOT / "data" / "pancreatic_cancer_bioactivity_data_3class_pIC50_morgan_fp.csv")

    selection_path = require(results_dir / "model_selection.json")
    nested_path = require(results_dir / "nested_cv_performance.csv")
    test_path = require(results_dir / "model_performance_test.csv")
    predictions_path = require(results_dir / "holdout_predictions.csv")
    split_path = require(results_dir / "split_manifest.json")
    ad_summary_path = require(results_dir / "applicability_domain_summary.csv")
    ad_points_path = require(results_dir / "applicability_domain_per_compound.csv")
    physchem_path = require(results_dir / "physchem_importance.csv")
    ridge_path = require(results_dir / "ridge_coefficients.csv")
    interpretability_summary_path = require(results_dir / "interpretability_summary.json")

    with selection_path.open() as handle:
        selection = json.load(handle)
    with split_path.open() as handle:
        split_manifest = json.load(handle)
    with interpretability_summary_path.open() as handle:
        interpretability_summary = json.load(handle)
    selected_model = selection["selected_model"]
    if selection.get("random_seed") != args.seed or split_manifest.get("seed") != args.seed:
        raise ValueError("Selected-model and split-manifest seeds must match the requested figure seed.")
    if split_manifest.get("selected_model") != selected_model:
        raise ValueError("Split manifest and model-selection artifact name different selected models.")

    nested = pd.read_csv(nested_path)
    test = pd.read_csv(test_path)
    predictions = pd.read_csv(predictions_path)
    ad_summary = read_summary(ad_summary_path)
    ad_points = pd.read_csv(ad_points_path)
    physchem = pd.read_csv(physchem_path)
    ridge = pd.read_csv(ridge_path)
    modeling = pd.read_csv(modeling_path)
    if "target_chembl_id" in modeling.columns:
        modeling = modeling.loc[modeling["target_chembl_id"] == args.target_id].copy()
    if modeling.empty:
        raise ValueError(f"No modeling rows found for {args.target_id}.")
    if selected_model not in set(nested["Model"]) or selected_model not in set(test["Model"]):
        raise ValueError("Selected model is absent from the nested-CV or holdout result table.")
    if set(predictions["Selected_Model"]) != {selected_model} or set(predictions["Seed"]) != {args.seed}:
        raise ValueError("Holdout-prediction artifact belongs to a different model or seed.")
    if set(ad_points["Selected_Model"]) != {selected_model} or set(ad_points["Seed"]) != {args.seed}:
        raise ValueError("Applicability-domain artifact belongs to a different model or seed.")
    if (
        interpretability_summary.get("best_model") != selected_model
        or interpretability_summary.get("permutation_model") != selected_model
        or interpretability_summary.get("seed") != args.seed
    ):
        raise ValueError("Interpretability artifact belongs to a different model or seed.")

    y_true = predictions["Experimental_pIC50"].to_numpy()
    y_pred = predictions["Predicted_pIC50"].to_numpy()
    reproduced_r2 = r2_score(y_true, y_pred)
    reproduced_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    stored_test = test.loc[test["Model"] == selected_model].iloc[0]
    if not np.isclose(reproduced_r2, stored_test["R-Squared"], atol=1e-8):
        raise ValueError("Holdout-prediction R2 does not reproduce the test-results table.")
    if not np.isclose(reproduced_rmse, stored_test["RMSE"], atol=1e-8):
        raise ValueError("Holdout-prediction RMSE does not reproduce the test-results table.")
    if not np.isclose(float(ad_summary["R2_Overall"]), reproduced_r2, atol=1e-8):
        raise ValueError("Applicability-domain overall R2 does not reproduce selected-model R2.")

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.dpi": 150,
    })
    selected_color = "#21618C"
    neutral = "#B8B8B8"
    figures = {item.strip() for item in args.figures.split(",") if item.strip()}

    if "1" in figures:
        values = modeling["pIC50"].to_numpy(dtype=float)
        scaffold_groups, _ = get_scaffold_groups(modeling["canonical_smiles"].tolist())
        scaffold_sizes = pd.Series(scaffold_groups).value_counts().sort_values(ascending=False).to_numpy()
        singleton_count = int(np.sum(scaffold_sizes == 1))
        bins = np.linspace(values.min(), values.max(), 32)
        x_density = np.linspace(values.min(), values.max(), 250)
        density = gaussian_kde(values)(x_density)
        fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
        axes[0].hist(values, bins=bins, density=True, color="#9FB3C8", edgecolor="white", label="all compounds")
        high_values = values[values > 10]
        if len(high_values):
            high_weights = np.full(len(high_values), 1 / (len(values) * (bins[1] - bins[0])))
            axes[0].hist(high_values, bins=bins, weights=high_weights, color="#F5B041", edgecolor="white", label="pIC$_{50}$ > 10")
        axes[0].plot(x_density, density, color=selected_color, linewidth=2, label="kernel density")
        axes[0].axvline(np.median(values), color="#333333", linestyle="--", linewidth=1.2, label=f"median {np.median(values):.2f}")
        axes[0].set(xlabel="pIC$_{50}$", ylabel="Density", title="a  Activity distribution")
        axes[0].legend(fontsize=8)
        axes[1].plot(np.arange(1, len(scaffold_sizes) + 1), scaffold_sizes, color=selected_color, linewidth=1.8)
        axes[1].scatter(np.arange(1, len(scaffold_sizes) + 1), scaffold_sizes, color=selected_color, s=8, alpha=0.65)
        axes[1].set_yscale("log")
        axes[1].set(xlabel="Scaffold rank", ylabel="Compounds per scaffold (log scale)", title="b  Bemis--Murcko scaffold sizes")
        axes[1].text(0.97, 0.95, f"{len(scaffold_sizes):,} scaffolds\n{singleton_count:,} singletons ({singleton_count / len(scaffold_sizes):.0%})", transform=axes[1].transAxes, ha="right", va="top", bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9})
        fig.suptitle("Frozen PIM1 data set composition", y=1.02, fontweight="bold")
        save_figure(fig, out_dir, "fig1_dataset")

    if "2" in figures:
        combined = nested.merge(test[["Model", "R-Squared", "RMSE", "MAE"]], on="Model", how="left")
        combined = combined.sort_values("Nested_CV_R_Squared_Mean", ascending=True)
        labels = combined["Model"].str.replace("Regressor", "", regex=False)
        colors = [selected_color if name == selected_model else neutral for name in combined["Model"]]
        fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2), sharey=True)
        axes[0].errorbar(
            combined["Nested_CV_R_Squared_Mean"], labels,
            xerr=combined["Nested_CV_R_Squared_Std"], fmt="none", ecolor="#8FA7B5", capsize=3, zorder=1,
        )
        axes[0].scatter(combined["Nested_CV_R_Squared_Mean"], labels, c=colors, s=55, zorder=2)
        axes[0].set_xlabel("Mean outer-fold $R^2$ (±1 SD)")
        axes[0].set_title("a  Prespecified selection statistic")
        axes[0].grid(axis="x", alpha=0.25)
        selected_cv = combined.loc[
            combined["Model"] == selected_model, "Nested_CV_R_Squared_Mean"
        ].iloc[0]
        selected_y = list(combined["Model"]).index(selected_model)
        axes[0].annotate(
            "selected before holdout", xy=(selected_cv, selected_y),
            xytext=(8, 14), textcoords="offset points", color=selected_color, fontsize=9,
        )
        axes[1].barh(labels, combined["RMSE"], color=colors)
        axes[1].set_xlabel("Holdout RMSE (pIC$_{50}$ units)")
        axes[1].set_title("b  Single fixed holdout (descriptive)")
        axes[1].grid(axis="x", alpha=0.25)
        fig.suptitle("Model comparison under scaffold-group-disjoint validation", y=1.02, fontweight="bold")
        save_figure(fig, out_dir, "fig2_model_comparison")

    if "3" in figures:
        xy = np.vstack([y_true, y_pred])
        density = gaussian_kde(xy)(xy)
        order = density.argsort()
        slope, intercept = np.polyfit(y_true, y_pred, 1)
        lower = min(y_true.min(), y_pred.min()) - 0.2
        upper = max(y_true.max(), y_pred.max()) + 0.2
        fig, ax = plt.subplots(figsize=(6.2, 5.8))
        scatter = ax.scatter(y_true[order], y_pred[order], c=density[order], cmap="Blues", s=20, alpha=0.78, edgecolors="none")
        ax.plot([lower, upper], [lower, upper], "--", color="#333333", label="identity")
        x_line = np.array([lower, upper])
        ax.plot(x_line, slope * x_line + intercept, color="#C87525", linewidth=2.1, label=f"OLS fit (slope {slope:.2f})")
        ax.set(xlim=(lower, upper), ylim=(lower, upper), xlabel="Experimental pIC$_{50}$", ylabel="Predicted pIC$_{50}$")
        ax.set_title("Observed holdout predictions for the outer-fold-selected model")
        ax.legend(loc="lower right")
        ax.text(0.04, 0.96, f"$R^2$ = {reproduced_r2:.3f}\nRMSE = {reproduced_rmse:.3f}\n$n$ = {len(predictions)} compounds\n{split_manifest['holdout']['scaffold_groups']} scaffold groups", transform=ax.transAxes, va="top", bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9})
        fig.colorbar(scatter, ax=ax, label="local point density")
        save_figure(fig, out_dir, "fig3_predictions")

    if "4" in figures:
        category_order = ["High", "Medium", "Low", "Outside"]
        category_colors = {"High": "#21618C", "Medium": "#5DADE2", "Low": "#F5B041", "Outside": "#7F8C8D"}
        fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8))
        for category in category_order:
            mask = ad_points["Reliability"] == category
            axes[0].scatter(ad_points.loc[mask, "Distance_to_Training"], ad_points.loc[mask, "Prediction_Error"], s=24, alpha=0.72, color=category_colors[category], label=f"{category} (n={mask.sum()})")
        for value, label, color in (
            (float(ad_summary["Reliability_High_Upper_Distance_P50"]), "$P_{50}$", "#5DADE2"),
            (float(ad_summary["Reliability_Medium_Upper_Distance_P90"]), "$P_{90}$", "#F5B041"),
            (float(ad_summary["AD_Threshold"]), "$P_{95}$", "#7F8C8D"),
        ):
            axes[0].axvline(value, color=color, linestyle="--", linewidth=1.2, label=label)
        axes[0].set(xlabel="Mean distance to 5 development neighbors", ylabel="Absolute prediction error (pIC$_{50}$)", title="a  Error versus distance")
        axes[0].legend(fontsize=7.5, ncol=2)

        domain_labels = ["In-domain", "Outside"]
        r2_values = [float(ad_summary["R2_In_Domain"]), float(ad_summary["R2_Outside"])]
        r2_errors = asymmetric_errorbar(
            r2_values,
            [float(ad_summary["R2_In_Domain_CI_Lower"]), float(ad_summary["R2_Outside_CI_Lower"])],
            [float(ad_summary["R2_In_Domain_CI_Upper"]), float(ad_summary["R2_Outside_CI_Upper"])],
            "Applicability-domain R2",
        )
        axes[1].bar(domain_labels, r2_values, color=[selected_color, "#7F8C8D"], yerr=r2_errors, capsize=4)
        axes[1].set(ylabel="$R^2$", title="b  $R^2$ by applicability domain")
        axes[1].grid(axis="y", alpha=0.25)

        rmse_values = [float(ad_summary["RMSE_In_Domain"]), float(ad_summary["RMSE_Outside"])]
        rmse_errors = asymmetric_errorbar(
            rmse_values,
            [float(ad_summary["RMSE_In_Domain_CI_Lower"]), float(ad_summary["RMSE_Outside_CI_Lower"])],
            [float(ad_summary["RMSE_In_Domain_CI_Upper"]), float(ad_summary["RMSE_Outside_CI_Upper"])],
            "Applicability-domain RMSE",
        )
        axes[2].bar(domain_labels, rmse_values, color=[selected_color, "#7F8C8D"], yerr=rmse_errors, capsize=4)
        axes[2].set(ylabel="RMSE (pIC$_{50}$ units)", title="c  RMSE by applicability domain")
        axes[2].grid(axis="y", alpha=0.25)
        fig.suptitle("Distance-based applicability-domain results", y=1.02, fontweight="bold")
        save_figure(fig, out_dir, "fig4_applicability_domain")

    if "5" in figures:
        physchem = physchem.sort_values("Importance_Mean", ascending=True)
        physchem_features = list(physchem["Feature"])
        ridge_physchem = ridge[ridge["Feature"].isin(physchem_features)].set_index("Feature").reindex(physchem_features).reset_index()
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
        axes[0].barh(physchem["Feature"], physchem["Importance_Mean"], xerr=physchem["Importance_Std"], color="#5DADE2", capsize=3)
        axes[0].axvline(0, color="#333333", linewidth=0.8)
        axes[0].set(xlabel="Permutation importance (decrease in holdout $R^2$)", title="a  Selected-model descriptor associations")
        ridge_colors = ["#21618C" if value >= 0 else "#C87525" for value in ridge_physchem["Coefficient"]]
        axes[1].barh(ridge_physchem["Feature"], ridge_physchem["Coefficient"], color=ridge_colors)
        axes[1].axvline(0, color="#333333", linewidth=0.8)
        axes[1].set(xlabel="Standardized Ridge coefficient", title="b  Separate Ridge comparator")
        fig.suptitle("Feature associations in fitted prediction models", y=1.02, fontweight="bold")
        save_figure(fig, out_dir, "fig5_interpretability")

    manifest = {
        "target_id": args.target_id,
        "seed": args.seed,
        "selected_model": selected_model,
        "selection_rule": selection["selection_rule"],
        "displayed_holdout_metrics": {"R2": float(reproduced_r2), "RMSE": float(reproduced_rmse)},
        "input_sha256": {path.name: sha256(path) for path in [modeling_path, selection_path, nested_path, test_path, predictions_path, split_path, ad_summary_path, ad_points_path, physchem_path, ridge_path, interpretability_summary_path]},
        "generated_figures": sorted(figures),
    }
    with (out_dir / "paper_figure_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(f"Generated figures {', '.join(sorted(figures))} in {out_dir}")


if __name__ == "__main__":
    main()
