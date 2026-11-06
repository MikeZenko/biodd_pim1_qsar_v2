"""Write LaTeX-ready manuscript metrics and table rows from one completed run.

The manuscript imports these generated files rather than maintaining a second,
hand-edited copy of model-comparison and statistical-test values.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent


def tex_escape(value):
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
    }
    text = str(value)
    for source, replacement in replacements.items():
        text = text.replace(source, replacement)
    return text


def display_model_name(name):
    return str(name).replace("Regressor", "")


def numeric_mapping(path):
    frame = pd.read_csv(path)
    values = {}
    for _, row in frame.iterrows():
        try:
            values[row["Metric"]] = float(row["Value"])
        except (TypeError, ValueError):
            continue
    return values


def fixed(value, places=3):
    return f"{float(value):.{places}f}"


def write(path, content):
    path.write_text(content, encoding="utf-8")
    print(f"Wrote {path}")


def require_close(label, observed, expected, tolerance=1e-8):
    if not np.isclose(float(observed), float(expected), atol=tolerance, rtol=0):
        raise ValueError(f"{label} ({observed}) does not match the selected-run value ({expected}).")


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX manuscript artifacts from saved result files.")
    parser.add_argument("--target-id", default="CHEMBL2147")
    args = parser.parse_args()

    results_dir = PROJECT_ROOT / "results" / args.target_id
    required = {
        "selection": results_dir / "model_selection.json",
        "nested": results_dir / "nested_cv_performance.csv",
        "holdout": results_dir / "model_performance_test.csv",
        "bootstrap": results_dir / "confidence_intervals.csv",
        "yrandom": results_dir / "y_randomization_results.csv",
        "statistics": results_dir / "statistical_comparison.csv",
        "ad": results_dir / "applicability_domain_summary.csv",
        "ad_points": results_dir / "applicability_domain_per_compound.csv",
        "interpretability": results_dir / "interpretability_summary.json",
        "split": results_dir / "split_manifest.json",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing completed-evaluation artifacts:\n" + "\n".join(missing))

    with required["selection"].open(encoding="utf-8") as handle:
        selection = json.load(handle)
    with required["split"].open(encoding="utf-8") as handle:
        split = json.load(handle)

    selected_model = selection["selected_model"]
    nested = pd.read_csv(required["nested"])
    holdout = pd.read_csv(required["holdout"])
    selected_nested = nested.loc[nested["Model"] == selected_model]
    selected_holdout = holdout.loc[holdout["Model"] == selected_model]
    if len(selected_nested) != 1 or len(selected_holdout) != 1:
        raise ValueError("Selected model must occur exactly once in both performance tables.")
    selected_nested = selected_nested.iloc[0]
    selected_holdout = selected_holdout.iloc[0]

    bootstrap = pd.read_csv(required["bootstrap"]).set_index("Metric")
    yrandom = numeric_mapping(required["yrandom"])
    ad = numeric_mapping(required["ad"])
    ad_points = pd.read_csv(required["ad_points"])
    with required["interpretability"].open(encoding="utf-8") as handle:
        interpretability = json.load(handle)
    if selection.get("random_seed") != split.get("seed"):
        raise ValueError("Model-selection and split-manifest seed values do not match.")
    if split.get("selected_model") != selected_model:
        raise ValueError("Model-selection and split-manifest model names do not match.")

    for metric, expected in {
        "R2": selected_holdout["R-Squared"],
        "RMSE": selected_holdout["RMSE"],
        "MAE": selected_holdout["MAE"],
    }.items():
        if metric not in bootstrap.index:
            raise ValueError(f"Bootstrap result for {metric} is missing.")
        require_close(f"Bootstrap {metric} point estimate", bootstrap.loc[metric, "Point_Estimate"], expected)
    require_close("Y-randomization original R2", yrandom["Original_R2"], selected_holdout["R-Squared"])
    require_close("AD overall R2", ad["R2_Overall"], selected_holdout["R-Squared"])
    require_close("AD overall RMSE", ad["RMSE_Overall"], selected_holdout["RMSE"])

    statistics = pd.read_csv(required["statistics"])
    if statistics.empty or "Best_Model_R2_Mean" not in statistics:
        raise ValueError("Statistical-comparison table is missing selected-model fold scores.")
    if not np.allclose(
        statistics["Best_Model_R2_Mean"].astype(float),
        float(selected_nested["Nested_CV_R_Squared_Mean"]),
        atol=1e-8,
        rtol=0,
    ):
        raise ValueError("Statistical-comparison table belongs to a different selected-model run.")
    if set(ad_points["Selected_Model"]) != {selected_model} or set(ad_points["Seed"]) != {selection["random_seed"]}:
        raise ValueError("Applicability-domain per-compound results belong to a different model or seed.")
    if interpretability.get("best_model") != selected_model or interpretability.get("seed") != selection["random_seed"]:
        raise ValueError("Interpretability summary belongs to a different selected-model run.")

    macros = {
        "SelectedModel": tex_escape(display_model_name(selected_model)),
        "SelectedCVRtwo": fixed(selected_nested["Nested_CV_R_Squared_Mean"]),
        "SelectedCVRtwoSD": fixed(selected_nested["Nested_CV_R_Squared_Std"]),
        "SelectedHoldoutRtwo": fixed(selected_holdout["R-Squared"]),
        "SelectedHoldoutRMSE": fixed(selected_holdout["RMSE"]),
        "SelectedHoldoutMAE": fixed(selected_holdout["MAE"]),
        "BootstrapRtwoLower": fixed(bootstrap.loc["R2", "CI_Lower_2.5"]),
        "BootstrapRtwoUpper": fixed(bootstrap.loc["R2", "CI_Upper_97.5"]),
        "BootstrapRMSELower": fixed(bootstrap.loc["RMSE", "CI_Lower_2.5"]),
        "BootstrapRMSEUpper": fixed(bootstrap.loc["RMSE", "CI_Upper_97.5"]),
        "BootstrapMAELower": fixed(bootstrap.loc["MAE", "CI_Lower_2.5"]),
        "BootstrapMAEUpper": fixed(bootstrap.loc["MAE", "CI_Upper_97.5"]),
        "YRandomMean": fixed(yrandom["YRand_Test_Mean"]),
        "YRandomSD": fixed(yrandom["YRand_Test_Std"]),
        "HoldoutCompounds": str(split["holdout"]["compounds"]),
        "HoldoutScaffoldGroups": str(split["holdout"]["scaffold_groups"]),
        "InDomainCount": str(int(ad["Test_In_Domain_Count"])),
        "InDomainPercent": fixed(ad["Test_In_Domain_Percent"], 1),
        "OutsideDomainCount": str(int(ad["Test_Outside_Count"])),
        "ADRtwoIn": fixed(ad["R2_In_Domain"]),
        "ADRtwoOut": fixed(ad["R2_Outside"]),
        "ADRMSEIn": fixed(ad["RMSE_In_Domain"]),
        "ADRMSEOut": fixed(ad["RMSE_Outside"]),
        "ADRtwoInLower": fixed(ad["R2_In_Domain_CI_Lower"]),
        "ADRtwoInUpper": fixed(ad["R2_In_Domain_CI_Upper"]),
        "ADRtwoOutLower": fixed(ad["R2_Outside_CI_Lower"]),
        "ADRtwoOutUpper": fixed(ad["R2_Outside_CI_Upper"]),
        "ADRMSEInLower": fixed(ad["RMSE_In_Domain_CI_Lower"]),
        "ADRMSEInUpper": fixed(ad["RMSE_In_Domain_CI_Upper"]),
        "ADRMSEOutLower": fixed(ad["RMSE_Outside_CI_Lower"]),
        "ADRMSEOutUpper": fixed(ad["RMSE_Outside_CI_Upper"]),
        "ADMedianDistance": fixed(ad["Reliability_High_Upper_Distance_P50"], 2),
        "ADNinetiethDistance": fixed(ad["Reliability_Medium_Upper_Distance_P90"], 2),
        "ADNinetyFifthDistance": fixed(ad["AD_Threshold"], 2),
    }
    metrics_tex = "% Generated by generate_manuscript_artifacts.py; do not edit by hand.\n"
    metrics_tex += "\n".join(f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macros.items()) + "\n"
    write(results_dir / "manuscript_metrics.tex", metrics_tex)

    combined = nested.merge(holdout[["Model", "R-Squared", "RMSE", "MAE"]], on="Model", how="inner")
    combined = combined.sort_values("Nested_CV_R_Squared_Mean", ascending=False)
    model_rows = [
        "% Generated by generate_manuscript_artifacts.py; do not edit by hand.",
    ]
    for _, row in combined.iterrows():
        name = tex_escape(display_model_name(row["Model"]))
        if row["Model"] == selected_model:
            name = r"\textbf{" + name + "}"
        model_rows.append(
            f"{name} & ${fixed(row['Nested_CV_R_Squared_Mean'])} \\pm {fixed(row['Nested_CV_R_Squared_Std'])}$ "
            f"& {fixed(row['R-Squared'])} & {fixed(row['RMSE'])} & {fixed(row['MAE'])} \\\\"
        )
    write(results_dir / "manuscript_model_comparison_rows.tex", "\n".join(model_rows) + "\n")

    statistics = statistics.sort_values("CV_R2_Mean", ascending=False)
    statistic_rows = [
        "% Generated by generate_manuscript_artifacts.py; do not edit by hand.",
    ]
    for _, row in statistics.iterrows():
        statistic_rows.append(
            f"{tex_escape(display_model_name(row['Model']))} & {fixed(row['CV_R2_Mean'])} & "
            f"{fixed(row['Mean_Difference'])} & {fixed(row['Wilcoxon_p_value'], 4)} & "
            f"{fixed(row['Cohens_d'], 2)} \\\\"
        )
    write(results_dir / "manuscript_statistical_rows.tex", "\n".join(statistic_rows) + "\n")

    summary = {
        "target_id": args.target_id,
        "selected_model": selected_model,
        "seed": selection["random_seed"],
        "selection_rule": selection["selection_rule"],
        "displayed_metrics": macros,
    }
    write(results_dir / "manuscript_metrics.json", json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
