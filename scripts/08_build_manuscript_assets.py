#!/usr/bin/env python3
"""Build manuscript numbers, tables, and figures from sealed v2 artifacts.

This script deliberately has no model-fitting path. It verifies completed input
artifacts and then materializes publication assets. All prose-facing numerical
claims originate here, which prevents transcription drift between results and the
manuscript.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT_DIR = PROJECT_ROOT / "manuscript_v2"
GENERATED_DIR = MANUSCRIPT_DIR / "generated"
FIGURE_DIR = MANUSCRIPT_DIR / "figures"
CURATION_DIR = PROJECT_ROOT / "data" / "curated" / "chembl37_pim1_v2"
SNAPSHOT_DIR = PROJECT_ROOT / "data" / "releases" / "chembl37_pim1_20260501"
SPLIT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "split_locks"
    / "chembl37_strict_temporal2018_scaffold3x5_v1"
)
RUNS_DIR = PROJECT_ROOT / "artifacts" / "runs"
CLASSICAL_DIR = RUNS_DIR / "classical_nested_v2"
GNN_DIR = RUNS_DIR / "gnn_nested_v2"
TUNED_GNN_DIR = RUNS_DIR / "gnn_tuned_posthoc_v1"
TUNED_GNN_SEARCH_CONFIG = PROJECT_ROOT / "configs" / "gnn_search_v1.yaml"
SELECTION_DIR = RUNS_DIR / "development_selection_v2"
TEMPORAL_DIR = RUNS_DIR / "temporal_evaluation_v2"
ROBUSTNESS_AUDIT_DIR = (
    PROJECT_ROOT / "artifacts" / "audits" / "temporal_robustness_v2"
)
ENVIRONMENT_DIR = PROJECT_ROOT / "artifacts" / "environment"


ATP_NUMERIC_PATTERN = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>nM|uM|µM|μM|mM)\s*"
    r"(?:\[\s*)?ATP(?:\s*\])?",
    flags=re.IGNORECASE,
)
ATP_AT_KM_PATTERN = re.compile(
    r"(?:ATP\s*(?:=|at)\s*K\s*[_-]?m|conc\.\s*ATP\s*=\s*K\s*[_-]?m)",
    flags=re.IGNORECASE,
)


MODEL_LABELS = {
    "development_mean": "Development mean",
    "tanimoto_knn_ecfp6": "Tanimoto kNN (ECFP6)",
    "ridge_combined": "Ridge (ECFP6 + RDKit2D)",
    "random_forest_ecfp6": "Random forest (ECFP6)",
    "random_forest_combined": "Random forest (ECFP6 + RDKit2D)",
    "gradient_boosting_combined": "Gradient boosting (ECFP6 + RDKit2D)",
    "rbf_svr_combined": "RBF-SVR (ECFP6 + RDKit2D)",
    "edge_aware_gine_graph": "Edge-aware GINE (graph)",
    "edge_aware_gine_graph_plus_rdkit2d": "Edge-aware GINE (graph + RDKit2D)",
    "chemprop_dmpnn_graph": "Chemprop D-MPNN (graph)",
    "chemprop_dmpnn_graph_plus_rdkit2d": "Chemprop D-MPNN (graph + RDKit2D)",
}


REPRESENTATION_LABELS = {
    "development_mean": "none",
    "tanimoto_knn_ecfp6": "ECFP6",
    "ridge_combined": "ECFP6 + 23 descriptors",
    "random_forest_ecfp6": "ECFP6",
    "random_forest_combined": "ECFP6 + 23 descriptors",
    "gradient_boosting_combined": "ECFP6 + 23 descriptors",
    "rbf_svr_combined": "ECFP6 + 23 descriptors",
    "edge_aware_gine_graph": "molecular graph",
    "edge_aware_gine_graph_plus_rdkit2d": "graph + 23 descriptors",
    "chemprop_dmpnn_graph": "molecular graph",
    "chemprop_dmpnn_graph_plus_rdkit2d": "graph + 23 descriptors",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def verify_hash_inventory(root: Path, inventory: Mapping[str, str]) -> None:
    for filename, expected in inventory.items():
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(f"Checksum mismatch for {path}: {observed} != {expected}")


def verify_inputs() -> dict[str, dict[str, Any]]:
    required_complete = (
        CLASSICAL_DIR,
        GNN_DIR,
        TUNED_GNN_DIR,
        SELECTION_DIR,
        TEMPORAL_DIR,
    )
    for directory in required_complete:
        if not (directory / "RUN_COMPLETE").is_file():
            raise RuntimeError(f"Production artifact is incomplete: {directory}")
    if not (ROBUSTNESS_AUDIT_DIR / "RUN_COMPLETE").is_file():
        raise RuntimeError(f"Post hoc robustness audit is incomplete: {ROBUSTNESS_AUDIT_DIR}")

    snapshot = load_json(SNAPSHOT_DIR / "snapshot_manifest.json")
    curation = load_json(CURATION_DIR / "curation_manifest.json")
    split = load_json(SPLIT_DIR / "split_lock_manifest.json")
    classical = load_json(CLASSICAL_DIR / "run_manifest.json")
    gnn = load_json(GNN_DIR / "run_manifest.json")
    tuned_gnn = load_json(TUNED_GNN_DIR / "run_manifest.json")
    selection = load_json(SELECTION_DIR / "model_selection.json")
    temporal = load_json(TEMPORAL_DIR / "temporal_evaluation_manifest.json")
    temporal_metrics = load_json(TEMPORAL_DIR / "temporal_metrics.json")
    robustness_audit_manifest = load_json(ROBUSTNESS_AUDIT_DIR / "audit_manifest.json")
    robustness_audit = load_json(ROBUSTNESS_AUDIT_DIR / "audit_metrics.json")
    classical_environment = load_json(ENVIRONMENT_DIR / "classical_curation.json")
    gnn_environment = load_json(ENVIRONMENT_DIR / "gnn.json")

    verify_hash_inventory(CURATION_DIR, curation["output_files_sha256"])
    verify_hash_inventory(SPLIT_DIR, split["output_files_sha256"])
    verify_hash_inventory(SELECTION_DIR, selection["output_hashes"])
    verify_hash_inventory(TUNED_GNN_DIR, tuned_gnn["output_hashes"])
    verify_hash_inventory(TEMPORAL_DIR, temporal["output_hashes"])
    verify_hash_inventory(
        ROBUSTNESS_AUDIT_DIR, robustness_audit_manifest["output_hashes"]
    )
    if selection.get("temporal_test_consulted") is not False:
        raise RuntimeError("Development selection is not certified temporal-test blind")
    if temporal.get("status") != "complete" or not temporal.get(
        "one_time_selected_model_evaluation"
    ):
        raise RuntimeError("Temporal result is not a completed one-time evaluation")
    if temporal.get("selected_model") != selection.get("selected_model"):
        raise RuntimeError("Selection and temporal manifests name different models")
    if classical.get("temporal_test_consulted") is True or gnn.get(
        "temporal_test_consulted"
    ) is True:
        raise RuntimeError("A development benchmark consulted the temporal test")
    if tuned_gnn.get("status") != "post_hoc_development_only" or tuned_gnn.get(
        "temporal_test_consulted"
    ) is not False:
        raise RuntimeError("Tuned GNN benchmark is not certified development-only")
    if tuned_gnn.get("schema_version") != "gnn_tuned_posthoc_v1":
        raise RuntimeError("Unexpected tuned GNN artifact schema")
    if tuned_gnn.get("search_config_sha256") != sha256_file(TUNED_GNN_SEARCH_CONFIG):
        raise RuntimeError("Tuned GNN artifact does not match the local search contract")
    tuned_search = yaml.safe_load(TUNED_GNN_SEARCH_CONFIG.read_text(encoding="utf-8"))
    if tuned_search.get("status") != "post_hoc_development_only":
        raise RuntimeError("Tuned GNN search config is not labeled development-only")
    if (TUNED_GNN_DIR / "RUN_COMPLETE").read_text(encoding="utf-8").strip() != str(
        tuned_gnn.get("contract_sha256")
    ):
        raise RuntimeError("Tuned GNN completion marker does not match its contract")
    expected_tuned_outputs = {
        "inner_trial_metrics.csv",
        "selected_hyperparameters.csv",
        "seed_oof_predictions.csv",
        "ensemble_oof_predictions.csv",
        "seed_metrics.csv",
        "repeat_metrics.csv",
        "model_summary.csv",
        "paired_scaffold_bootstrap.csv",
    }
    if set(tuned_gnn.get("output_hashes", {})) != expected_tuned_outputs:
        raise RuntimeError("Unexpected tuned GNN summary-output inventory")
    expected_families = {
        "edge_aware_gine",
        "chemprop_dmpnn",
        "chemprop_chemeleon",
    }
    if set(tuned_gnn.get("families", [])) != expected_families:
        raise RuntimeError("Unexpected tuned GNN family inventory")
    candidate_budget = int(tuned_gnn.get("candidate_budget_per_family_per_outer_fold", 0))
    rerank_seeds = tuple(int(value) for value in tuned_gnn.get("rerank_seeds", []))
    refit_seeds = tuple(int(value) for value in tuned_gnn.get("outer_refit_seeds", []))
    rerank_top = int(tuned_gnn.get("rerank_top_candidates", 0))
    if (
        candidate_budget != int(tuned_search["candidate_budget_per_family"])
        or rerank_top != int(tuned_search["rerank_top_candidates"])
        or rerank_seeds
        != tuple(int(value) for value in tuned_search["rerank_initialization_seeds"])
        or refit_seeds
        != tuple(int(value) for value in tuned_search["outer_refit_initialization_seeds"])
    ):
        raise RuntimeError("Tuned GNN manifest and search config disagree")
    expected_trials = len(expected_families) * 3 * 5 * (
        candidate_budget + rerank_top * (len(rerank_seeds) - 1)
    )
    if int(tuned_gnn.get("trial_count", -1)) != expected_trials:
        raise RuntimeError("Tuned GNN trial count is inconsistent with its contract")
    if int(tuned_gnn.get("inner_fit_count", -1)) != expected_trials * 3:
        raise RuntimeError("Tuned GNN inner-fit count is inconsistent with its contract")
    if candidate_budget < 2 or rerank_top < 1 or len(rerank_seeds) != 3 or len(refit_seeds) != 3:
        raise RuntimeError("Unexpected tuned GNN search or seed budget")
    if tuned_gnn["source_hashes"]["development"] != split["output_files_sha256"][
        "development.csv"
    ] or tuned_gnn["source_hashes"]["assignments"] != split[
        "output_files_sha256"
    ]["outer_fold_assignments.csv"]:
        raise RuntimeError("Tuned GNN benchmark does not use the immutable split inputs")

    return {
        "snapshot": snapshot,
        "curation": curation,
        "split": split,
        "classical": classical,
        "gnn": gnn,
        "tuned_gnn": tuned_gnn,
        "selection": selection,
        "temporal": temporal,
        "temporal_metrics": temporal_metrics,
        "robustness_audit_manifest": robustness_audit_manifest,
        "robustness_audit": robustness_audit,
        "classical_environment": classical_environment,
        "gnn_environment": gnn_environment,
    }


def tex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def tex_breakable_identifier(value: Any) -> str:
    """Escape an identifier while allowing line breaks after underscores."""

    return tex_escape(value).replace(r"\_", r"\_\allowbreak{}")


def format_int(value: Any) -> str:
    return f"{int(value):,}"


def fmt(value: Any, digits: int = 3) -> str:
    number = float(value)
    if not math.isfinite(number):
        return "NA"
    if abs(number) < 0.5 * 10 ** (-digits):
        number = 0.0
    return f"{number:.{digits}f}"


def regression_metrics(observed: Iterable[float], predicted: Iterable[float]) -> dict[str, float]:
    y = np.asarray(list(observed), dtype=float)
    p = np.asarray(list(predicted), dtype=float)
    if y.shape != p.shape or y.ndim != 1 or not len(y):
        raise ValueError("Metric vectors must be aligned and non-empty")
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    mae = float(np.mean(np.abs(y - p)))
    denominator = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1.0 - np.sum((y - p) ** 2) / denominator) if denominator > 0 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def correlation_metrics(
    observed: Iterable[float], predicted: Iterable[float]
) -> dict[str, float]:
    """Return Pearson and tie-aware Spearman correlations without SciPy."""

    y = np.asarray(list(observed), dtype=float)
    p = np.asarray(list(predicted), dtype=float)
    if y.shape != p.shape or y.ndim != 1 or not len(y):
        raise ValueError("Correlation vectors must be aligned and non-empty")

    def pearson(left: np.ndarray, right: np.ndarray) -> float:
        if len(left) < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
            return float("nan")
        return float(np.corrcoef(left, right)[0, 1])

    y_rank = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    p_rank = pd.Series(p).rank(method="average").to_numpy(dtype=float)
    return {
        "pearson_r": pearson(y, p),
        "spearman_rho": pearson(y_rank, p_rank),
    }


def temporal_drift_metrics(predictions: pd.DataFrame) -> dict[str, Any]:
    """Compute explicitly post hoc, descriptive temporal-drift diagnostics.

    These calculations only summarize the immutable one-time temporal prediction
    file. They are not model-selection inputs and do not modify any prespecified
    temporal-evaluation artifact.
    """

    required = {"first_record_year", "y_true", "y_pred", "scaffold_smiles"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Temporal predictions are missing columns: {sorted(missing)}")

    annual: list[dict[str, Any]] = []
    for year, group in predictions.groupby("first_record_year", sort=True):
        scores = regression_metrics(group["y_true"], group["y_pred"])
        correlations = correlation_metrics(group["y_true"], group["y_pred"])
        annual.append(
            {
                "year": int(year),
                "n": int(len(group)),
                "n_scaffolds": int(group["scaffold_smiles"].nunique()),
                "mean_observed": float(group["y_true"].mean()),
                "mean_predicted": float(group["y_pred"].mean()),
                "sd_observed": float(group["y_true"].std(ddof=0)),
                **scores,
                **correlations,
            }
        )

    years = {row["year"] for row in annual}
    if 2019 not in years or 2024 not in years:
        raise RuntimeError("Temporal drift summary requires the 2019 and 2024 cohorts")

    observed = predictions["y_true"].to_numpy(dtype=float)
    overall_mean = float(observed.mean())
    total_sum_squares = float(np.sum((observed - overall_mean) ** 2))
    between_year_sum_squares = 0.0
    for row in annual:
        between_year_sum_squares += row["n"] * (row["mean_observed"] - overall_mean) ** 2
    if total_sum_squares <= 0:
        raise RuntimeError("Cannot decompose a constant temporal target")

    def within_year_summary(frame: pd.DataFrame) -> dict[str, float]:
        centered_observed = frame["y_true"] - frame.groupby("first_record_year")[
            "y_true"
        ].transform("mean")
        centered_predicted = frame["y_pred"] - frame.groupby("first_record_year")[
            "y_pred"
        ].transform("mean")
        return {
            "r2": regression_metrics(centered_observed, centered_predicted)["r2"],
            "spearman_rho": correlation_metrics(
                centered_observed, centered_predicted
            )["spearman_rho"],
        }

    within_all = within_year_summary(predictions)
    post_2019 = predictions.loc[predictions["first_record_year"].ge(2020)].copy()
    within_post_2019 = within_year_summary(post_2019)
    annual_by_year = {row["year"]: row for row in annual}
    rmse_2019 = float(annual_by_year[2019]["rmse"])
    rmse_2024 = float(annual_by_year[2024]["rmse"])
    rmse_increase = rmse_2024 - rmse_2019

    return {
        "status": "post_hoc_descriptive",
        "annual_metrics": annual,
        "between_year_target_variance_percent": 100.0
        * between_year_sum_squares
        / total_sum_squares,
        "within_year_centered_r2_all_years": float(within_all["r2"]),
        "within_year_centered_spearman_all_years": float(
            within_all["spearman_rho"]
        ),
        "within_year_centered_r2_2020_2025": float(within_post_2019["r2"]),
        "within_year_centered_spearman_2020_2025": float(
            within_post_2019["spearman_rho"]
        ),
        "rmse_2019": rmse_2019,
        "rmse_2024": rmse_2024,
        "rmse_2019_to_2024_increase": rmse_increase,
        "rmse_2019_to_2024_increase_percent": 100.0 * rmse_increase / rmse_2019,
        "definitions": {
            "between_year_target_variance_percent": (
                "100 times between-year target sum of squares divided by total "
                "target sum of squares"
            ),
            "within_year_centered_r2": (
                "R2 after subtracting each year-specific mean separately from "
                "observed and predicted pIC50"
            ),
            "within_year_centered_spearman": (
                "Spearman correlation after subtracting each year-specific mean "
                "separately from observed and predicted pIC50"
            ),
            "annual_sd_observed": "population SD (ddof=0) within each year",
        },
    }


def _atp_concentrations_um(description: Any) -> tuple[float, ...]:
    """Extract explicitly stated ATP concentrations and normalize them to micromolar."""

    if not isinstance(description, str):
        return ()
    values: list[float] = []
    for match in ATP_NUMERIC_PATTERN.finditer(description):
        value = float(match.group("value"))
        unit = match.group("unit").casefold()
        if unit == "nm":
            value /= 1000.0
        elif unit == "mm":
            value *= 1000.0
        elif unit not in {"um", "µm", "μm"}:
            raise RuntimeError(f"Unhandled ATP concentration unit: {unit}")
        if value <= 0 or not math.isfinite(value):
            raise ValueError(f"Invalid ATP concentration in description: {description}")
        values.append(value)
    return tuple(sorted(set(values)))


def assay_context_metrics(
    events: pd.DataFrame,
    development_identities: Iterable[str] | None = None,
    temporal_identities: Iterable[str] | None = None,
    cutoff_year: int = 2018,
) -> dict[str, Any]:
    """Describe residual assay provenance without estimating causal assay effects.

    ATP fields are lower bounds based only on machine-readable text in the pinned
    assay descriptions. Partition summaries exclude quarantined post-cutoff events
    attached to development identities.
    """

    required = {
        "event_id",
        "representative_assay_chembl_id",
        "document_chembl_id",
        "document_year",
        "assay_description",
        "standardized_parent_inchi_key",
        "scaffold_smiles",
        "standard_value_nm",
        "pIC50",
    }
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"Strict events are missing columns: {sorted(missing)}")
    if events.empty:
        raise ValueError("Strict event table is empty")
    if events["event_id"].duplicated().any():
        raise ValueError("Strict event table contains duplicate event IDs")
    if events[list(required - {"assay_description", "scaffold_smiles"})].isna().any().any():
        raise ValueError("Strict event table contains missing required values")
    if not np.isfinite(events[["standard_value_nm", "pIC50"]].to_numpy(dtype=float)).all():
        raise ValueError("Strict event table contains non-finite potency values")

    frame = events.copy()
    assay_column = "representative_assay_chembl_id"
    document_column = "document_chembl_id"
    identity_column = "standardized_parent_inchi_key"
    frame["_atp_um_values"] = frame["assay_description"].map(_atp_concentrations_um)
    ambiguous = frame["_atp_um_values"].map(len).gt(1)
    if ambiguous.any():
        examples = frame.loc[ambiguous, "event_id"].astype(str).tolist()[:5]
        raise ValueError(f"Assay descriptions contain multiple ATP concentrations: {examples}")
    frame["_atp_um"] = frame["_atp_um_values"].map(
        lambda values: values[0] if values else np.nan
    )
    atp_mention = frame["assay_description"].fillna("").str.contains(
        r"\bATP\b", case=False, regex=True
    )
    numeric_atp = frame["_atp_um"].notna()
    atp_at_km = frame["assay_description"].fillna("").map(
        lambda value: bool(ATP_AT_KM_PATTERN.search(value))
    )

    event_count = int(len(frame))
    assay_event_counts = frame.groupby(assay_column, sort=True).size().sort_values(
        ascending=False
    )
    document_event_counts = frame.groupby(document_column, sort=True).size().sort_values(
        ascending=False
    )
    assay_count = int(len(assay_event_counts))
    document_count = int(len(document_event_counts))
    compound_event_counts = frame.groupby(identity_column, sort=True).size()
    compound_assay_counts = frame.groupby(identity_column, sort=True)[assay_column].nunique()
    compound_document_counts = frame.groupby(identity_column, sort=True)[document_column].nunique()

    scaffold_key = frame["scaffold_smiles"].fillna("[ACYCLIC]").astype(str)
    scaffold_assay_counts = frame.assign(_scaffold=scaffold_key).groupby(
        "_scaffold", sort=True
    )[assay_column].nunique()
    assay_exclusive_scaffolds = set(scaffold_assay_counts.loc[lambda values: values.eq(1)].index)
    assay_exclusive_scaffold_events = int(scaffold_key.isin(assay_exclusive_scaffolds).sum())

    numeric_values = frame.loc[numeric_atp, "_atp_um"].to_numpy(dtype=float)
    if len(numeric_values):
        numeric_min = float(numeric_values.min())
        numeric_max = float(numeric_values.max())
        numeric_fold_range = numeric_max / numeric_min
    else:
        numeric_min = numeric_max = numeric_fold_range = float("nan")

    def assay_rows(mask: pd.Series, condition: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for assay_id, group in frame.loc[mask].groupby(assay_column, sort=True):
            concentrations = sorted(
                {
                    float(value)
                    for values in group["_atp_um_values"]
                    for value in values
                }
            )
            rows.append(
                {
                    "assay_chembl_id": str(assay_id),
                    "document_chembl_ids": sorted(
                        group[document_column].astype(str).unique().tolist()
                    ),
                    "event_count": int(len(group)),
                    "condition": condition,
                    "atp_concentrations_um": concentrations,
                    "assay_descriptions": sorted(
                        group["assay_description"].dropna().astype(str).unique().tolist()
                    ),
                }
            )
        return rows

    # Enumerate every compound/document bridge between assays with explicit but
    # different ATP concentrations; no pair is selected by potency difference.
    numeric_records = (
        frame.loc[numeric_atp]
        .groupby(
            [identity_column, document_column, assay_column, "_atp_um"],
            sort=True,
            dropna=False,
        )
        .agg(
            event_count=("event_id", "size"),
            standard_value_nm=("standard_value_nm", "median"),
            pIC50=("pIC50", "median"),
        )
        .reset_index()
    )
    matched_pairs: list[dict[str, Any]] = []
    for (identity, document), group in numeric_records.groupby(
        [identity_column, document_column], sort=True
    ):
        records = list(group.sort_values(["_atp_um", assay_column]).to_dict("records"))
        for left_index, left in enumerate(records):
            for right in records[left_index + 1 :]:
                if left[assay_column] == right[assay_column]:
                    continue
                if float(left["_atp_um"]) == float(right["_atp_um"]):
                    continue
                low, high = sorted((left, right), key=lambda row: float(row["_atp_um"]))
                matched_pairs.append(
                    {
                        "standardized_parent_inchi_key": str(identity),
                        "document_chembl_id": str(document),
                        "low_atp_assay_chembl_id": str(low[assay_column]),
                        "high_atp_assay_chembl_id": str(high[assay_column]),
                        "low_atp_um": float(low["_atp_um"]),
                        "high_atp_um": float(high["_atp_um"]),
                        "low_atp_standard_value_nm": float(low["standard_value_nm"]),
                        "high_atp_standard_value_nm": float(high["standard_value_nm"]),
                        "pic50_delta_high_minus_low_atp": float(
                            high["pIC50"] - low["pIC50"]
                        ),
                    }
                )

    partition: dict[str, Any] = {}
    if (development_identities is None) != (temporal_identities is None):
        raise ValueError("Development and temporal identity sets must be supplied together")
    if development_identities is not None and temporal_identities is not None:
        development_set = {str(value) for value in development_identities}
        temporal_set = {str(value) for value in temporal_identities}
        if not development_set or not temporal_set:
            raise ValueError("Development and temporal identity sets must be non-empty")
        if development_set.intersection(temporal_set):
            raise ValueError("Development and temporal identity sets overlap")
        known_identities = set(frame[identity_column].astype(str))
        missing_identities = (development_set | temporal_set).difference(known_identities)
        if missing_identities:
            raise ValueError(
                f"Partition identities are absent from strict events: {sorted(missing_identities)[:5]}"
            )
        development_events = frame.loc[
            frame[identity_column].astype(str).isin(development_set)
            & frame["document_year"].le(cutoff_year)
        ]
        temporal_events = frame.loc[
            frame[identity_column].astype(str).isin(temporal_set)
        ]
        if temporal_events["document_year"].le(cutoff_year).any():
            raise ValueError("Temporal identities contain an event at or before the cutoff")
        development_assays = set(development_events[assay_column].astype(str))
        temporal_assays = set(temporal_events[assay_column].astype(str))
        development_documents = set(development_events[document_column].astype(str))
        temporal_documents = set(temporal_events[document_column].astype(str))
        partition = {
            "development_event_count": int(len(development_events)),
            "development_assay_count": int(len(development_assays)),
            "development_document_count": int(len(development_documents)),
            "temporal_event_count": int(len(temporal_events)),
            "temporal_assay_count": int(len(temporal_assays)),
            "temporal_document_count": int(len(temporal_documents)),
            "development_temporal_assay_intersection_count": int(
                len(development_assays.intersection(temporal_assays))
            ),
            "development_temporal_document_intersection_count": int(
                len(development_documents.intersection(temporal_documents))
            ),
            "quarantined_post_cutoff_development_event_count": int(
                (
                    frame[identity_column].astype(str).isin(development_set)
                    & frame["document_year"].gt(cutoff_year)
                ).sum()
            ),
        }

    single_event_compounds = int(compound_event_counts.eq(1).sum())
    return {
        "status": "post_hoc_descriptive",
        "event_count": event_count,
        "assay_count": assay_count,
        "document_count": document_count,
        "description_count": int(frame["assay_description"].nunique(dropna=True)),
        "assay_event_count_min": int(assay_event_counts.min()),
        "assay_event_count_q1": float(assay_event_counts.quantile(0.25)),
        "assay_event_count_median": float(assay_event_counts.median()),
        "assay_event_count_q3": float(assay_event_counts.quantile(0.75)),
        "assay_event_count_max": int(assay_event_counts.max()),
        "single_event_assay_count": int(assay_event_counts.eq(1).sum()),
        "assay_lt_five_event_count": int(assay_event_counts.lt(5).sum()),
        "assay_ge_ten_event_count": int(assay_event_counts.ge(10).sum()),
        "largest_assay_event_count": int(assay_event_counts.iloc[0]),
        "largest_assay_event_percent": 100.0 * float(assay_event_counts.iloc[0]) / event_count,
        "top_five_assay_event_count": int(assay_event_counts.head(5).sum()),
        "top_five_assay_event_percent": 100.0 * float(assay_event_counts.head(5).sum()) / event_count,
        "top_ten_assay_event_count": int(assay_event_counts.head(10).sum()),
        "top_ten_assay_event_percent": 100.0 * float(assay_event_counts.head(10).sum()) / event_count,
        "atp_mention_assay_count": int(frame.loc[atp_mention, assay_column].nunique()),
        "atp_mention_event_count": int(atp_mention.sum()),
        "atp_mention_event_percent": 100.0 * float(atp_mention.sum()) / event_count,
        "numeric_atp_assay_count": int(frame.loc[numeric_atp, assay_column].nunique()),
        "numeric_atp_event_count": int(numeric_atp.sum()),
        "numeric_atp_event_percent": 100.0 * float(numeric_atp.sum()) / event_count,
        "numeric_atp_min_um": numeric_min,
        "numeric_atp_max_um": numeric_max,
        "numeric_atp_fold_range": numeric_fold_range,
        "atp_at_km_assay_count": int(frame.loc[atp_at_km, assay_column].nunique()),
        "atp_at_km_event_count": int(atp_at_km.sum()),
        "single_event_compound_count": single_event_compounds,
        "single_event_compound_percent": 100.0 * single_event_compounds / len(compound_event_counts),
        "cross_assay_compound_count": int(compound_assay_counts.gt(1).sum()),
        "cross_document_compound_count": int(compound_document_counts.gt(1).sum()),
        "scaffold_count": int(len(scaffold_assay_counts)),
        "multi_assay_scaffold_count": int(scaffold_assay_counts.gt(1).sum()),
        "assay_exclusive_scaffold_event_count": assay_exclusive_scaffold_events,
        "assay_exclusive_scaffold_event_percent": 100.0
        * assay_exclusive_scaffold_events
        / event_count,
        "numeric_atp_assays": assay_rows(numeric_atp, "numeric"),
        "atp_at_km_assays": assay_rows(atp_at_km, "ATP at Km"),
        "matched_numeric_atp_pair_count": int(len(matched_pairs)),
        "matched_numeric_atp_pairs": matched_pairs,
        **partition,
        "definitions": {
            "atp_mention": "case-insensitive whole-word ATP in the pinned assay description",
            "numeric_atp": "a positive numeric nM, uM, microM, or mM value immediately modifying ATP in the pinned assay description",
            "assay_exclusive_scaffold_event": "an event whose Bemis--Murcko scaffold occurs in exactly one representative assay",
            "partition_event_counts": "development label events at or before the cutoff and all events for first-seen-after-cutoff temporal identities",
        },
    }


def gnn_training_duration_metrics(
    checkpoint_dir: Path,
    max_epochs: int,
    *,
    require_prediction_files: bool = False,
    expected_contract_sha256: str | None = None,
) -> dict[str, Any]:
    """Audit realized early-stopping durations from immutable GNN checkpoints."""

    if max_epochs < 1:
        raise ValueError("max_epochs must be positive")
    paths = sorted(checkpoint_dir.glob("*.json"))
    if not paths:
        raise ValueError(f"No GNN checkpoint manifests found in {checkpoint_dir}")
    inner_epochs: list[int] = []
    refit_epochs: list[int] = []
    identifiers: set[tuple[str, int, int]] = set()
    for path in paths:
        checkpoint = load_json(path)
        metrics = checkpoint.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError(f"Checkpoint lacks a metrics object: {path}")
        required = {"model", "repeat", "fold", "inner_best_epochs", "refit_epochs"}
        missing = required.difference(metrics)
        if missing:
            raise ValueError(f"Checkpoint {path} is missing fields: {sorted(missing)}")
        identifier = (str(metrics["model"]), int(metrics["repeat"]), int(metrics["fold"]))
        if identifier in identifiers:
            raise ValueError(f"Duplicate GNN checkpoint identifier: {identifier}")
        identifiers.add(identifier)
        values = metrics["inner_best_epochs"]
        if not isinstance(values, list) or not values:
            raise ValueError(f"Checkpoint has no inner epoch selections: {path}")
        if any(isinstance(value, bool) or int(value) != value for value in values):
            raise ValueError(f"Checkpoint has a non-integer inner epoch: {path}")
        refit = metrics["refit_epochs"]
        if isinstance(refit, bool) or int(refit) != refit:
            raise ValueError(f"Checkpoint has a non-integer refit epoch: {path}")
        selected = [int(value) for value in values]
        refit_value = int(refit)
        if min(selected + [refit_value]) < 1 or max(selected + [refit_value]) > max_epochs:
            raise ValueError(f"Checkpoint epoch lies outside 1..{max_epochs}: {path}")
        inner_epochs.extend(selected)
        refit_epochs.append(refit_value)
        if expected_contract_sha256 is not None and checkpoint.get(
            "contract_sha256"
        ) != expected_contract_sha256:
            raise RuntimeError(f"GNN checkpoint contract mismatch: {path}")
        if require_prediction_files:
            prediction_path = path.with_suffix(".csv")
            if not prediction_path.is_file():
                raise FileNotFoundError(prediction_path)
            expected_hash = checkpoint.get("predictions_sha256")
            if not isinstance(expected_hash, str) or sha256_file(prediction_path) != expected_hash:
                raise RuntimeError(f"GNN checkpoint prediction hash mismatch: {prediction_path}")

    inner_cap_hits = int(sum(epoch == max_epochs for epoch in inner_epochs))
    refit_cap_hits = int(sum(epoch == max_epochs for epoch in refit_epochs))
    return {
        "status": "realized_checkpoint_audit",
        "max_epochs": int(max_epochs),
        "checkpoint_count": int(len(paths)),
        "inner_epoch_count": int(len(inner_epochs)),
        "inner_epoch_min": int(min(inner_epochs)),
        "inner_epoch_max": int(max(inner_epochs)),
        "inner_epoch_cap_hit_count": inner_cap_hits,
        "refit_epoch_count": int(len(refit_epochs)),
        "refit_epoch_min": int(min(refit_epochs)),
        "refit_epoch_max": int(max(refit_epochs)),
        "refit_epoch_cap_hit_count": refit_cap_hits,
        "total_epoch_cap_hit_count": inner_cap_hits + refit_cap_hits,
    }


def friendly_model(model: str) -> str:
    if model not in MODEL_LABELS:
        raise ValueError(f"No manuscript label for model {model}")
    return MODEL_LABELS[model]


def family(model: str) -> str:
    if model == "development_mean" or model.startswith("tanimoto"):
        return "Baseline"
    if "gine" in model or "dmpnn" in model:
        return "Graph neural network"
    return "Classical"


def cross_cutoff_shifts() -> tuple[float, float, int]:
    development = pd.read_csv(SPLIT_DIR / "development.csv")
    cross = pd.read_csv(SPLIT_DIR / "cross_cutoff_records.csv")
    identity = "standardized_parent_inchi_key"
    future = cross.groupby(identity, sort=True)["pIC50"].median()
    training = development.set_index(identity)["pIC50"].loc[future.index]
    absolute = (future - training).abs()
    return float(absolute.median()), float(absolute.max()), int(len(absolute))


def build_results(manifests: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    snapshot = manifests["snapshot"]
    curation = manifests["curation"]
    split = manifests["split"]
    selection = manifests["selection"]
    diagnostics = manifests["temporal_metrics"]
    robustness = manifests["robustness_audit"]
    classical_environment = manifests["classical_environment"]
    gnn_environment = manifests["gnn_environment"]
    tuned_gnn_manifest = manifests["tuned_gnn"]

    model_summary = pd.read_csv(SELECTION_DIR / "model_summary.csv")
    repeat_metrics = pd.read_csv(SELECTION_DIR / "repeat_metrics.csv")
    bootstrap = pd.read_csv(SELECTION_DIR / "paired_scaffold_bootstrap.csv")
    predictions = pd.read_csv(TEMPORAL_DIR / "temporal_predictions.csv")
    tuned_summary = pd.read_csv(TUNED_GNN_DIR / "model_summary.csv")
    tuned_repeats = pd.read_csv(TUNED_GNN_DIR / "repeat_metrics.csv")
    tuned_seed_metrics = pd.read_csv(TUNED_GNN_DIR / "seed_metrics.csv")
    tuned_bootstrap = pd.read_csv(TUNED_GNN_DIR / "paired_scaffold_bootstrap.csv")
    tuned_selections = pd.read_csv(TUNED_GNN_DIR / "selected_hyperparameters.csv")
    tuned_families = (
        "edge_aware_gine",
        "chemprop_dmpnn",
        "chemprop_chemeleon",
    )
    expected_summary_keys = {(family,) for family in tuned_families}
    expected_repeat_keys = {
        (family, repeat) for family in tuned_families for repeat in range(3)
    }
    expected_seed_keys = {
        (family, repeat, seed)
        for family in tuned_families
        for repeat in range(3)
        for seed in tuned_gnn_manifest["outer_refit_seeds"]
    }
    expected_selection_keys = {
        (family, repeat, fold)
        for family in tuned_families
        for repeat in range(3)
        for fold in range(5)
    }
    expected_development_n = int(split["counts"]["development_compounds"])

    def observed_keys(frame: pd.DataFrame, columns: list[str]) -> set[tuple[Any, ...]]:
        if frame.duplicated(columns).any():
            raise RuntimeError(f"Duplicate tuned GNN keys in {columns}")
        return set(frame[columns].itertuples(index=False, name=None))

    if observed_keys(tuned_summary, ["family"]) != expected_summary_keys or not tuned_summary[
        "n_repeats"
    ].eq(3).all():
        raise RuntimeError("Tuned GNN family/repeat inventory is incomplete")
    if observed_keys(tuned_repeats, ["family", "repeat"]) != expected_repeat_keys or not tuned_repeats[
        "n_oof"
    ].eq(expected_development_n).all():
        raise RuntimeError("Tuned GNN OOF or selection inventory is incomplete")
    if observed_keys(tuned_seed_metrics, ["family", "repeat", "seed"]) != expected_seed_keys or not tuned_seed_metrics[
        "n_oof"
    ].eq(expected_development_n).all():
        raise RuntimeError("Tuned GNN seed-metric inventory is incomplete")
    if observed_keys(tuned_selections, ["family", "repeat", "fold"]) != expected_selection_keys:
        raise RuntimeError("Tuned GNN selected-configuration inventory is incomplete")
    selection_table_columns = {
        "representation",
        "depth",
        "hidden_dim",
        "ffn_hidden_dim",
        "ffn_layers",
        "dropout",
        "learning_rate",
        "weight_decay",
        "batch_size",
        "pooling",
        "normalization",
        "residual",
        "initialization",
        "activation",
        "featurization",
    }
    if not selection_table_columns.issubset(tuned_selections.columns) or tuned_selections[
        sorted(selection_table_columns)
    ].isna().any().any():
        raise RuntimeError("Tuned GNN selections cannot support the manuscript table")
    expected_bootstrap_keys = {
        (family, metric)
        for family in tuned_families
        for metric in ("RMSE", "MAE")
    }
    if observed_keys(tuned_bootstrap, ["family", "metric"]) != expected_bootstrap_keys or not tuned_bootstrap[
        "reference_model"
    ].eq("ridge_combined").all():
        raise RuntimeError("Tuned GNN paired-bootstrap inventory is incomplete")
    if not tuned_bootstrap["bootstrap_replicates"].eq(10000).all():
        raise RuntimeError("Tuned GNN paired bootstrap does not use 10,000 replicates")
    numeric_frames = (
        (tuned_summary, ["rmse_mean", "rmse_sd", "mae_mean", "mae_sd", "r2_mean", "r2_sd", "mean_seed_prediction_sd"]),
        (tuned_repeats, ["rmse", "mae", "r2", "mean_seed_prediction_sd"]),
        (tuned_seed_metrics, ["rmse", "mae", "r2"]),
        (tuned_bootstrap, ["point_difference", "ci_lower", "ci_upper"]),
    )
    for frame, columns in numeric_frames:
        if not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
            raise RuntimeError(f"Tuned GNN summary contains non-finite values: {columns}")
    tuned_best = tuned_summary.sort_values(["rmse_mean", "family"]).iloc[0]
    tuned_best_delta = tuned_bootstrap.loc[
        tuned_bootstrap["family"].eq(tuned_best["family"])
        & tuned_bootstrap["metric"].eq("RMSE")
    ].iloc[0]
    tuned_best_seed_metrics = tuned_seed_metrics.loc[
        tuned_seed_metrics["family"].eq(tuned_best["family"])
    ]
    tuned_best_within_repeat_seed_rmse_sd = float(
        tuned_best_seed_metrics.groupby("repeat")["rmse"]
        .std(ddof=0)
        .mean()
    )
    if len(predictions) != int(split["counts"]["temporal_test_compounds"]):
        raise RuntimeError("Temporal prediction inventory does not match split lock")
    if predictions["standardized_parent_inchi_key"].duplicated().any():
        raise RuntimeError("Temporal prediction inventory contains duplicate identities")
    numeric_prediction_columns = [
        "y_true",
        "y_pred",
        "absolute_error",
        "mean_5nn_tanimoto",
    ]
    if not np.isfinite(predictions[numeric_prediction_columns].to_numpy(dtype=float)).all():
        raise RuntimeError("Temporal predictions contain non-finite values")
    temporal_drift = temporal_drift_metrics(predictions)

    strict_events = pd.read_csv(CURATION_DIR / "strict_measurement_events.csv")
    development_table = pd.read_csv(SPLIT_DIR / "development.csv")
    temporal_table = pd.read_csv(SPLIT_DIR / "temporal_test_sealed.csv")
    identity_column = "standardized_parent_inchi_key"
    assay_context = assay_context_metrics(
        strict_events,
        development_table[identity_column],
        temporal_table[identity_column],
        cutoff_year=int(split["split_definitions"]["temporal"]["cutoff_year"]),
    )
    if assay_context["event_count"] != int(
        curation["cohort_counts"]["strict"]["measurement_events"]
    ):
        raise RuntimeError("Assay-context event inventory disagrees with curation")
    if assay_context["development_event_count"] != int(
        development_table["measurement_count"].sum()
    ) or assay_context["temporal_event_count"] != int(
        temporal_table["measurement_count"].sum()
    ):
        raise RuntimeError("Assay-context partition event counts disagree with split labels")
    accounted_events = (
        assay_context["development_event_count"]
        + assay_context["temporal_event_count"]
        + assay_context["quarantined_post_cutoff_development_event_count"]
    )
    if accounted_events != assay_context["event_count"]:
        raise RuntimeError("Assay-context partition accounting is incomplete")
    if assay_context["matched_numeric_atp_pair_count"] != 1:
        raise RuntimeError("Pinned assay audit no longer has exactly one matched ATP pair")

    gnn_training = gnn_training_duration_metrics(
        GNN_DIR / "checkpoints",
        int(manifests["gnn"]["config"]["max_epochs"]),
        require_prediction_files=True,
        expected_contract_sha256=str(manifests["gnn"]["contract_sha256"]),
    )
    graph_candidate_count = int(
        model_summary["model"].str.contains("gine|dmpnn", regex=True).sum()
    )
    expected_checkpoints = (
        graph_candidate_count
        * int(model_summary["n_repeats"].min())
        * int(repeat_metrics["model"].map(lambda _: 5).iloc[0])
    )
    if gnn_training["checkpoint_count"] != expected_checkpoints:
        raise RuntimeError("GNN checkpoint inventory is incomplete")
    if gnn_training["inner_epoch_count"] != expected_checkpoints * int(
        manifests["gnn"]["config"]["inner_folds"]
    ):
        raise RuntimeError("GNN inner early-stopping inventory is incomplete")

    selected_model = str(selection["selected_model"])
    selected_row = model_summary.loc[model_summary["model"].eq(selected_model)].iloc[0]
    graph_rows = model_summary.loc[
        model_summary["model"].str.contains("gine|dmpnn", regex=True)
    ].sort_values("rmse_mean")
    best_graph = graph_rows.iloc[0]
    graph_delta = bootstrap.loc[
        bootstrap["candidate_model"].eq(best_graph["model"])
        & bootstrap["metric"].eq("RMSE")
    ].iloc[0]
    runner_up = model_summary.iloc[1]
    runner_up_delta = bootstrap.loc[
        bootstrap["candidate_model"].eq(runner_up["model"])
        & bootstrap["metric"].eq("RMSE")
    ].iloc[0]
    primary = diagnostics["primary_metrics"]
    point = primary["point_estimates"]
    confidence = primary["confidence_intervals"]
    novelty = {
        row["group"]: row for row in diagnostics["scaffold_novelty"]["group_metrics"]
    }
    ad_groups = {
        row["group"]: row for row in diagnostics["applicability_domain"]["group_metrics"]
    }
    cross_median, cross_maximum, cross_identity_count = cross_cutoff_shifts()

    gine_graph = float(
        model_summary.loc[
            model_summary["model"].eq("edge_aware_gine_graph"), "rmse_mean"
        ].iloc[0]
    )
    gine_descriptors = float(
        model_summary.loc[
            model_summary["model"].eq("edge_aware_gine_graph_plus_rdkit2d"),
            "rmse_mean",
        ].iloc[0]
    )
    dmpnn_graph = float(
        model_summary.loc[
            model_summary["model"].eq("chemprop_dmpnn_graph"), "rmse_mean"
        ].iloc[0]
    )
    dmpnn_descriptors = float(
        model_summary.loc[
            model_summary["model"].eq("chemprop_dmpnn_graph_plus_rdkit2d"),
            "rmse_mean",
        ].iloc[0]
    )
    if gine_descriptors < gine_graph and dmpnn_descriptors >= dmpnn_graph:
        descriptor_conclusion = "improved GINE but not the D-MPNN"
    elif gine_descriptors < gine_graph and dmpnn_descriptors < dmpnn_graph:
        descriptor_conclusion = "improved both evaluated graph backends"
    elif gine_descriptors >= gine_graph and dmpnn_descriptors >= dmpnn_graph:
        descriptor_conclusion = "did not improve either evaluated graph backend"
    else:
        descriptor_conclusion = "improved the D-MPNN but not GINE"

    fixed_wins = float(best_graph["rmse_mean"]) > float(selected_row["rmse_mean"])
    if fixed_wins:
        comparison_conclusion = (
            "under the prespecified modest search budget, none of the four evaluated "
            "graph variants improved on the leading ECFP6-plus-descriptor models"
        )
        discussion_finding = (
            "the bounded development comparison selected a regularized ECFP6-plus-"
            "descriptor model, without establishing a general ranking of fixed and "
            "learned representations"
        )
        conclusion_finding = (
            "the sealed temporal audit revealed cohort-level degradation and failure "
            "of the evaluated reliability diagnostics to identify high-error predictions"
        )
    else:
        comparison_conclusion = (
            "a learned graph representation led the evaluated fixed representations "
            "under the locked protocol"
        )
        discussion_finding = (
            "the selected graph representation improved on the fixed-representation "
            "candidates and retained predictive value under temporal shift"
        )
        conclusion_finding = (
            "the selected graph model led development performance, while temporal "
            "testing still revealed a material generalization gap"
        )

    return {
        "snapshot_activity_count": int(snapshot["counts"]["activities"]),
        "snapshot_assay_count": int(snapshot["counts"]["assays_for_target"]),
        "snapshot_parent_count": int(snapshot["counts"]["parent_molecules"]),
        "broad_eligible_count": int(curation["cohort_counts"]["broad"]["eligible_records"]),
        "broad_event_count": int(curation["cohort_counts"]["broad"]["measurement_events"]),
        "broad_compound_count": int(curation["cohort_counts"]["broad"]["compounds"]),
        "strict_event_count": int(curation["cohort_counts"]["strict"]["measurement_events"]),
        "strict_compound_count": int(curation["cohort_counts"]["strict"]["compounds"]),
        "no_high_tail_event_count": int(
            curation["cohort_counts"]["strict_no_high_tail"]["measurement_events"]
        ),
        "no_high_tail_compound_count": int(
            curation["cohort_counts"]["strict_no_high_tail"]["compounds"]
        ),
        "homologous_excluded_count": int(
            curation["exclusion_reason_counts"]["strict:assay_confidence_not_strict"]
        ),
        "structure_failure_count": int(curation["input_counts"]["structure_standardization_failures"]),
        "multi_event_compound_count": int(
            curation["measurement_dispersion"]["strict_compounds_with_multiple_events"]
        ),
        "dispersion_half_count": int(
            curation["measurement_dispersion"]["strict_compounds_range_gt_0_5"]
        ),
        "dispersion_one_count": int(
            curation["measurement_dispersion"]["strict_compounds_range_gt_1"]
        ),
        "development_count": int(split["counts"]["development_compounds"]),
        "development_scaffold_count": int(split["counts"]["development_scaffolds"]),
        "temporal_count": int(split["counts"]["temporal_test_compounds"]),
        "temporal_scaffold_count": int(split["counts"]["temporal_test_scaffolds"]),
        "new_temporal_scaffold_count": int(split["counts"]["new_temporal_test_scaffolds"]),
        "new_temporal_compound_count": int(
            split["counts"]["temporal_test_compounds_on_new_scaffolds"]
        ),
        "new_temporal_compound_percent": 100.0
        * int(split["counts"]["temporal_test_compounds_on_new_scaffolds"])
        / int(split["counts"]["temporal_test_compounds"]),
        "seen_temporal_compound_count": int(novelty["seen_scaffold"]["n"]),
        "cross_cutoff_identity_count": cross_identity_count,
        "cross_cutoff_median_shift": cross_median,
        "cross_cutoff_maximum_shift": cross_maximum,
        "outer_repeat_count": int(model_summary["n_repeats"].min()),
        "outer_fold_count": int(repeat_metrics["model"].map(lambda _: 5).iloc[0]),
        "selected_model": selected_model,
        "selected_model_text": friendly_model(selected_model),
        "development_rmse": float(selected_row["rmse_mean"]),
        "best_graph_model": str(best_graph["model"]),
        "best_graph_model_text": friendly_model(str(best_graph["model"])),
        "best_graph_rmse": float(best_graph["rmse_mean"]),
        "best_graph_delta": float(graph_delta["candidate_minus_reference"]),
        "best_graph_delta_lower": float(graph_delta["ci_lower"]),
        "best_graph_delta_upper": float(graph_delta["ci_upper"]),
        "runner_up_model": str(runner_up["model"]),
        "runner_up_model_text": friendly_model(str(runner_up["model"])),
        "runner_up_rmse": float(runner_up["rmse_mean"]),
        "runner_up_delta": float(runner_up_delta["candidate_minus_reference"]),
        "runner_up_delta_lower": float(runner_up_delta["ci_lower"]),
        "runner_up_delta_upper": float(runner_up_delta["ci_upper"]),
        "descriptor_ablation_conclusion": descriptor_conclusion,
        "model_comparison_conclusion": comparison_conclusion,
        "discussion_model_finding": discussion_finding,
        "conclusion_finding": conclusion_finding,
        "temporal_rmse": float(point["RMSE"]),
        "temporal_rmse_lower": float(confidence["RMSE"][0]),
        "temporal_rmse_upper": float(confidence["RMSE"][1]),
        "temporal_mae": float(point["MAE"]),
        "temporal_mae_lower": float(confidence["MAE"][0]),
        "temporal_mae_upper": float(confidence["MAE"][1]),
        "temporal_r2": float(point["R2"]),
        "temporal_r2_lower": float(confidence["R2"][0]),
        "temporal_r2_upper": float(confidence["R2"][1]),
        "temporal_generalization_gap": float(point["RMSE"] - selected_row["rmse_mean"]),
        "temporal_drift_analysis_status": temporal_drift["status"],
        "temporal_drift_metric_definitions": temporal_drift["definitions"],
        "temporal_annual_metrics": temporal_drift["annual_metrics"],
        "between_year_variance_percent": float(
            temporal_drift["between_year_target_variance_percent"]
        ),
        "within_year_centered_r2_all": float(
            temporal_drift["within_year_centered_r2_all_years"]
        ),
        "within_year_centered_spearman_all": float(
            temporal_drift["within_year_centered_spearman_all_years"]
        ),
        "within_year_centered_r2_2020_2025": float(
            temporal_drift["within_year_centered_r2_2020_2025"]
        ),
        "within_year_centered_spearman_2020_2025": float(
            temporal_drift["within_year_centered_spearman_2020_2025"]
        ),
        "temporal_rmse_2019": float(temporal_drift["rmse_2019"]),
        "temporal_rmse_2024": float(temporal_drift["rmse_2024"]),
        "temporal_rmse_increase": float(
            temporal_drift["rmse_2019_to_2024_increase"]
        ),
        "temporal_rmse_increase_percent": float(
            temporal_drift["rmse_2019_to_2024_increase_percent"]
        ),
        "temporal_robustness_audit": robustness,
        "temporal_rmse_increase_scaffold_lower": float(
            robustness["temporal_degradation"]["scaffold_smiles"]["rmse_delta"][
                "ci_lower"
            ]
        ),
        "temporal_rmse_increase_scaffold_upper": float(
            robustness["temporal_degradation"]["scaffold_smiles"]["rmse_delta"][
                "ci_upper"
            ]
        ),
        "absolute_error_slope_per_year": float(
            robustness["temporal_degradation"]["scaffold_smiles"][
                "absolute_error_linear_slope_per_year"
            ]["point_estimate"]
        ),
        "absolute_error_slope_scaffold_lower": float(
            robustness["temporal_degradation"]["scaffold_smiles"][
                "absolute_error_linear_slope_per_year"
            ]["ci_lower"]
        ),
        "absolute_error_slope_scaffold_upper": float(
            robustness["temporal_degradation"]["scaffold_smiles"][
                "absolute_error_linear_slope_per_year"
            ]["ci_upper"]
        ),
        "document_cluster_rmse_lower": float(
            robustness["primary_uncertainty_by_cluster"]["document_cluster"][
                "confidence_intervals"
            ]["RMSE"][0]
        ),
        "document_cluster_rmse_upper": float(
            robustness["primary_uncertainty_by_cluster"]["document_cluster"][
                "confidence_intervals"
            ]["RMSE"][1]
        ),
        "document_cluster_r2_lower": float(
            robustness["primary_uncertainty_by_cluster"]["document_cluster"][
                "confidence_intervals"
            ]["R2"][0]
        ),
        "document_cluster_r2_upper": float(
            robustness["primary_uncertainty_by_cluster"]["document_cluster"][
                "confidence_intervals"
            ]["R2"][1]
        ),
        "document_centered_count": int(
            robustness["source_centered_ordering"]["document"]["n"]
        ),
        "document_centered_group_count": int(
            robustness["source_centered_ordering"]["document"]["n_groups"]
        ),
        "document_centered_r2": float(
            robustness["source_centered_ordering"]["document"]["centered_R2"]
        ),
        "document_centered_spearman": float(
            robustness["source_centered_ordering"]["document"][
                "centered_spearman_rho"
            ]
        ),
        "assay_centered_count": int(
            robustness["source_centered_ordering"]["assay"]["n"]
        ),
        "assay_centered_group_count": int(
            robustness["source_centered_ordering"]["assay"]["n_groups"]
        ),
        "assay_centered_r2": float(
            robustness["source_centered_ordering"]["assay"]["centered_R2"]
        ),
        "assay_centered_spearman": float(
            robustness["source_centered_ordering"]["assay"]["centered_spearman_rho"]
        ),
        "earlier_broad_record_count": int(
            robustness["broad_record_chronology"]["count"]
        ),
        "strict_chronology_sensitivity_count": int(
            robustness["broad_record_chronology"][
                "remaining_strict_chronology_count"
            ]
        ),
        "strict_chronology_sensitivity_rmse": float(
            robustness["broad_record_chronology"]["sensitivity_point_metrics"][
                "RMSE"
            ]
        ),
        "strict_chronology_sensitivity_r2": float(
            robustness["broad_record_chronology"]["sensitivity_point_metrics"][
                "R2"
            ]
        ),
        "nearest_neighbor_gap_error_rho": float(
            robustness["nearest_neighbor_potency_discordance"]["point_estimate"]
        ),
        "nearest_neighbor_gap_error_lower": float(
            robustness["nearest_neighbor_potency_discordance"]["ci_lower"]
        ),
        "nearest_neighbor_gap_error_upper": float(
            robustness["nearest_neighbor_potency_discordance"]["ci_upper"]
        ),
        "assay_context_audit": assay_context,
        "gnn_training_duration_audit": gnn_training,
        "tuned_gnn_best_family": str(tuned_best["family"]),
        "tuned_gnn_best_model_text": {
            "edge_aware_gine": "nested-tuned edge-aware GINE ensemble",
            "chemprop_dmpnn": "nested-tuned Chemprop D-MPNN ensemble",
            "chemprop_chemeleon": "nested-tuned frozen CheMeleon ensemble",
        }[str(tuned_best["family"])],
        "tuned_gnn_best_rmse": float(tuned_best["rmse_mean"]),
        "tuned_gnn_best_rmse_sd": float(tuned_best["rmse_sd"]),
        "tuned_gnn_best_delta": float(tuned_best_delta["point_difference"]),
        "tuned_gnn_best_delta_lower": float(tuned_best_delta["ci_lower"]),
        "tuned_gnn_best_delta_upper": float(tuned_best_delta["ci_upper"]),
        "tuned_gnn_best_mean_prediction_seed_sd": float(
            tuned_best["mean_seed_prediction_sd"]
        ),
        "tuned_gnn_best_within_repeat_seed_rmse_sd": (
            tuned_best_within_repeat_seed_rmse_sd
        ),
        "tuned_gnn_candidate_budget": int(
            tuned_gnn_manifest["candidate_budget_per_family_per_outer_fold"]
        ),
        "tuned_gnn_random_draw_count": int(
            tuned_gnn_manifest["candidate_budget_per_family_per_outer_fold"] - 1
        ),
        "tuned_gnn_rerank_count": int(tuned_gnn_manifest["rerank_top_candidates"]),
        "tuned_gnn_seed_count": len(tuned_gnn_manifest["outer_refit_seeds"]),
        "tuned_gnn_screening_evaluations": int(
            tuned_gnn_manifest["candidate_budget_per_family_per_outer_fold"] * 45
        ),
        "tuned_gnn_inner_fits": int(
            (
                tuned_gnn_manifest["candidate_budget_per_family_per_outer_fold"]
                + tuned_gnn_manifest["rerank_top_candidates"]
                * (len(tuned_gnn_manifest["rerank_seeds"]) - 1)
            )
            * 45
            * 3
        ),
        "calibration_slope": float(diagnostics["calibration"]["observed_on_prediction_slope"]),
        "calibration_intercept": float(diagnostics["calibration"]["observed_on_prediction_intercept"]),
        "interval_width": float(diagnostics["conformal"]["mean_interval_width"]),
        "interval_fold_span": float(
            10 ** float(diagnostics["conformal"]["mean_interval_width"])
        ),
        "temporal_coverage_percent": 100.0
        * float(diagnostics["conformal"]["empirical_temporal_coverage"]),
        "coverage_lower_percent": 100.0
        * float(diagnostics["conformal"]["coverage_cluster_bootstrap_ci"][0]),
        "coverage_upper_percent": 100.0
        * float(diagnostics["conformal"]["coverage_cluster_bootstrap_ci"][1]),
        "outside_ad_count": int(diagnostics["applicability_domain"]["outside_count"]),
        "similarity_error_rho": float(
            diagnostics["applicability_domain"]["absolute_error_spearman_rho"]
        ),
        "new_scaffold_rmse": float(novelty["new_scaffold"]["rmse"]),
        "seen_scaffold_rmse": float(novelty["seen_scaffold"]["rmse"]),
        "inside_ad_rmse": float(ad_groups["inside"]["rmse"]),
        "outside_ad_rmse": float(ad_groups["outside"]["rmse"]),
        "temporal_high_tail_removed": int(
            diagnostics["prespecified_high_tail_sensitivity"]["removed_compounds"]
        ),
        "no_high_tail_temporal_rmse": float(
            diagnostics["prespecified_high_tail_sensitivity"]["metrics"]["point_estimates"]["RMSE"]
        ),
        "classical_python_version": str(classical_environment["python"]).split()[0],
        "rdkit_version": str(classical_environment["packages"]["rdkit"]),
        "numpy_version": str(classical_environment["packages"]["numpy"]),
        "pandas_version": str(classical_environment["packages"]["pandas"]),
        "sklearn_version": str(classical_environment["packages"]["scikit-learn"]),
        "torch_version": str(gnn_environment["packages"]["torch"]),
        "pyg_version": str(gnn_environment["packages"]["torch-geometric"]),
        "chemprop_version": str(gnn_environment["packages"]["chemprop"]),
    }


def write_macros(results: Mapping[str, Any]) -> None:
    assay = results["assay_context_audit"]
    gnn_training = results["gnn_training_duration_audit"]
    matched_atp = assay["matched_numeric_atp_pairs"][0]
    definitions = {
        "SnapshotActivityCount": format_int(results["snapshot_activity_count"]),
        "SnapshotAssayCount": format_int(results["snapshot_assay_count"]),
        "SnapshotParentCount": format_int(results["snapshot_parent_count"]),
        "BroadEligibleCount": format_int(results["broad_eligible_count"]),
        "BroadEventCount": format_int(results["broad_event_count"]),
        "BroadCompoundCount": format_int(results["broad_compound_count"]),
        "StrictEventCount": format_int(results["strict_event_count"]),
        "StrictCompoundCount": format_int(results["strict_compound_count"]),
        "NoHighTailEventCount": format_int(results["no_high_tail_event_count"]),
        "NoHighTailCompoundCount": format_int(results["no_high_tail_compound_count"]),
        "HomologousExcludedCount": format_int(results["homologous_excluded_count"]),
        "StructureFailureCount": format_int(results["structure_failure_count"]),
        "MultiEventCompoundCount": format_int(results["multi_event_compound_count"]),
        "DispersionHalfCount": format_int(results["dispersion_half_count"]),
        "DispersionOneCount": format_int(results["dispersion_one_count"]),
        "StrictAssayContextCount": format_int(assay["assay_count"]),
        "StrictDocumentContextCount": format_int(assay["document_count"]),
        "StrictDescriptionContextCount": format_int(assay["description_count"]),
        "AssayEventMedian": fmt(assay["assay_event_count_median"], 0),
        "AssayEventMinimum": format_int(assay["assay_event_count_min"]),
        "AssayEventMaximum": format_int(assay["assay_event_count_max"]),
        "LargestAssayEventPercent": fmt(assay["largest_assay_event_percent"], 1),
        "ATPDescriptionAssayCount": format_int(assay["atp_mention_assay_count"]),
        "ATPDescriptionEventCount": format_int(assay["atp_mention_event_count"]),
        "NumericATPAssayCount": format_int(assay["numeric_atp_assay_count"]),
        "NumericATPEventCount": format_int(assay["numeric_atp_event_count"]),
        "NumericATPEventPercent": fmt(assay["numeric_atp_event_percent"], 1),
        "ATPMinimumMicromolar": fmt(assay["numeric_atp_min_um"], 0),
        "ATPMaximumMicromolar": fmt(assay["numeric_atp_max_um"], 0),
        "ATPFoldRange": fmt(assay["numeric_atp_fold_range"], 0),
        "ATPKmAssayCount": format_int(assay["atp_at_km_assay_count"]),
        "SingleEventCompoundPercent": fmt(assay["single_event_compound_percent"], 1),
        "CrossAssayCompoundCount": format_int(assay["cross_assay_compound_count"]),
        "AssayExclusiveScaffoldEventPercent": fmt(
            assay["assay_exclusive_scaffold_event_percent"], 1
        ),
        "DevelopmentAssayCount": format_int(assay["development_assay_count"]),
        "TemporalAssayCount": format_int(assay["temporal_assay_count"]),
        "DevelopmentDocumentCount": format_int(assay["development_document_count"]),
        "TemporalDocumentCount": format_int(assay["temporal_document_count"]),
        "AssayPartitionOverlapCount": format_int(
            assay["development_temporal_assay_intersection_count"]
        ),
        "DocumentPartitionOverlapCount": format_int(
            assay["development_temporal_document_intersection_count"]
        ),
        "MatchedATPLowMicromolar": fmt(matched_atp["low_atp_um"], 0),
        "MatchedATPHighMicromolar": fmt(matched_atp["high_atp_um"], 0),
        "MatchedATPLowNanomolar": fmt(
            matched_atp["low_atp_standard_value_nm"], 0
        ),
        "MatchedATPHighNanomolar": fmt(
            matched_atp["high_atp_standard_value_nm"], 0
        ),
        "MatchedATPShift": fmt(
            abs(matched_atp["pic50_delta_high_minus_low_atp"])
        ),
        "DevelopmentCount": format_int(results["development_count"]),
        "DevelopmentScaffoldCount": format_int(results["development_scaffold_count"]),
        "TemporalCount": format_int(results["temporal_count"]),
        "TemporalScaffoldCount": format_int(results["temporal_scaffold_count"]),
        "NewTemporalScaffoldCount": format_int(results["new_temporal_scaffold_count"]),
        "NewTemporalCompoundCount": format_int(results["new_temporal_compound_count"]),
        "NewTemporalCompoundPercent": fmt(results["new_temporal_compound_percent"], 1),
        "SeenTemporalCompoundCount": format_int(results["seen_temporal_compound_count"]),
        "CrossCutoffMedianShift": fmt(results["cross_cutoff_median_shift"]),
        "CrossCutoffMaximumShift": fmt(results["cross_cutoff_maximum_shift"]),
        "OuterRepeatCount": format_int(results["outer_repeat_count"]),
        "OuterFoldCount": format_int(results["outer_fold_count"]),
        "SelectedModelText": tex_escape(results["selected_model_text"]),
        "DevelopmentRMSE": fmt(results["development_rmse"]),
        "BestGraphModelText": tex_escape(results["best_graph_model_text"]),
        "BestGraphRMSE": fmt(results["best_graph_rmse"]),
        "TunedGNNBestModelText": tex_escape(results["tuned_gnn_best_model_text"]),
        "TunedGNNBestRMSE": fmt(results["tuned_gnn_best_rmse"]),
        "TunedGNNBestRMSESD": fmt(results["tuned_gnn_best_rmse_sd"]),
        "TunedGNNBestDelta": fmt(results["tuned_gnn_best_delta"]),
        "TunedGNNBestDeltaLower": fmt(results["tuned_gnn_best_delta_lower"]),
        "TunedGNNBestDeltaUpper": fmt(results["tuned_gnn_best_delta_upper"]),
        "TunedGNNBestMeanPredictionSeedSD": fmt(
            results["tuned_gnn_best_mean_prediction_seed_sd"]
        ),
        "TunedGNNBestWithinRepeatSeedRMSESD": fmt(
            results["tuned_gnn_best_within_repeat_seed_rmse_sd"]
        ),
        "TunedGNNCandidateBudget": format_int(results["tuned_gnn_candidate_budget"]),
        "TunedGNNRandomDrawCount": format_int(results["tuned_gnn_random_draw_count"]),
        "TunedGNNRerankCount": format_int(results["tuned_gnn_rerank_count"]),
        "TunedGNNSeedCount": format_int(results["tuned_gnn_seed_count"]),
        "TunedGNNScreeningEvaluations": format_int(
            results["tuned_gnn_screening_evaluations"]
        ),
        "TunedGNNInnerFitCount": format_int(results["tuned_gnn_inner_fits"]),
        "BestGraphDelta": fmt(results["best_graph_delta"]),
        "BestGraphDeltaLower": fmt(results["best_graph_delta_lower"]),
        "BestGraphDeltaUpper": fmt(results["best_graph_delta_upper"]),
        "RunnerUpModelText": tex_escape(results["runner_up_model_text"]),
        "RunnerUpRMSE": fmt(results["runner_up_rmse"]),
        "RunnerUpDelta": fmt(results["runner_up_delta"]),
        "RunnerUpDeltaLower": fmt(results["runner_up_delta_lower"]),
        "RunnerUpDeltaUpper": fmt(results["runner_up_delta_upper"]),
        "DescriptorAblationConclusion": tex_escape(results["descriptor_ablation_conclusion"]),
        "GNNCheckpointCount": format_int(gnn_training["checkpoint_count"]),
        "GNNInnerEpochCount": format_int(gnn_training["inner_epoch_count"]),
        "GNNInnerEpochMinimum": format_int(gnn_training["inner_epoch_min"]),
        "GNNInnerEpochMaximum": format_int(gnn_training["inner_epoch_max"]),
        "GNNRefitEpochCount": format_int(gnn_training["refit_epoch_count"]),
        "GNNRefitEpochMinimum": format_int(gnn_training["refit_epoch_min"]),
        "GNNRefitEpochMaximum": format_int(gnn_training["refit_epoch_max"]),
        "GNNEpochCap": format_int(gnn_training["max_epochs"]),
        "GNNEpochCapHitCount": format_int(
            gnn_training["total_epoch_cap_hit_count"]
        ),
        "ModelComparisonConclusion": tex_escape(results["model_comparison_conclusion"]),
        "DiscussionModelFinding": tex_escape(results["discussion_model_finding"]),
        "ConclusionFinding": tex_escape(results["conclusion_finding"]),
        "TemporalRMSE": fmt(results["temporal_rmse"]),
        "TemporalRMSELower": fmt(results["temporal_rmse_lower"]),
        "TemporalRMSEUpper": fmt(results["temporal_rmse_upper"]),
        "TemporalMAE": fmt(results["temporal_mae"]),
        "TemporalMAELower": fmt(results["temporal_mae_lower"]),
        "TemporalMAEUpper": fmt(results["temporal_mae_upper"]),
        "TemporalRtwo": fmt(results["temporal_r2"]),
        "TemporalRtwoLower": fmt(results["temporal_r2_lower"]),
        "TemporalRtwoUpper": fmt(results["temporal_r2_upper"]),
        "TemporalGeneralizationGap": fmt(results["temporal_generalization_gap"]),
        "BetweenYearVariancePercent": fmt(
            results["between_year_variance_percent"], 1
        ),
        "WithinYearCenteredRtwoAll": fmt(
            results["within_year_centered_r2_all"]
        ),
        "LaterCohortCenteredRtwo": fmt(
            results["within_year_centered_r2_2020_2025"]
        ),
        "LaterCohortCenteredSpearman": fmt(
            results["within_year_centered_spearman_2020_2025"]
        ),
        "TemporalRMSETwentyNineteen": fmt(results["temporal_rmse_2019"]),
        "TemporalRMSETwentyTwentyFour": fmt(results["temporal_rmse_2024"]),
        "TemporalRMSEIncrease": fmt(results["temporal_rmse_increase"]),
        "TemporalRMSEIncreasePercent": fmt(
            results["temporal_rmse_increase_percent"], 1
        ),
        "TemporalRMSEIncreaseScaffoldLower": fmt(
            results["temporal_rmse_increase_scaffold_lower"]
        ),
        "TemporalRMSEIncreaseScaffoldUpper": fmt(
            results["temporal_rmse_increase_scaffold_upper"]
        ),
        "AbsoluteErrorSlopePerYear": fmt(results["absolute_error_slope_per_year"]),
        "AbsoluteErrorSlopeScaffoldLower": fmt(
            results["absolute_error_slope_scaffold_lower"]
        ),
        "AbsoluteErrorSlopeScaffoldUpper": fmt(
            results["absolute_error_slope_scaffold_upper"]
        ),
        "DocumentClusterRMSELower": fmt(results["document_cluster_rmse_lower"]),
        "DocumentClusterRMSEUpper": fmt(results["document_cluster_rmse_upper"]),
        "DocumentClusterRtwoLower": fmt(results["document_cluster_r2_lower"]),
        "DocumentClusterRtwoUpper": fmt(results["document_cluster_r2_upper"]),
        "DocumentCenteredCount": format_int(results["document_centered_count"]),
        "DocumentCenteredGroupCount": format_int(
            results["document_centered_group_count"]
        ),
        "DocumentCenteredRtwo": fmt(results["document_centered_r2"]),
        "DocumentCenteredSpearman": fmt(results["document_centered_spearman"]),
        "AssayCenteredCount": format_int(results["assay_centered_count"]),
        "AssayCenteredGroupCount": format_int(results["assay_centered_group_count"]),
        "AssayCenteredRtwo": fmt(results["assay_centered_r2"]),
        "AssayCenteredSpearman": fmt(results["assay_centered_spearman"]),
        "EarlierBroadRecordCount": format_int(results["earlier_broad_record_count"]),
        "StrictChronologySensitivityCount": format_int(
            results["strict_chronology_sensitivity_count"]
        ),
        "StrictChronologySensitivityRMSE": fmt(
            results["strict_chronology_sensitivity_rmse"]
        ),
        "StrictChronologySensitivityRtwo": fmt(
            results["strict_chronology_sensitivity_r2"]
        ),
        "NearestNeighborGapErrorRho": fmt(results["nearest_neighbor_gap_error_rho"]),
        "NearestNeighborGapErrorLower": fmt(
            results["nearest_neighbor_gap_error_lower"]
        ),
        "NearestNeighborGapErrorUpper": fmt(
            results["nearest_neighbor_gap_error_upper"]
        ),
        "CalibrationSlope": fmt(results["calibration_slope"]),
        "CalibrationIntercept": fmt(results["calibration_intercept"]),
        "IntervalWidth": fmt(results["interval_width"]),
        "IntervalFoldSpan": fmt(results["interval_fold_span"], 0),
        "TemporalCoveragePercent": fmt(results["temporal_coverage_percent"], 1),
        "CoverageLowerPercent": fmt(results["coverage_lower_percent"], 1),
        "CoverageUpperPercent": fmt(results["coverage_upper_percent"], 1),
        "OutsideADCount": format_int(results["outside_ad_count"]),
        "SimilarityErrorRho": fmt(results["similarity_error_rho"]),
        "InsideADRMSE": fmt(results["inside_ad_rmse"]),
        "OutsideADRMSE": fmt(results["outside_ad_rmse"]),
        "NewScaffoldRMSE": fmt(results["new_scaffold_rmse"]),
        "SeenScaffoldRMSE": fmt(results["seen_scaffold_rmse"]),
        "TemporalHighTailRemoved": format_int(results["temporal_high_tail_removed"]),
        "NoHighTailTemporalRMSE": fmt(results["no_high_tail_temporal_rmse"]),
        "ClassicalPythonVersion": tex_escape(results["classical_python_version"]),
        "RDKitVersion": tex_escape(results["rdkit_version"]),
        "NumPyVersion": tex_escape(results["numpy_version"]),
        "PandasVersion": tex_escape(results["pandas_version"]),
        "SklearnVersion": tex_escape(results["sklearn_version"]),
        "TorchVersion": tex_escape(results["torch_version"]),
        "PyGVersion": tex_escape(results["pyg_version"]),
        "ChempropVersion": tex_escape(results["chemprop_version"]),
    }
    lines = [
        "% Generated by scripts/08_build_manuscript_assets.py; do not edit by hand."
    ]
    lines.extend(
        rf"\newcommand{{\{name}}}{{{value}}}" for name, value in definitions.items()
    )
    write_text_atomic(GENERATED_DIR / "results_macros.tex", "\n".join(lines) + "\n")


def latex_table_model_comparison() -> str:
    table = pd.read_csv(SELECTION_DIR / "model_summary.csv").sort_values("rmse_mean")
    selected = load_json(SELECTION_DIR / "model_selection.json")["selected_model"]
    rows = []
    for row in table.itertuples(index=False):
        label = tex_escape(friendly_model(row.model))
        if row.model == selected:
            label = rf"\textbf{{{label}}}"
        values = [
            label,
            tex_escape(REPRESENTATION_LABELS[row.model]),
            f"{fmt(row.rmse_mean)} $\\pm$ {fmt(row.rmse_sd)}",
            f"{fmt(row.mae_mean)} $\\pm$ {fmt(row.mae_sd)}",
            f"{fmt(row.r2_mean)} $\\pm$ {fmt(row.r2_sd)}",
        ]
        rows.append(" & ".join(values) + r" \\")
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Development performance on identical repeated scaffold-disjoint outer folds.}",
            r"\label{tab:model-comparison}",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{3.5pt}",
            r"\begin{tabular}{L{0.29\textwidth}L{0.21\textwidth}R{0.14\textwidth}R{0.14\textwidth}R{0.14\textwidth}}",
            r"\toprule",
            r"\textbf{Candidate} & \textbf{Representation} & \textbf{RMSE} & \textbf{MAE} & \textbf{$R^2$} \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\par\vspace{3pt}\parbox{0.94\textwidth}{\footnotesize \textit{Note.} Values are mean $\pm$ population SD across three pooled out-of-fold repeat metrics. The bold row was selected using mean RMSE only.}",
            r"\end{table}",
            "",
        ]
    )


def latex_table_tuned_gnn() -> str:
    manifest = load_json(TUNED_GNN_DIR / "run_manifest.json")
    summary = pd.read_csv(TUNED_GNN_DIR / "model_summary.csv").sort_values(
        ["rmse_mean", "family"]
    )
    bootstrap = pd.read_csv(TUNED_GNN_DIR / "paired_scaffold_bootstrap.csv")
    labels = {
        "edge_aware_gine": "Tuned edge-aware GINE",
        "chemprop_dmpnn": "Tuned Chemprop D-MPNN",
        "chemprop_chemeleon": "Frozen CheMeleon + tuned head",
    }
    initialization = {
        "edge_aware_gine": "random",
        "chemprop_dmpnn": "random",
        "chemprop_chemeleon": "pretrained",
    }
    rows: list[str] = []
    for row in summary.itertuples(index=False):
        delta = bootstrap.loc[
            bootstrap["family"].eq(row.family)
            & bootstrap["metric"].eq("RMSE")
        ].iloc[0]
        rows.append(
            " & ".join(
                [
                    tex_escape(labels[row.family]),
                    initialization[row.family],
                    f"{fmt(row.rmse_mean)} $\\pm$ {fmt(row.rmse_sd)}",
                    f"{fmt(row.mae_mean)} $\\pm$ {fmt(row.mae_sd)}",
                    f"{fmt(row.r2_mean)} $\\pm$ {fmt(row.r2_sd)}",
                    (
                        f"{fmt(delta['point_difference'])} "
                        f"[{fmt(delta['ci_lower'])}, {fmt(delta['ci_upper'])}]"
                    ),
                ]
            )
            + r" \\"
        )
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Post hoc development-only nested graph-model search.}",
            r"\label{tab:tuned-gnn}",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{2.5pt}",
            r"\begin{tabular}{L{0.24\textwidth}L{0.10\textwidth}R{0.13\textwidth}R{0.13\textwidth}R{0.13\textwidth}R{0.20\textwidth}}",
            r"\toprule",
            r"\textbf{Family} & \textbf{Encoder} & \textbf{RMSE} & \textbf{MAE} & \textbf{$R^2$} & \textbf{$\Delta$RMSE vs Ridge [95\% CI]} \\ ",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            f"\\par\\vspace{{3pt}}\\parbox{{0.97\\textwidth}}{{\\footnotesize \\textit{{Note.}} Values are mean $\\pm$ population SD across three complete outer-CV repeats. Within every outer fold, each family received one anchored baseline plus {int(manifest['candidate_budget_per_family_per_outer_fold']) - 1} deterministic random draws over architecture, optimization, representation, and regularization; the top {int(manifest['rerank_top_candidates'])} were reranked over {len(manifest['rerank_seeds'])} initialization seeds. Outer predictions average {len(manifest['outer_refit_seeds'])} refits. Confidence intervals are paired 10,000-replicate scaffold-cluster bootstraps after averaging predictions across repeats; they are family-specific and not multiplicity-adjusted for choosing the lowest-RMSE family. This analysis was designed after the temporal results had been inspected and was not eligible to alter the locked temporal analysis.}}",
            r"\end{table}",
            "",
        ]
    )


def latex_table_tuned_gnn_selections() -> str:
    selections = pd.read_csv(TUNED_GNN_DIR / "selected_hyperparameters.csv")
    families = ["edge_aware_gine", "chemprop_dmpnn", "chemprop_chemeleon"]

    def counts(series: pd.Series) -> str:
        observed = series.astype(str).value_counts().sort_index()
        return "; ".join(
            f"{tex_escape(key.replace('_', ' '))}: {int(value)}"
            for key, value in observed.items()
        )

    def numeric_range(series: pd.Series, digits: int = 3) -> str:
        values = pd.to_numeric(series)
        return f"{fmt(values.median(), digits)} [{fmt(values.min(), digits)}, {fmt(values.max(), digits)}]"

    row_specs = [
        ("Descriptor fusion (yes/no)", lambda x: counts(x["representation"].eq("graph_plus_rdkit2d").map({True: "yes", False: "no"}))),
        ("Depth, median [range]", lambda x: numeric_range(x["depth"], 0)),
        ("Message width, median [range]", lambda x: numeric_range(x["hidden_dim"], 0)),
        ("Head width, median [range]", lambda x: numeric_range(x["ffn_hidden_dim"], 0)),
        ("Head layers, median [range]", lambda x: numeric_range(x["ffn_layers"], 0)),
        ("Dropout, median [range]", lambda x: numeric_range(x["dropout"])),
        ("Learning rate, median [range]", lambda x: numeric_range(x["learning_rate"], 6)),
        ("Nonzero weight decay", lambda x: f"{int(pd.to_numeric(x['weight_decay']).gt(0).sum())}/15"),
        ("Batch-size selections", lambda x: counts(x["batch_size"])),
        ("Pooling selections", lambda x: counts(x["pooling"])),
        ("Normalization selections", lambda x: counts(x["normalization"])),
        (
            "Residual updates (yes/no)",
            lambda x: counts(
                x["residual"]
                .astype(str)
                .str.lower()
                .map({"true": "yes", "false": "no"})
            ),
        ),
        ("Initialization selections", lambda x: counts(x["initialization"])),
        ("Activation selections", lambda x: counts(x["activation"])),
        ("Featurization selections", lambda x: counts(x["featurization"])),
    ]
    rows: list[str] = []
    for row_label, summarizer in row_specs:
        values = [
            summarizer(selections.loc[selections["family"].eq(family)])
            for family in families
        ]
        rows.append(
            " & ".join([tex_escape(row_label), *values]) + r" \\"
        )
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Hyperparameters selected independently in the 15 outer folds of each post hoc graph family.}",
            r"\label{tab:tuned-gnn-selections}",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{3pt}",
            r"\begin{tabular}{L{0.25\textwidth}L{0.22\textwidth}L{0.22\textwidth}L{0.22\textwidth}}",
            r"\toprule",
            r"\textbf{Selection summary} & \textbf{GINE} & \textbf{D-MPNN} & \textbf{CheMeleon} \\ ",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\par\vspace{3pt}\parbox{0.95\textwidth}{\footnotesize \textit{Note.} Counts are out of 15 outer folds. Numeric entries are median [minimum, maximum]. CheMeleon depth, message width, mean pooling, and v2 featurization were fixed by the pretrained encoder contract; its downstream head and descriptor fusion were tuned. Its initialization entry denotes the frozen pretrained encoder; each downstream head used seeded Xavier initialization.}",
            r"\end{table}",
            "",
        ]
    )


def latex_table_tuned_gnn_search_space() -> str:
    """Render the exact tuned-GNN candidate distributions and anchored baselines."""

    config = yaml.safe_load(TUNED_GNN_SEARCH_CONFIG.read_text(encoding="utf-8"))
    families = config["families"]
    family_order = ("edge_aware_gine", "chemprop_dmpnn", "chemprop_chemeleon")
    budget = int(config["candidate_budget_per_family"])

    def values(family: str, key: str, *, fixed: str = "---") -> str:
        raw = families[family].get(key)
        if raw is None:
            return fixed
        labels = {
            "graph": "graph",
            "graph_plus_rdkit2d": "graph + RDKit2D",
            "mean": "mean",
            "add": "sum",
            "sum": "sum",
            "norm": "size-normalized sum",
            "none": "none",
            "batch": "batch",
            "layer": "layer",
            "relu": "ReLU",
            "leakyrelu": "leaky ReLU",
            "elu": "ELU",
            "chemprop_v1": "Chemprop v1",
            "chemprop_v2": "Chemprop v2",
            "chemprop_organic": "Chemprop organic",
            "rich_atom_bond_v1": "rich atom/bond v1",
            "xavier_uniform": "Xavier uniform",
            "kaiming_uniform": "Kaiming uniform",
            "frozen": "frozen",
        }

        def scalar(value: Any) -> str:
            if isinstance(value, bool):
                return "yes" if value else "no"
            if isinstance(value, str):
                return labels.get(value, value.replace("_", " "))
            number = float(value)
            if number == 0:
                return "0"
            if abs(number) <= 0.001:
                exponent = int(math.floor(math.log10(abs(number))))
                coefficient = number / (10**exponent)
                prefix = "" if abs(coefficient - 1.0) < 1e-12 else f"{coefficient:g}\\times"
                return f"${prefix}10^{{{exponent}}}$"
            return f"{number:g}"

        return ", ".join(scalar(value) for value in raw)

    rows = [
        (
            "Candidate construction",
            f"1 baseline + {budget - 1} draws; seed 62017",
            f"1 baseline + {budget - 1} draws; seed 63026",
            f"1 baseline + {budget - 1} draws; seed 64035",
        ),
        ("Representation", *(values(family, "representation") for family in family_order)),
        ("Encoder depth", values("edge_aware_gine", "depth"), values("chemprop_dmpnn", "depth"), "6 (fixed)"),
        ("Message width", values("edge_aware_gine", "hidden_dim"), values("chemprop_dmpnn", "hidden_dim"), "2,048 (fixed)"),
        ("Head width", *(values(family, "ffn_hidden_dim") for family in family_order)),
        ("Head layers", *(values(family, "ffn_layers") for family in family_order)),
        ("Dropout", *("continuous uniform [0, 0.4]" for _ in family_order)),
        ("Learning rate", *("log-uniform [$10^{-5}$, $3\\times10^{-3}$]" for _ in family_order)),
        ("AdamW weight decay", *(values(family, "weight_decay") for family in family_order)),
        ("Batch size", *(values(family, "batch_size") for family in family_order)),
        ("Pooling", values("edge_aware_gine", "pooling"), values("chemprop_dmpnn", "pooling"), "mean (fixed)"),
        ("Normalization", *(values(family, "normalization") for family in family_order)),
        ("Residual updates", values("edge_aware_gine", "residual"), "no (fixed)", "no (fixed)"),
        ("Initialization", values("edge_aware_gine", "initialization"), "Chemprop default (seeded)", "pretrained encoder; Xavier head"),
        ("Activation", "ReLU (fixed)", values("chemprop_dmpnn", "activation"), values("chemprop_chemeleon", "activation")),
        ("Atom/bond features", values("edge_aware_gine", "featurization"), values("chemprop_dmpnn", "featurization"), "Chemprop v2 (fixed)"),
        ("Encoder training", "task-trained", "task-trained", "frozen"),
        ("Baseline: representation / encoder / head", "graph + RDKit2D; 3 $\\times$ 128; 2 $\\times$ 128", "graph; 3 $\\times$ 128; 2 $\\times$ 128", "graph; 6 $\\times$ 2,048 frozen; 2 $\\times$ 128"),
        ("Baseline: dropout / learning rate / weight decay / batch", "0.1 / $3\\times10^{-4}$ / $10^{-5}$ / 64", "0.1 / $3\\times10^{-4}$ / 0 / 64", "0.1 / $3\\times10^{-4}$ / $10^{-5}$ / 64"),
        ("Baseline: pool / normalization / activation", "mean / batch / ReLU", "mean / batch / ReLU", "mean / batch / ReLU"),
    ]
    body = [
        " & ".join([tex_escape(row[0]), *row[1:]]) + r" \\"
        for row in rows
    ]
    return "\n".join(
        [
            r"\begingroup\scriptsize",
            r"\begin{longtable}{L{0.18\textwidth}L{0.225\textwidth}L{0.225\textwidth}L{0.225\textwidth}}",
            r"\caption{Exact post hoc graph-model candidate spaces and anchored baselines.}\label{tab:tuned-gnn-search-space}\\",
            r"\toprule",
            r"\textbf{Choice} & \textbf{GINE} & \textbf{D-MPNN} & \textbf{Frozen CheMeleon} \\ ",
            r"\midrule",
            r"\endfirsthead",
            r"\multicolumn{4}{l}{\footnotesize\textit{Table \thetable\ continued}}\\",
            r"\toprule",
            r"\textbf{Choice} & \textbf{GINE} & \textbf{D-MPNN} & \textbf{Frozen CheMeleon} \\ ",
            r"\midrule",
            r"\endhead",
            r"\midrule\multicolumn{4}{r}{\footnotesize\textit{Continued on next page}}\\",
            r"\endfoot",
            r"\bottomrule",
            r"\endlastfoot",
            *body,
            f"\\multicolumn{{4}}{{L{{0.84\\textwidth}}}}{{\\footnotesize \\textit{{Note.}} Each family received one anchored baseline plus {budget - 1} deterministic random draws; the same family-specific candidate set was evaluated in every outer fold. Dropout was sampled uniformly and learning rate log-uniformly; other listed multi-value choices were categorical. Candidate hashes broke exact scoring ties. The complete machine-readable contract is \\nolinkurl{{configs/gnn_search_v1.yaml}}.}}\\\\",
            r"\end{longtable}",
            r"\endgroup",
            "",
        ]
    )


def latex_table_temporal() -> str:
    diagnostics = load_json(TEMPORAL_DIR / "temporal_metrics.json")
    primary = diagnostics["primary_metrics"]
    rows: list[tuple[str, int, int, float, float, float]] = []
    predictions = pd.read_csv(TEMPORAL_DIR / "temporal_predictions.csv")
    rows.append(
        (
            "Primary temporal test",
            len(predictions),
            predictions["scaffold_smiles"].nunique(),
            primary["point_estimates"]["RMSE"],
            primary["point_estimates"]["MAE"],
            primary["point_estimates"]["R2"],
        )
    )
    for row in diagnostics["scaffold_novelty"]["group_metrics"]:
        rows.append(
            (
                "New scaffold" if row["group"] == "new_scaffold" else "Seen scaffold",
                row["n"],
                row["n_scaffolds"],
                row["rmse"],
                row["mae"],
                row["r2"],
            )
        )
    for row in diagnostics["applicability_domain"]["group_metrics"]:
        rows.append(
            (
                "Inside training-defined AD" if row["group"] == "inside" else "Outside training-defined AD",
                row["n"],
                row["n_scaffolds"],
                row["rmse"],
                row["mae"],
                row["r2"],
            )
        )
    sensitivity = diagnostics["prespecified_high_tail_sensitivity"]
    sensitivity_predictions = predictions.loc[predictions["y_true"].le(10.0)]
    values = sensitivity["metrics"]["point_estimates"]
    rows.append(
        (
            r"Sensitivity: observed pIC$_{50}\leq 10$",
            len(sensitivity_predictions),
            sensitivity_predictions["scaffold_smiles"].nunique(),
            values["RMSE"],
            values["MAE"],
            values["R2"],
        )
    )
    body = [
        f"{label} & {n:,} & {scaffolds:,} & {fmt(rmse)} & {fmt(mae)} & {fmt(r2)} \\\\"
        for label, n, scaffolds, rmse, mae, r2 in rows
    ]
    ci = primary["confidence_intervals"]
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{One-time temporal performance and prespecified diagnostics.}",
            r"\label{tab:temporal}",
            r"\small",
            r"\begin{tabular}{L{0.40\textwidth}rrrrr}",
            r"\toprule",
            r"\textbf{Subset} & \textbf{$n$} & \textbf{Scaffolds} & \textbf{RMSE} & \textbf{MAE} & \textbf{$R^2$} \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\par\vspace{3pt}\parbox{0.94\textwidth}{\footnotesize \textit{Note.} Primary 95\% scaffold-cluster bootstrap CIs: RMSE "
                + f"{fmt(ci['RMSE'][0])}--{fmt(ci['RMSE'][1])}; MAE "
                + f"{fmt(ci['MAE'][0])}--{fmt(ci['MAE'][1])}; "
                + f"$R^2$ {fmt(ci['R2'][0])}--{fmt(ci['R2'][1])}. "
                + r"Subgroup rows are descriptive and are not model-selection criteria. "
                + r"Subgroup $R^2$ values use subgroup-specific outcome variance and should not be interpreted as error-stratification evidence.}"
            ),
            r"\end{table}",
            "",
        ]
    )


def latex_table_temporal_robustness(results: Mapping[str, Any]) -> str:
    audit = results["temporal_robustness_audit"]
    cluster_rows = []
    labels = (
        ("scaffold_smiles", "Bemis--Murcko scaffold"),
        ("assay_cluster", "Assay-source set"),
        ("document_cluster", "Document-source set"),
    )
    for key, label in labels:
        uncertainty = audit["primary_uncertainty_by_cluster"][key]
        degradation = audit["temporal_degradation"][key]["rmse_delta"]
        intervals = uncertainty["confidence_intervals"]
        cluster_rows.append(
            f"{label} & {uncertainty['n_clusters']:,} & "
            f"{fmt(intervals['RMSE'][0])}--{fmt(intervals['RMSE'][1])} & "
            f"{fmt(intervals['R2'][0])}--{fmt(intervals['R2'][1])} & "
            f"{fmt(degradation['ci_lower'])}--{fmt(degradation['ci_upper'])} \\\\"
        )
    chronology = audit["broad_record_chronology"]
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Post hoc robustness audits of the immutable temporal predictions.}",
            r"\label{tab:temporal-robustness}",
            r"\small",
            r"\begin{tabular}{L{0.28\textwidth}rrrr}",
            r"\toprule",
            r"\textbf{Resampling unit} & \textbf{Clusters} & \textbf{RMSE 95\% CI} & \textbf{$R^2$ 95\% CI} & \textbf{$\Delta$RMSE 95\% CI} \\",
            r"\midrule",
            *cluster_rows,
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\par\vspace{3pt}\parbox{0.94\textwidth}{\footnotesize \textit{Note.} "
                r"All intervals use 10,000 percentile-bootstrap replicates. $\Delta$RMSE is "
                r"2024 minus 2019. Assay- and document-source sets are sorted combinations "
                r"of the sources contributing to each aggregated compound label; they are "
                r"dependence-sensitivity analyses, not replacements for the prespecified "
                r"scaffold bootstrap. The strict-chronology sensitivity excluded "
                + f"{chronology['count']} compounds with an earlier broad-cohort record and "
                + f"gave RMSE {fmt(chronology['sensitivity_point_metrics']['RMSE'])} and "
                + f"$R^2={fmt(chronology['sensitivity_point_metrics']['R2'])}$.}}"
            ),
            r"\end{table}",
            "",
        ]
    )


def pretty_grid(grid: Mapping[str, Any]) -> str:
    parts = []
    for key, values in grid.items():
        clean_key = key.replace("model__", "")
        rendered = ", ".join("None" if value is None else str(value) for value in values)
        parts.append(f"{clean_key}: [{rendered}]")
    return "; ".join(parts) if parts else "fixed"


def latex_table_hyperparameters(manifests: Mapping[str, Mapping[str, Any]]) -> str:
    candidates = manifests["classical"]["candidates"]
    gnn_config = manifests["gnn"]["config"]
    rows = []
    for candidate in candidates:
        rows.append(
            f"{tex_escape(friendly_model(candidate['name']))} & "
            f"{tex_escape(REPRESENTATION_LABELS[candidate['name']])} & "
            f"{tex_escape(pretty_grid(candidate['param_grid']))} \\\\"
        )
    gnn_grid = (
        f"hidden={gnn_config['hidden_dim']}; depth={gnn_config['depth']}; "
        f"dropout={gnn_config['dropout']}; max epochs={gnn_config['max_epochs']}; "
        f"patience={gnn_config['patience']}"
    )
    for model in (
        "edge_aware_gine_graph",
        "edge_aware_gine_graph_plus_rdkit2d",
        "chemprop_dmpnn_graph",
        "chemprop_dmpnn_graph_plus_rdkit2d",
    ):
        rows.append(
            f"{tex_escape(friendly_model(model))} & {tex_escape(REPRESENTATION_LABELS[model])} & "
            f"{tex_escape(gnn_grid)} \\\\"
        )
    return "\n".join(
        [
            r"\begin{longtable}{L{0.25\textwidth}L{0.19\textwidth}L{0.48\textwidth}}",
            r"\caption{Prespecified candidate representations and search spaces.}\label{tab:hyperparameters}\\",
            r"\toprule",
            r"\textbf{Candidate} & \textbf{Representation} & \textbf{Settings or inner grid} \\",
            r"\midrule",
            r"\endfirsthead",
            r"\toprule",
            r"\textbf{Candidate} & \textbf{Representation} & \textbf{Settings or inner grid} \\",
            r"\midrule",
            r"\endhead",
            *rows,
            r"\bottomrule",
            r"\end{longtable}",
            "",
        ]
    )


def latex_table_curation_exclusions(manifests: Mapping[str, Mapping[str, Any]]) -> str:
    counts = manifests["curation"]["exclusion_reason_counts"]
    rows = []
    for reason, count in sorted(counts.items(), key=lambda item: (-int(item[1]), item[0])):
        cohort, label = reason.split(":", 1)
        rows.append(
            f"{tex_escape(cohort)} & {tex_escape(label.replace('_', ' '))} & {format_int(count)} \\\\"
        )
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Logged record-level and sensitivity-cohort exclusions.}",
            r"\label{tab:curation-exclusions}",
            r"\small",
            r"\begin{tabular}{L{0.20\textwidth}L{0.56\textwidth}r}",
            r"\toprule",
            r"\textbf{Cohort} & \textbf{Reason} & \textbf{Count} \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def latex_table_repeat_metrics() -> str:
    table = pd.read_csv(SELECTION_DIR / "repeat_metrics.csv")
    table["label"] = table["model"].map(friendly_model)
    rows = []
    for row in table.sort_values(["model", "repeat"]).itertuples(index=False):
        rows.append(
            f"{tex_escape(row.label)} & {int(row.repeat) + 1} & {int(row.n_oof):,} & "
            f"{fmt(row.rmse)} & {fmt(row.mae)} & {fmt(row.r2)} \\\\"
        )
    return "\n".join(
        [
            r"\begingroup",
            r"\small",
            r"\begin{longtable}{L{0.40\textwidth}rrrrr}",
            r"\caption{Complete pooled out-of-fold metrics by repeat.}\label{tab:repeat-metrics}\\",
            r"\toprule",
            r"\textbf{Candidate} & \textbf{Repeat} & \textbf{$n$} & \textbf{RMSE} & \textbf{MAE} & \textbf{$R^2$} \\",
            r"\midrule",
            r"\endfirsthead",
            r"\toprule",
            r"\textbf{Candidate} & \textbf{Repeat} & \textbf{$n$} & \textbf{RMSE} & \textbf{MAE} & \textbf{$R^2$} \\",
            r"\midrule",
            r"\endhead",
            *rows,
            r"\bottomrule",
            r"\end{longtable}",
            r"\endgroup",
            "",
        ]
    )


def latex_table_temporal_by_year() -> str:
    predictions = pd.read_csv(TEMPORAL_DIR / "temporal_predictions.csv")
    drift = temporal_drift_metrics(predictions)
    rows = []
    for row in drift["annual_metrics"]:
        year = f"{row['year']}" + (r"$^\ast$" if row["year"] == 2025 else "")
        rows.append(
            f"{year} & {row['n']:,} & "
            f"{fmt(row['mean_observed'])} ({fmt(row['sd_observed'])}) & "
            f"{fmt(row['mean_predicted'])} & {fmt(row['rmse'])} & "
            f"{fmt(row['mae'])} & {fmt(row['r2'])} & "
            f"{fmt(row['pearson_r'])} & {fmt(row['spearman_rho'])} \\\\"
        )
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Post hoc descriptive temporal performance by first eligible record year. These analyses were not prespecified and were not used for model selection.}",
            r"\label{tab:temporal-year}",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{2.4pt}",
            r"\begin{tabular*}{0.99\textwidth}{@{\extracolsep{\fill}}rrrrrrrrr@{}}",
            r"\toprule",
            r"\textbf{Year} & \textbf{$n$} & \textbf{Observed mean (SD)} & \textbf{Predicted mean} & \textbf{RMSE} & \textbf{MAE} & \textbf{$R^2$} & \textbf{$r$} & \textbf{$\rho_s$} \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular*}",
            (
                r"\par\vspace{2pt}\parbox{0.97\textwidth}{\footnotesize \textit{Note.} "
                r"Observed SD uses the population convention. Pearson $r$ and Spearman $\rho_s$ "
                r"summarize within-cohort association and are unstable in small cohorts. "
                r"$^\ast$The 2025 row contains only four compounds and should not be interpreted independently.}"
            ),
            r"\end{table}",
            "",
        ]
    )


def _format_atp_condition(row: Mapping[str, Any]) -> str:
    concentrations = [float(value) for value in row["atp_concentrations_um"]]
    if not concentrations:
        return r"ATP = $K_m$ (value unstated)"
    labels: list[str] = []
    for value in concentrations:
        if value >= 1000 and value % 1000 == 0:
            labels.append(f"{value / 1000:g} mM")
        else:
            labels.append(f"{value:g} " + r"$\mu$M")
    return ", ".join(labels)


def latex_table_assay_context(results: Mapping[str, Any]) -> str:
    audit = results["assay_context_audit"]
    rows = []
    assay_records = audit["numeric_atp_assays"] + audit["atp_at_km_assays"]
    for row in assay_records:
        documents = ", ".join(row["document_chembl_ids"])
        rows.append(
            f"{tex_escape(row['assay_chembl_id'])} & "
            f"{_format_atp_condition(row)} & {int(row['event_count']):,} & "
            f"{tex_escape(documents)} \\\\"
        )
    pair = audit["matched_numeric_atp_pairs"][0]
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Post hoc audit of strict assays whose pinned descriptions state a numeric ATP concentration or ATP at $K_m$.}",
            r"\label{tab:assay-context}",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{3pt}",
            r"\begin{tabular}{L{0.23\textwidth}L{0.23\textwidth}rL{0.27\textwidth}}",
            r"\toprule",
            r"\textbf{Assay} & \textbf{Reported ATP condition} & \textbf{Events} & \textbf{Document(s)} \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\par\vspace{2pt}\parbox{0.94\textwidth}{\scriptsize \textit{Note.} This text audit is a lower bound on ATP metadata completeness: absence of a machine-readable concentration does not imply that ATP was absent or constant. An exhaustive within-document identity bridge search found one matched compound, in "
            + tex_escape(pair["document_chembl_id"])
            + r", measured at "
            + _format_atp_condition({"atp_concentrations_um": [pair["low_atp_um"]]})
            + r" and "
            + _format_atp_condition({"atp_concentrations_um": [pair["high_atp_um"]]})
            + f" ({fmt(pair['low_atp_standard_value_nm'], 0)} versus "
            + f"{fmt(pair['high_atp_standard_value_nm'], 0)} nM; "
            + f"$\\Delta$pIC$_{{50}}={fmt(pair['pic50_delta_high_minus_low_atp'])}$). "
            + r"This $n=1$ observation is illustrative, not an assay correction or causal estimate \citep{dakin2012pim}.}",
            r"\end{table}",
            "",
        ]
    )


def latex_table_hashes(manifests: Mapping[str, Mapping[str, Any]]) -> str:
    rows = [
        ("ChEMBL snapshot", manifests["snapshot"]["snapshot_id"], manifests["curation"]["source_snapshot"]["snapshot_manifest_sha256"]),
        ("Curated strict dataset", manifests["curation"]["curation_protocol"], manifests["curation"]["curated_dataset_sha256"]),
        ("Immutable split", manifests["split"]["lock_id"], manifests["split"]["split_lock_sha256"]),
        ("Classical benchmark", manifests["classical"]["schema_version"], manifests["classical"]["run_sha256"]),
        ("GNN benchmark contract", manifests["gnn"]["schema_version"], manifests["gnn"]["contract_sha256"]),
        ("Development selection", manifests["selection"]["schema_version"], manifests["selection"]["selection_sha256"]),
        ("Temporal evaluation", manifests["temporal"]["schema_version"], manifests["temporal"]["evaluation_sha256"]),
        (
            "Post hoc robustness audit",
            manifests["robustness_audit_manifest"]["schema_version"],
            manifests["robustness_audit_manifest"]["audit_sha256"],
        ),
        (
            "Post hoc tuned GNN benchmark",
            manifests["tuned_gnn"]["schema_version"],
            manifests["tuned_gnn"]["contract_sha256"],
        ),
    ]
    body = [
        f"{tex_escape(stage)} & {tex_breakable_identifier(identifier)} & \\texttt{{{digest[:16]}...}} \\\\"
        for stage, identifier, digest in rows
    ]
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Primary provenance identifiers and abbreviated SHA-256 values.}",
            r"\label{tab:artifact-hashes}",
            r"\small",
            r"\begin{tabular}{L{0.28\textwidth}L{0.37\textwidth}L{0.25\textwidth}}",
            r"\toprule",
            r"\textbf{Stage} & \textbf{Identifier} & \textbf{SHA-256 prefix} \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(figure: mpl.figure.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        path = FIGURE_DIR / f"{stem}.{suffix}"
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=f".{suffix}", dir=path.parent)
        os.close(fd)
        try:
            metadata = (
                {
                    "Creator": "scripts/08_build_manuscript_assets.py",
                    "Producer": "Matplotlib",
                    "CreationDate": None,
                    "ModDate": None,
                }
                if suffix == "pdf"
                else {"Software": "scripts/08_build_manuscript_assets.py"}
            )
            figure.savefig(
                temporary,
                bbox_inches="tight",
                facecolor="white",
                format=suffix,
                metadata=metadata,
            )
            os.replace(temporary, path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
    plt.close(figure)


def figure_study_design(results: Mapping[str, Any]) -> None:
    """Render a manuscript-native study-design figure with readable labels."""

    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 6)
    axis.axis("off")
    def draw_box(
        x: float,
        y: float,
        width: float,
        height: float,
        title: str,
        subtitle: str,
        color: str,
    ) -> None:
        axis.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.03,rounding_size=0.08",
                linewidth=1.1,
                edgecolor="#404040",
                facecolor=color,
            )
        )
        axis.text(
            x + width / 2,
            y + height * 0.66,
            title,
            ha="center",
            va="center",
            weight="bold",
            fontsize=8.3,
            linespacing=1.05,
        )
        axis.text(
            x + width / 2,
            y + height * 0.28,
            subtitle,
            ha="center",
            va="center",
            fontsize=7.5,
            linespacing=1.05,
        )

    draw_box(
        0.30,
        4.25,
        2.15,
        1.05,
        "ChEMBL 37\nsnapshot",
        f"{results['snapshot_activity_count']:,} activities",
        "#E8F1FA",
    )
    draw_box(
        2.90,
        4.25,
        2.15,
        1.05,
        "Strict direct-assay\ncuration",
        f"{results['strict_event_count']:,} events\n{results['strict_compound_count']:,} compounds",
        "#DCEFE2",
    )
    draw_box(
        5.50,
        4.25,
        2.15,
        1.05,
        "Outcome-blind\nsplit lock",
        "2018 cutoff\ncount rule only",
        "#FFF1CC",
    )
    draw_box(
        2.55,
        2.35,
        2.25,
        1.10,
        "Development",
        f"{results['development_count']:,} compounds\n{results['development_scaffold_count']:,} scaffolds",
        "#E8F1FA",
    )
    draw_box(
        5.15,
        2.35,
        2.25,
        1.10,
        "Shared 3 x 5\nscaffold CV",
        "7 fixed/base + 4 GNN",
        "#E7E2F5",
    )
    draw_box(
        7.75,
        2.35,
        1.85,
        1.10,
        "Sealed temporal\ntest",
        f"{results['temporal_count']:,} compounds\n2019-2025",
        "#F8DEDE",
    )
    draw_box(
        5.15,
        0.55,
        2.25,
        1.10,
        "Locked\nselection",
        "Ridge model",
        "#E7E2F5",
    )
    draw_box(
        7.75,
        0.55,
        1.85,
        1.10,
        "One-time\nevaluation",
        f"temporal RMSE\n{results['temporal_rmse']:.3f}",
        "#F8DEDE",
    )
    arrows = (
        ((2.45, 4.78), (2.90, 4.78)),
        ((5.05, 4.78), (5.50, 4.78)),
        ((6.48, 4.25), (3.67, 3.45)),
        ((4.80, 2.90), (5.15, 2.90)),
        ((6.27, 2.35), (6.27, 1.65)),
        ((7.43, 4.25), (8.68, 3.45)),
        ((7.40, 1.10), (7.75, 1.10)),
        ((8.68, 2.35), (8.68, 1.65)),
    )
    for start, end in arrows:
        axis.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.15,
                color="#555555",
            )
        )
    axis.plot([7.58, 9.78], [1.97, 1.97], linestyle=(0, (4, 3)), color="#A33A3A", linewidth=1.25)
    axis.text(8.68, 2.10, "opened after selection", ha="center", va="bottom", color="#8A2E2E", fontsize=7.5)
    axis.text(0.30, 5.70, "Release-pinned curation", weight="bold", color="#2C4F70", fontsize=9.5)
    axis.text(5.50, 5.70, "Locked information flow", weight="bold", color="#705A20", fontsize=9.5)
    axis.text(0.30, 0.10, "Core inputs and outputs are linked by SHA-256 manifests; temporal labels are opened once after development-only selection.", fontsize=7.4, color="#555555")
    figure.subplots_adjust(left=0.02, right=0.99, top=0.97, bottom=0.04)
    save_figure(figure, "figure1_study_design")


def figure_development_models() -> None:
    summary = pd.read_csv(SELECTION_DIR / "model_summary.csv").sort_values("rmse_mean", ascending=False)
    repeats = pd.read_csv(SELECTION_DIR / "repeat_metrics.csv")
    selected = load_json(SELECTION_DIR / "model_selection.json")["selected_model"]
    colors = {"Baseline": "#8B8B8B", "Classical": "#336699", "Graph neural network": "#B24A3A"}
    figure, axis = plt.subplots(figsize=(7.0, 5.6))
    positions = np.arange(len(summary))
    for position, row in zip(positions, summary.itertuples(index=False)):
        values = repeats.loc[repeats["model"].eq(row.model), "rmse"].to_numpy(dtype=float)
        lower = row.rmse_mean - values.min()
        upper = values.max() - row.rmse_mean
        marker = "D" if row.model == selected else "o"
        axis.errorbar(
            row.rmse_mean,
            position,
            xerr=np.array([[lower], [upper]]),
            fmt=marker,
            markersize=7.3 if row.model == selected else 6.0,
            color=colors[family(row.model)],
            markeredgecolor="black" if row.model == selected else "white",
            markeredgewidth=0.8,
            capsize=3,
            linewidth=1.4,
            zorder=3,
        )
    axis.set_yticks(positions, [friendly_model(model) for model in summary["model"]])
    axis.set_xlabel("Pooled out-of-fold RMSE (pIC$_{50}$; lower is better)")
    axis.set_xlim(0.68, max(1.36, float(summary["rmse_mean"].max()) + 0.08))
    axis.xaxis.grid(True, color="#DDDDDD", linewidth=0.7)
    axis.set_axisbelow(True)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor="white", markersize=7, label=label)
        for label, color in colors.items()
    ]
    handles.append(Line2D([0], [0], marker="D", color="none", markerfacecolor=colors["Classical"], markeredgecolor="black", markersize=7, label="Selected"))
    axis.legend(handles=handles, loc="upper right", frameon=True, ncol=2, fontsize=8.5)
    axis.set_title("Shared 3 x 5 scaffold-disjoint development validation")
    figure.subplots_adjust(left=0.43, right=0.98, top=0.92, bottom=0.12)
    save_figure(figure, "figure2_development_models")


def figure_temporal_evaluation() -> None:
    predictions = pd.read_csv(TEMPORAL_DIR / "temporal_predictions.csv")
    diagnostics = load_json(TEMPORAL_DIR / "temporal_metrics.json")
    robustness = load_json(ROBUSTNESS_AUDIT_DIR / "audit_metrics.json")
    drift = temporal_drift_metrics(predictions)
    annual = pd.DataFrame(drift["annual_metrics"]).sort_values("year")
    figure = plt.figure(figsize=(7.2, 6.0))
    grid = figure.add_gridspec(2, 2, hspace=0.50, wspace=0.39)
    scatter_axis = figure.add_subplot(grid[0, 0])
    rmse_axis = figure.add_subplot(grid[0, 1])
    rank_axis = figure.add_subplot(grid[1, 0])
    ad_axis = figure.add_subplot(grid[1, 1])

    years = annual["year"].to_numpy(dtype=int)
    cohort_sizes = annual["n"].to_numpy(dtype=int)
    cmap = plt.get_cmap("viridis", len(years))
    norm = mpl.colors.BoundaryNorm(
        np.arange(years.min() - 0.5, years.max() + 1.5), cmap.N
    )
    low_n = predictions["first_record_year"].eq(2025)
    pooled = scatter_axis.scatter(
        predictions.loc[~low_n, "y_true"],
        predictions.loc[~low_n, "y_pred"],
        c=predictions.loc[~low_n, "first_record_year"],
        cmap=cmap,
        norm=norm,
        s=18,
        alpha=0.68,
        edgecolors="none",
    )
    if low_n.any():
        scatter_axis.scatter(
            predictions.loc[low_n, "y_true"],
            predictions.loc[low_n, "y_pred"],
            s=30,
            facecolors="none",
            edgecolors=[cmap(norm(2025))],
            linewidths=1.2,
            label="2025 (n=4; low n)",
        )
    limits = [
        float(min(predictions["y_true"].min(), predictions["y_pred"].min()) - 0.25),
        float(max(predictions["y_true"].max(), predictions["y_pred"].max()) + 0.25),
    ]
    scatter_axis.plot(limits, limits, "--", color="#444444", linewidth=1.0)
    scatter_axis.set(
        xlim=limits,
        ylim=limits,
        xlabel="Observed pIC$_{50}$",
        ylabel="Predicted pIC$_{50}$",
        title="a  Pooled performance masks cohort structure",
    )
    primary = diagnostics["primary_metrics"]["point_estimates"]
    scatter_axis.text(
        0.03,
        0.97,
        f"Pooled RMSE = {primary['RMSE']:.3f}\nPooled $R^2$ = {primary['R2']:.3f}",
        transform=scatter_axis.transAxes,
        ha="left",
        va="top",
        fontsize=7.3,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.84, "edgecolor": "#CCCCCC"},
    )
    if low_n.any():
        scatter_axis.legend(loc="lower right", fontsize=6.7, frameon=False)
    colorbar = figure.colorbar(
        pooled,
        ax=scatter_axis,
        ticks=years,
        fraction=0.050,
        pad=0.025,
    )
    colorbar.set_label("First record year", fontsize=7.3)
    colorbar.ax.tick_params(labelsize=6.5, length=2)

    def annual_panel(
        axis: mpl.axes.Axes,
        column: str,
        color: str,
        title: str,
        ylabel: str,
    ) -> None:
        values = annual[column].to_numpy(dtype=float)
        regular = years <= 2024
        axis.plot(
            years[regular],
            values[regular],
            color=color,
            linewidth=1.5,
            zorder=2,
        )
        axis.scatter(
            years[regular],
            values[regular],
            s=34,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        if np.any(years == 2025):
            last_regular = int(np.flatnonzero(regular)[-1])
            low_index = int(np.flatnonzero(years == 2025)[0])
            axis.plot(
                [years[last_regular], years[low_index]],
                [values[last_regular], values[low_index]],
                color=color,
                linestyle="--",
                linewidth=1.0,
                alpha=0.7,
            )
            axis.scatter(
                years[low_index],
                values[low_index],
                s=42,
                facecolor="white",
                edgecolor=color,
                linewidth=1.4,
                zorder=4,
            )
        tick_labels = [
            f"{year}{'*' if year == 2025 else ''}\nn={size}"
            for year, size in zip(years, cohort_sizes)
        ]
        axis.set_xticks(years, tick_labels, fontsize=6.6)
        axis.set(xlabel="First record year", ylabel=ylabel, title=title)
        axis.tick_params(axis="x", length=2)

    annual_panel(
        rmse_axis,
        "rmse",
        "#B24A3A",
        "b  Annual error increases (post hoc)",
        "RMSE (pIC$_{50}$)",
    )
    rmse_axis.text(
        0.03,
        0.96,
        (
            f"2019 to 2024: +{drift['rmse_2019_to_2024_increase']:.3f} "
            f"(+{drift['rmse_2019_to_2024_increase_percent']:.1f}%)\n"
            "scaffold 95% CI "
            f"{robustness['temporal_degradation']['scaffold_smiles']['rmse_delta']['ci_lower']:.3f} to "
            f"{robustness['temporal_degradation']['scaffold_smiles']['rmse_delta']['ci_upper']:.3f}"
        ),
        transform=rmse_axis.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
    )

    annual_panel(
        rank_axis,
        "spearman_rho",
        "#6B4C9A",
        "c  Annual rank association is unstable (post hoc)",
        "Spearman $\\rho_s$",
    )
    rank_axis.axhline(0.0, color="#777777", linewidth=0.8, linestyle="--")
    rank_axis.set_ylim(-1.05, 1.05)
    rank_axis.text(
        0.03,
        0.96,
        f"2020-2025 centered $\\rho_s$ = {drift['within_year_centered_spearman_2020_2025']:.3f}",
        transform=rank_axis.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
    )

    inside = predictions["inside_applicability_domain"].astype(bool)
    ad_axis.scatter(
        predictions.loc[inside, "mean_5nn_tanimoto"],
        predictions.loc[inside, "absolute_error"],
        s=18,
        alpha=0.62,
        color="#336699",
        edgecolor="none",
        label="inside AD",
    )
    ad_axis.scatter(
        predictions.loc[~inside, "mean_5nn_tanimoto"],
        predictions.loc[~inside, "absolute_error"],
        s=18,
        alpha=0.62,
        color="#D07A2D",
        edgecolor="none",
        label="outside AD",
    )
    threshold = diagnostics["applicability_domain"]["mean_similarity_threshold"]
    ad_axis.axvline(threshold, linestyle="--", color="#555555", linewidth=1.0)
    ad_axis.set(
        xlabel="Mean 5-NN Tanimoto similarity",
        ylabel="Absolute error (pIC$_{50}$)",
        title="d  Similarity AD does not stratify absolute error",
    )
    ad_groups = {
        row["group"]: row
        for row in diagnostics["applicability_domain"]["group_metrics"]
    }
    ad_axis.text(
        0.03,
        0.97,
        (
            f"$\\rho_s$ = {diagnostics['applicability_domain']['absolute_error_spearman_rho']:.3f}\n"
            f"RMSE: {ad_groups['inside']['rmse']:.3f} in / "
            f"{ad_groups['outside']['rmse']:.3f} out"
        ),
        transform=ad_axis.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "alpha": 0.84, "edgecolor": "#CCCCCC"},
    )
    ad_axis.legend(loc="upper right", fontsize=6.8, frameon=False)

    for axis in (scatter_axis, rmse_axis, rank_axis, ad_axis):
        axis.grid(True, color="#E2E2E2", linewidth=0.6)
        axis.set_axisbelow(True)
    figure.subplots_adjust(left=0.09, right=0.98, top=0.95, bottom=0.09)
    save_figure(figure, "figure3_temporal_evaluation")


def write_generated_tables(
    manifests: Mapping[str, Mapping[str, Any]], results: Mapping[str, Any]
) -> None:
    tables = {
        "table_model_comparison.tex": latex_table_model_comparison(),
        "table_tuned_gnn.tex": latex_table_tuned_gnn(),
        "table_tuned_gnn_selections.tex": latex_table_tuned_gnn_selections(),
        "table_tuned_gnn_search_space.tex": latex_table_tuned_gnn_search_space(),
        "table_temporal_diagnostics.tex": latex_table_temporal(),
        "table_hyperparameters.tex": latex_table_hyperparameters(manifests),
        "table_curation_exclusions.tex": latex_table_curation_exclusions(manifests),
        "table_repeat_metrics.tex": latex_table_repeat_metrics(),
        "table_temporal_by_year.tex": latex_table_temporal_by_year(),
        "table_assay_context.tex": latex_table_assay_context(results),
        "table_temporal_robustness.tex": latex_table_temporal_robustness(results),
        "table_artifact_hashes.tex": latex_table_hashes(manifests),
    }
    for filename, contents in tables.items():
        write_text_atomic(GENERATED_DIR / filename, contents)


def write_asset_manifest(results: Mapping[str, Any], manifests: Mapping[str, Mapping[str, Any]]) -> None:
    results_path = GENERATED_DIR / "manuscript_results.json"
    write_text_atomic(
        results_path,
        json.dumps(dict(results), indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    output_paths = sorted(
        [path for path in GENERATED_DIR.iterdir() if path.is_file()]
        + [path for path in FIGURE_DIR.iterdir() if path.is_file()]
    )
    input_paths = {
        "snapshot_manifest": SNAPSHOT_DIR / "snapshot_manifest.json",
        "curation_manifest": CURATION_DIR / "curation_manifest.json",
        "strict_measurement_events": CURATION_DIR / "strict_measurement_events.csv",
        "split_lock_manifest": SPLIT_DIR / "split_lock_manifest.json",
        "development_split": SPLIT_DIR / "development.csv",
        "temporal_test_split": SPLIT_DIR / "temporal_test_sealed.csv",
        "classical_run_manifest": CLASSICAL_DIR / "run_manifest.json",
        "gnn_run_manifest": GNN_DIR / "run_manifest.json",
        "tuned_gnn_run_manifest": TUNED_GNN_DIR / "run_manifest.json",
        "tuned_gnn_search_config": TUNED_GNN_SEARCH_CONFIG,
        "tuned_gnn_model_summary": TUNED_GNN_DIR / "model_summary.csv",
        "tuned_gnn_repeat_metrics": TUNED_GNN_DIR / "repeat_metrics.csv",
        "tuned_gnn_seed_metrics": TUNED_GNN_DIR / "seed_metrics.csv",
        "tuned_gnn_selected_hyperparameters": TUNED_GNN_DIR
        / "selected_hyperparameters.csv",
        "tuned_gnn_inner_trial_metrics": TUNED_GNN_DIR / "inner_trial_metrics.csv",
        "tuned_gnn_paired_bootstrap": TUNED_GNN_DIR
        / "paired_scaffold_bootstrap.csv",
        "selection_manifest": SELECTION_DIR / "model_selection.json",
        "temporal_manifest": TEMPORAL_DIR / "temporal_evaluation_manifest.json",
        "temporal_metrics": TEMPORAL_DIR / "temporal_metrics.json",
        "temporal_predictions": TEMPORAL_DIR / "temporal_predictions.csv",
        "robustness_audit_manifest": ROBUSTNESS_AUDIT_DIR / "audit_manifest.json",
        "robustness_audit_metrics": ROBUSTNESS_AUDIT_DIR / "audit_metrics.json",
        "robustness_audit_assignments": ROBUSTNESS_AUDIT_DIR
        / "temporal_audit_assignments.csv",
        "builder": Path(__file__).resolve(),
    }
    for checkpoint_path in sorted((GNN_DIR / "checkpoints").glob("*.json")):
        input_paths[f"gnn_checkpoint_manifest__{checkpoint_path.stem}"] = checkpoint_path
        input_paths[f"gnn_checkpoint_predictions__{checkpoint_path.stem}"] = (
            checkpoint_path.with_suffix(".csv")
        )
    manifest = {
        "schema_version": "manuscript_assets_v2",
        "selected_model": results["selected_model"],
        "temporal_evaluation_sha256": manifests["temporal"]["evaluation_sha256"],
        "input_hashes": {name: sha256_file(path) for name, path in input_paths.items()},
        "output_hashes": {
            str(path.relative_to(MANUSCRIPT_DIR)): sha256_file(path)
            for path in output_paths
            if path.name != "asset_manifest.json"
        },
    }
    write_text_atomic(
        GENERATED_DIR / "asset_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def main() -> None:
    manifests = verify_inputs()
    results = build_results(manifests)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    configure_plotting()
    write_macros(results)
    write_generated_tables(manifests, results)
    figure_study_design(results)
    figure_development_models()
    figure_temporal_evaluation()
    write_asset_manifest(results, manifests)
    print(json.dumps({
        "selected_model": results["selected_model"],
        "development_rmse": results["development_rmse"],
        "temporal_rmse": results["temporal_rmse"],
        "temporal_r2": results["temporal_r2"],
        "assets": str(MANUSCRIPT_DIR),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
