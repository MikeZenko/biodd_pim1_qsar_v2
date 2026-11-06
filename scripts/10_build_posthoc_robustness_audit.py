#!/usr/bin/env python3
"""Build a hash-linked robustness audit from immutable temporal predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pim1_qsar.artifacts import (  # noqa: E402
    atomic_write_json,
    atomic_write_text,
    sha256_file,
    sha256_json,
)
from pim1_qsar.audits.temporal_robustness import (  # noqa: E402
    IDENTITY_COLUMN,
    build_audit_frame,
    group_centered_metrics,
    metric_uncertainty_by_cluster,
    point_metrics,
    spearman_cluster_bootstrap,
    temporal_degradation_bootstrap,
)


DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "audits" / "temporal_robustness_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def verify_frozen_inputs(paths: dict[str, Path]) -> dict[str, str]:
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    temporal_manifest = read_json(paths["temporal_manifest"])
    expected_predictions = temporal_manifest["output_hashes"]["temporal_predictions.csv"]
    if sha256_file(paths["predictions"]) != expected_predictions:
        raise RuntimeError("Temporal prediction checksum does not match its sealed manifest")
    split_manifest = read_json(paths["split_manifest"])
    expected_development = split_manifest["output_files_sha256"]["development.csv"]
    if sha256_file(paths["development"]) != expected_development:
        raise RuntimeError("Development checksum does not match its split-lock manifest")
    curation_manifest = read_json(paths["curation_manifest"])
    for key, manifest_key in (
        ("strict_events", "strict_measurement_events.csv"),
        ("broad_events", "broad_measurement_events.csv"),
    ):
        if sha256_file(paths[key]) != curation_manifest["output_files_sha256"][manifest_key]:
            raise RuntimeError(f"{key} checksum does not match its curation manifest")
    return {label: sha256_file(path) for label, path in sorted(paths.items())}


def completed_manifest(output_dir: Path) -> dict | None:
    manifest_path = output_dir / "audit_manifest.json"
    marker = output_dir / "RUN_COMPLETE"
    if not output_dir.exists():
        return None
    if not manifest_path.is_file() or not marker.is_file():
        raise RuntimeError("Robustness audit directory exists but is incomplete")
    manifest = read_json(manifest_path)
    for filename, expected in manifest.get("output_hashes", {}).items():
        path = output_dir / filename
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Robustness audit checksum mismatch: {filename}")
    return manifest


def main() -> None:
    args = parse_args()
    if args.bootstrap < 100:
        raise ValueError("--bootstrap must be at least 100")
    output_dir = args.output_dir.resolve()
    existing = completed_manifest(output_dir)
    if existing is not None and not args.force:
        print(json.dumps(existing, indent=2, sort_keys=True))
        print("Existing post hoc robustness audit verified.")
        return

    split_dir = (
        PROJECT_ROOT
        / "artifacts"
        / "split_locks"
        / "chembl37_strict_temporal2018_scaffold3x5_v1"
    )
    curation_dir = PROJECT_ROOT / "data" / "curated" / "chembl37_pim1_v2"
    temporal_dir = PROJECT_ROOT / "artifacts" / "runs" / "temporal_evaluation_v2"
    paths = {
        "predictions": temporal_dir / "temporal_predictions.csv",
        "temporal_manifest": temporal_dir / "temporal_evaluation_manifest.json",
        "development": split_dir / "development.csv",
        "split_manifest": split_dir / "split_lock_manifest.json",
        "strict_events": curation_dir / "strict_measurement_events.csv",
        "broad_events": curation_dir / "broad_measurement_events.csv",
        "record_qc": curation_dir / "record_qc.csv",
        "curation_manifest": curation_dir / "curation_manifest.json",
        "classical_environment": PROJECT_ROOT
        / "artifacts"
        / "environment"
        / "classical_curation.json",
        "audit_runner": Path(__file__).resolve(),
        "audit_module": PROJECT_ROOT
        / "src"
        / "pim1_qsar"
        / "audits"
        / "temporal_robustness.py",
    }
    input_hashes = verify_frozen_inputs(paths)
    predictions = pd.read_csv(paths["predictions"])
    development = pd.read_csv(paths["development"])
    strict_events = pd.read_csv(paths["strict_events"])
    broad_events = pd.read_csv(paths["broad_events"])
    audit_frame = build_audit_frame(
        predictions,
        development,
        strict_events,
        broad_events,
        cutoff_year=2018,
    )

    cluster_columns = ("scaffold_smiles", "document_cluster", "assay_cluster")
    primary_uncertainty = {
        column: metric_uncertainty_by_cluster(
            audit_frame,
            column,
            n_bootstrap=args.bootstrap,
            random_state=4101,
        )
        for column in cluster_columns
    }
    degradation = {
        column: temporal_degradation_bootstrap(
            audit_frame,
            column,
            first_year=2019,
            last_year=2024,
            n_bootstrap=args.bootstrap,
            random_state=4201,
        )
        for column in cluster_columns
    }

    early_broad = audit_frame.loc[audit_frame["broad_record_by_cutoff"]].copy()
    strict_chronology = audit_frame.loc[~audit_frame["broad_record_by_cutoff"]].copy()
    if early_broad.empty:
        raise RuntimeError("Broad-record chronology cross-check unexpectedly found no cases")
    record_qc = pd.read_csv(paths["record_qc"])
    earlier_reasons = (
        record_qc.loc[
            record_qc[IDENTITY_COLUMN].astype(str).isin(early_broad[IDENTITY_COLUMN].astype(str))
            & record_qc["broad_eligible"].astype(bool)
            & pd.to_numeric(record_qc["document_year"]).le(2018)
        ]
        .groupby(IDENTITY_COLUMN, sort=True)["strict_exclusion_reasons"]
        .agg(lambda values: "|".join(sorted(set(values.dropna().astype(str)))))
    )
    early_rows = early_broad[
        [IDENTITY_COLUMN, "first_broad_record_year", "first_record_year"]
    ].copy()
    early_rows["earlier_strict_exclusion_reasons"] = early_rows[IDENTITY_COLUMN].map(
        earlier_reasons
    )

    results = {
        "schema_version": "temporal_robustness_audit_v2",
        "status": "post_hoc_descriptive",
        "information_boundary": (
            "Consumes immutable selected-model predictions only; no fitting, tuning, "
            "candidate comparison, or prediction changes are permitted."
        ),
        "n_temporal_compounds": int(len(audit_frame)),
        "source_inventory": {
            "scaffold_clusters": int(audit_frame["scaffold_smiles"].nunique()),
            "document_source_sets": int(audit_frame["document_cluster"].nunique()),
            "assay_source_sets": int(audit_frame["assay_cluster"].nunique()),
            "multi_document_labels": int(audit_frame["document_cluster_count"].gt(1).sum()),
            "multi_assay_labels": int(audit_frame["assay_cluster_count"].gt(1).sum()),
        },
        "primary_uncertainty_by_cluster": primary_uncertainty,
        "temporal_degradation": degradation,
        "source_centered_ordering": {
            "document": group_centered_metrics(
                audit_frame, "document_cluster", minimum_group_size=3
            ),
            "assay": group_centered_metrics(
                audit_frame, "assay_cluster", minimum_group_size=3
            ),
        },
        "broad_record_chronology": {
            "definition": (
                "Temporal identities with any broad-cohort eligible record at or before "
                "the 2018 cutoff, despite first meeting the strict cohort after the cutoff."
            ),
            "count": int(len(early_broad)),
            "remaining_strict_chronology_count": int(len(strict_chronology)),
            "identity_rows": early_rows.to_dict("records"),
            "sensitivity_point_metrics": point_metrics(strict_chronology),
            "sensitivity_scaffold_bootstrap": metric_uncertainty_by_cluster(
                strict_chronology,
                "scaffold_smiles",
                n_bootstrap=args.bootstrap,
                random_state=4301,
            ),
        },
        "nearest_neighbor_potency_discordance": spearman_cluster_bootstrap(
            audit_frame,
            "nearest_neighbor_potency_gap",
            "absolute_error",
            "scaffold_smiles",
            n_bootstrap=args.bootstrap,
            random_state=4401,
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    assignments_name = "temporal_audit_assignments.csv"
    assignments_text = audit_frame.to_csv(index=False, lineterminator="\n", float_format="%.12g")
    atomic_write_text(output_dir / assignments_name, assignments_text)
    atomic_write_json(output_dir / "audit_metrics.json", results)
    manifest_core = {
        "schema_version": "temporal_robustness_audit_manifest_v2",
        "status": "complete_post_hoc_audit",
        "input_hashes": input_hashes,
        "n_bootstrap": args.bootstrap,
        "seeds": {
            "primary_uncertainty": 4101,
            "degradation": 4201,
            "strict_chronology_sensitivity": 4301,
            "nearest_neighbor_discordance": 4401,
        },
        "output_hashes": {
            assignments_name: sha256_file(output_dir / assignments_name),
            "audit_metrics.json": sha256_file(output_dir / "audit_metrics.json"),
        },
    }
    manifest = dict(manifest_core)
    manifest["audit_sha256"] = sha256_json(manifest_core)
    atomic_write_json(output_dir / "audit_manifest.json", manifest)
    atomic_write_text(output_dir / "RUN_COMPLETE", manifest["audit_sha256"] + "\n")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
