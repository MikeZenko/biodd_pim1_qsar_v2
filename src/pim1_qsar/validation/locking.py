"""Immutable, outcome-blind temporal and outer-fold split locking."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from pim1_qsar.artifacts import (
    atomic_write_json,
    atomic_write_text,
    sha256_file,
    sha256_json,
)
from pim1_qsar.validation.splits import (
    fold_assignments_to_frame,
    make_repeated_scaffold_folds,
    make_temporal_split,
    validate_outer_assignment_frame,
    validate_temporal_partitions,
)


LOCK_SCHEMA_VERSION = "2.0"
STRICT_EVENTS_FILENAME = "strict_measurement_events.csv"
FEASIBILITY_FILENAME = "temporal_feasibility.csv"
CURATION_MANIFEST_FILENAME = "curation_manifest.json"
OUTPUT_FILENAMES = (
    "development.csv",
    "temporal_test_sealed.csv",
    "cross_cutoff_records.csv",
    "outer_fold_assignments.csv",
)


@dataclass(frozen=True)
class SplitLockSettings:
    lock_id: str
    expected_cutoff: int
    minimum_development_compounds: int
    minimum_temporal_test_compounds: int
    minimum_new_temporal_test_scaffolds: int
    identity_col: str
    scaffold_col: str
    structure_col: str
    year_col: str
    value_col: str
    event_id_col: str
    n_splits: int
    n_repeats: int
    seeds: tuple[int, ...]
    n_bins: int
    inner_folds: int


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Protocol field {label} must be a mapping")
    return value


def load_split_lock_settings(protocol_path: Path) -> SplitLockSettings:
    """Read and strictly validate the split-defining protocol fields."""

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - exercised in environment setup
        raise RuntimeError("Split locking requires PyYAML to read protocol_v2.yaml") from exc

    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    root = _require_mapping(protocol, "root")
    dataset = _require_mapping(root.get("dataset"), "dataset")
    temporal = _require_mapping(root.get("temporal_test"), "temporal_test")
    selection = _require_mapping(temporal.get("cutoff_selection"), "temporal_test.cutoff_selection")
    outer = _require_mapping(root.get("outer_validation"), "outer_validation")
    inner = _require_mapping(root.get("inner_tuning"), "inner_tuning")
    artifacts = _require_mapping(root.get("artifact_policy"), "artifact_policy")
    lock_sources = _require_mapping(
        artifacts.get("split_lock_sources"), "artifact_policy.split_lock_sources"
    )

    if selection.get("rule") != "latest_cutoff_meeting_all_minimum_counts":
        raise ValueError("Unsupported or outcome-dependent temporal cutoff rule")
    if selection.get("outcome_columns_permitted") is not False:
        raise ValueError("Temporal cutoff selection must forbid outcome columns")
    if temporal.get("method") != "first_eligible_document_year":
        raise ValueError("Split lock requires first_eligible_document_year")
    if temporal.get("development_labels_use_post_cutoff_records") is not False:
        raise ValueError("Protocol would allow post-cutoff development labels")
    if outer.get("splitter") != "stratified_group_kfold":
        raise ValueError("Outer splitter must be stratified_group_kfold")
    if inner.get("precomputed_at_split_lock") is not False:
        raise ValueError("Inner folds must not be precomputed at split lock")
    if inner.get("creation_scope") != "within_each_outer_training_partition":
        raise ValueError("Inner folds must be created inside each outer-training partition")
    required_source_names = {
        "measurement_events": STRICT_EVENTS_FILENAME,
        "temporal_feasibility": FEASIBILITY_FILENAME,
        "curation_manifest": CURATION_MANIFEST_FILENAME,
    }
    for key, expected in required_source_names.items():
        if lock_sources.get(key) != expected:
            raise ValueError(f"Split-lock source {key} must be {expected}")

    expected_cutoff = int(selection["expected_selected_cutoff"])
    if int(temporal["development_through_year"]) != expected_cutoff:
        raise ValueError("Protocol cutoff fields disagree")
    if int(temporal["test_from_year"]) != expected_cutoff + 1:
        raise ValueError("Temporal test must begin in the year after the cutoff")
    n_repeats = int(outer["repeats"])
    seeds = tuple(int(seed) for seed in outer["seeds"])
    if len(seeds) != n_repeats or len(set(seeds)) != len(seeds):
        raise ValueError("Outer seeds must be unique and match the repeat count")

    return SplitLockSettings(
        lock_id=str(artifacts["split_lock_id"]),
        expected_cutoff=expected_cutoff,
        minimum_development_compounds=int(selection["minimum_development_compounds"]),
        minimum_temporal_test_compounds=int(selection["minimum_temporal_test_compounds"]),
        minimum_new_temporal_test_scaffolds=int(
            selection["minimum_new_temporal_test_scaffolds"]
        ),
        identity_col=str(dataset["identity_column"]),
        scaffold_col=str(dataset["scaffold_column"]),
        structure_col=str(dataset["structure_column"]),
        year_col=str(dataset["document_year_column"]),
        value_col=str(dataset["endpoint"]),
        event_id_col=str(dataset["measurement_event_id_column"]),
        n_splits=int(outer["folds"]),
        n_repeats=n_repeats,
        seeds=seeds,
        n_bins=int(outer["regression_activity_bins"]),
        inner_folds=int(inner["folds"]),
    )


def select_outcome_blind_cutoff(
    feasibility: pd.DataFrame,
    *,
    minimum_development_compounds: int,
    minimum_temporal_test_compounds: int,
    minimum_new_temporal_test_scaffolds: int,
    expected_cutoff: int,
) -> pd.Series:
    """Select the latest cutoff satisfying prespecified count-only criteria."""

    required = [
        "cutoff_year",
        "development_compounds",
        "temporal_test_compounds",
        "new_temporal_test_scaffolds",
    ]
    missing = sorted(set(required) - set(feasibility.columns))
    if missing:
        raise ValueError(f"Temporal feasibility table is missing: {', '.join(missing)}")
    if feasibility.empty:
        raise ValueError("Temporal feasibility table is empty")
    numeric = feasibility.copy()
    for column in required:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
        if numeric[column].isna().any() or not np.isfinite(numeric[column]).all():
            raise ValueError(f"Temporal feasibility column {column} must be finite numeric")
        if not np.equal(numeric[column], np.floor(numeric[column])).all():
            raise ValueError(f"Temporal feasibility column {column} must contain integers")
        numeric[column] = numeric[column].astype(int)
    duplicated_cutoffs = numeric.loc[numeric["cutoff_year"].duplicated(keep=False), "cutoff_year"]
    if not duplicated_cutoffs.empty:
        raise ValueError(
            "Temporal feasibility contains multiple rows for cutoff(s): "
            + ", ".join(str(value) for value in sorted(set(duplicated_cutoffs)))
        )
    eligible = numeric.loc[
        numeric["development_compounds"].ge(minimum_development_compounds)
        & numeric["temporal_test_compounds"].ge(minimum_temporal_test_compounds)
        & numeric["new_temporal_test_scaffolds"].ge(
            minimum_new_temporal_test_scaffolds
        )
    ]
    if eligible.empty:
        raise ValueError("No temporal cutoff satisfies the outcome-blind minimum-count rule")
    selected_cutoff = int(eligible["cutoff_year"].max())
    selected = numeric.loc[numeric["cutoff_year"].eq(selected_cutoff)]
    if len(selected) != 1:
        raise ValueError("Outcome-blind cutoff rule did not resolve to exactly one row")
    if selected_cutoff != expected_cutoff:
        raise ValueError(
            f"Outcome-blind rule selected {selected_cutoff}, not locked cutoff {expected_cutoff}"
        )
    # Return the original row so descriptive feasibility fields are preserved.
    original = feasibility.loc[
        pd.to_numeric(feasibility["cutoff_year"], errors="coerce").eq(selected_cutoff)
    ]
    if len(original) != 1:
        raise ValueError("Expected exactly one original feasibility row for the locked cutoff")
    return original.iloc[0].copy()


def temporal_split_statistics(
    development: pd.DataFrame,
    temporal_test: pd.DataFrame,
    cross_cutoff_records: pd.DataFrame,
    *,
    cutoff_year: int,
    scaffold_col: str,
) -> dict[str, int]:
    """Recompute every count used to verify the selected feasibility row."""

    development_scaffolds = set(development[scaffold_col].astype(str))
    temporal_scaffolds = set(temporal_test[scaffold_col].astype(str))
    new_scaffolds = temporal_scaffolds - development_scaffolds
    temporal_new_count = int(temporal_test[scaffold_col].astype(str).isin(new_scaffolds).sum())
    return {
        "cutoff_year": int(cutoff_year),
        "development_compounds": int(len(development)),
        "temporal_test_compounds": int(len(temporal_test)),
        "development_scaffolds": int(len(development_scaffolds)),
        "temporal_test_scaffolds": int(len(temporal_scaffolds)),
        "new_temporal_test_scaffolds": int(len(new_scaffolds)),
        "temporal_test_compounds_on_new_scaffolds": temporal_new_count,
        "cross_cutoff_event_count": int(len(cross_cutoff_records)),
        "earliest_temporal_test_year": int(temporal_test["first_record_year"].min()),
        "latest_temporal_test_year": int(temporal_test["last_record_year"].max()),
    }


def verify_selected_feasibility_row(
    selected_row: pd.Series,
    recomputed: Mapping[str, int],
) -> None:
    """Require one exact feasibility row for the event-derived split."""

    mismatches: list[str] = []
    for column, expected in recomputed.items():
        if column not in selected_row.index:
            mismatches.append(f"{column}=missing")
            continue
        observed = pd.to_numeric(pd.Series([selected_row[column]]), errors="coerce").iloc[0]
        if pd.isna(observed) or int(observed) != int(expected):
            mismatches.append(f"{column}: observed={selected_row[column]!r}, expected={expected}")
    if mismatches:
        raise ValueError(
            "Locked temporal feasibility row does not match strict measurement events: "
            + "; ".join(mismatches)
        )


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if pd.isna(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _csv_text(frame: pd.DataFrame) -> str:
    return frame.to_csv(index=False, lineterminator="\n", float_format="%.12g")


def _verify_curated_sources(
    curated_dir: Path,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    events_path = curated_dir / STRICT_EVENTS_FILENAME
    feasibility_path = curated_dir / FEASIBILITY_FILENAME
    curation_manifest_path = curated_dir / CURATION_MANIFEST_FILENAME
    if events_path.name != STRICT_EVENTS_FILENAME:
        raise RuntimeError("Split locking only accepts strict_measurement_events.csv")
    for path in (events_path, feasibility_path, curation_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required curated artifact: {path}")
    manifest = json.loads(curation_manifest_path.read_text(encoding="utf-8"))
    expected_hashes = manifest.get("output_files_sha256")
    if not isinstance(expected_hashes, Mapping):
        raise ValueError("Curation manifest omits output_files_sha256")
    for path in (events_path, feasibility_path):
        expected = expected_hashes.get(path.name)
        if not expected:
            raise ValueError(f"Curation manifest does not register {path.name}")
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(f"Curated-source checksum mismatch for {path.name}")
    if manifest.get("temporal_feasibility", {}).get("selected_cutoff") is not None:
        raise ValueError("Curation stage must not select the temporal cutoff")
    return events_path, feasibility_path, curation_manifest_path, manifest


def _verify_existing_lock(
    lock_dir: Path,
    *,
    protocol_path: Path,
    events_path: Path,
    feasibility_path: Path,
    curation_manifest_path: Path,
    settings: SplitLockSettings,
) -> dict[str, Any]:
    manifest_path = lock_dir / "split_lock_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Existing split lock has no manifest: {lock_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("lock_id") != settings.lock_id or manifest.get("status") != "locked":
        raise RuntimeError("Existing split lock identity/status mismatch")
    observed_lock_hash = manifest.get("split_lock_sha256")
    manifest_core = dict(manifest)
    manifest_core.pop("split_lock_sha256", None)
    if observed_lock_hash != sha256_json(manifest_core):
        raise RuntimeError("Existing split-lock manifest hash mismatch")
    expected_sources = {
        "protocol_sha256": sha256_file(protocol_path),
        "strict_measurement_events_sha256": sha256_file(events_path),
        "temporal_feasibility_sha256": sha256_file(feasibility_path),
        "curation_manifest_sha256": sha256_file(curation_manifest_path),
    }
    for key, expected in expected_sources.items():
        if manifest.get("source_hashes", {}).get(key) != expected:
            raise RuntimeError(f"Existing immutable lock was built from another {key}")
    output_hashes = manifest.get("output_files_sha256", {})
    if set(output_hashes) != set(OUTPUT_FILENAMES):
        raise RuntimeError("Existing split lock output inventory mismatch")
    for filename, expected in output_hashes.items():
        path = lock_dir / filename
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Existing split lock output mismatch: {filename}")

    development = pd.read_csv(lock_dir / "development.csv")
    temporal_test = pd.read_csv(lock_dir / "temporal_test_sealed.csv")
    cross_cutoff = pd.read_csv(lock_dir / "cross_cutoff_records.csv")
    assignments = pd.read_csv(lock_dir / "outer_fold_assignments.csv")
    validate_temporal_partitions(
        development,
        temporal_test,
        cross_cutoff,
        cutoff_year=settings.expected_cutoff,
        identity_col=settings.identity_col,
        scaffold_col=settings.scaffold_col,
        year_col=settings.year_col,
    )
    validate_outer_assignment_frame(
        assignments,
        development,
        identity_col=settings.identity_col,
        scaffold_col=settings.scaffold_col,
        n_repeats=settings.n_repeats,
        n_splits=settings.n_splits,
        seeds=settings.seeds,
    )
    return manifest


def lock_splits(
    *,
    protocol_path: Path,
    curated_dir: Path,
    output_root: Path,
) -> tuple[Path, dict[str, Any]]:
    """Create or verify an immutable split lock; never overwrite one."""

    protocol_path = protocol_path.resolve()
    curated_dir = curated_dir.resolve()
    output_root = output_root.resolve()
    settings = load_split_lock_settings(protocol_path)
    events_path, feasibility_path, curation_manifest_path, curation_manifest = (
        _verify_curated_sources(curated_dir)
    )
    lock_dir = output_root / settings.lock_id
    if lock_dir.exists():
        return lock_dir, _verify_existing_lock(
            lock_dir,
            protocol_path=protocol_path,
            events_path=events_path,
            feasibility_path=feasibility_path,
            curation_manifest_path=curation_manifest_path,
            settings=settings,
        )

    events = pd.read_csv(events_path, low_memory=False)
    feasibility = pd.read_csv(feasibility_path)
    selected_row = select_outcome_blind_cutoff(
        feasibility,
        minimum_development_compounds=settings.minimum_development_compounds,
        minimum_temporal_test_compounds=settings.minimum_temporal_test_compounds,
        minimum_new_temporal_test_scaffolds=settings.minimum_new_temporal_test_scaffolds,
        expected_cutoff=settings.expected_cutoff,
    )
    temporal = make_temporal_split(
        events,
        cutoff_year=settings.expected_cutoff,
        identity_col=settings.identity_col,
        year_col=settings.year_col,
        value_col=settings.value_col,
        scaffold_col=settings.scaffold_col,
        stable_cols=(settings.structure_col,),
        event_id_col=settings.event_id_col,
    )
    validate_temporal_partitions(
        temporal.development,
        temporal.temporal_test,
        temporal.cross_cutoff_records,
        cutoff_year=settings.expected_cutoff,
        identity_col=settings.identity_col,
        scaffold_col=settings.scaffold_col,
        year_col=settings.year_col,
    )
    recomputed = temporal_split_statistics(
        temporal.development,
        temporal.temporal_test,
        temporal.cross_cutoff_records,
        cutoff_year=settings.expected_cutoff,
        scaffold_col=settings.scaffold_col,
    )
    verify_selected_feasibility_row(selected_row, recomputed)

    folds = make_repeated_scaffold_folds(
        temporal.development[settings.identity_col],
        temporal.development[settings.scaffold_col],
        temporal.development[settings.value_col],
        n_splits=settings.n_splits,
        n_repeats=settings.n_repeats,
        seeds=settings.seeds,
        n_bins=settings.n_bins,
    )
    assignments = fold_assignments_to_frame(
        folds,
        temporal.development,
        identity_col=settings.identity_col,
        scaffold_col=settings.scaffold_col,
    )
    validate_outer_assignment_frame(
        assignments,
        temporal.development,
        identity_col=settings.identity_col,
        scaffold_col=settings.scaffold_col,
        n_repeats=settings.n_repeats,
        n_splits=settings.n_splits,
        seeds=settings.seeds,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{settings.lock_id}.", suffix=".tmp", dir=output_root)
    )
    try:
        output_frames = {
            "development.csv": temporal.development,
            "temporal_test_sealed.csv": temporal.temporal_test,
            "cross_cutoff_records.csv": temporal.cross_cutoff_records,
            "outer_fold_assignments.csv": assignments,
        }
        for filename, frame in output_frames.items():
            atomic_write_text(temporary_dir / filename, _csv_text(frame))
        output_hashes = {
            filename: sha256_file(temporary_dir / filename) for filename in OUTPUT_FILENAMES
        }

        development_scaffolds = set(temporal.development[settings.scaffold_col].astype(str))
        temporal_scaffolds = set(temporal.temporal_test[settings.scaffold_col].astype(str))
        source_hashes = {
            "protocol_sha256": sha256_file(protocol_path),
            "strict_measurement_events_sha256": sha256_file(events_path),
            "temporal_feasibility_sha256": sha256_file(feasibility_path),
            "curation_manifest_sha256": sha256_file(curation_manifest_path),
        }
        split_definitions = {
            "temporal": {
                "method": "first_eligible_document_year",
                "cutoff_year": settings.expected_cutoff,
                "test_from_year": settings.expected_cutoff + 1,
                "identity_column": settings.identity_col,
                "scaffold_column": settings.scaffold_col,
                "source_table": STRICT_EVENTS_FILENAME,
                "development_labels_use_post_cutoff_events": False,
                "selection_rule": "latest cutoff satisfying all three minimum counts",
                "minimum_development_compounds": settings.minimum_development_compounds,
                "minimum_temporal_test_compounds": settings.minimum_temporal_test_compounds,
                "minimum_new_temporal_test_scaffolds": (
                    settings.minimum_new_temporal_test_scaffolds
                ),
            },
            "outer_validation": {
                "splitter": "StratifiedGroupKFold",
                "folds": settings.n_splits,
                "repeats": settings.n_repeats,
                "seeds": list(settings.seeds),
                "activity_bins": settings.n_bins,
                "shared_across_all_model_families": True,
            },
            "inner_tuning": {
                "splitter": "StratifiedGroupKFold",
                "folds": settings.inner_folds,
                "group": settings.scaffold_col,
                "creation_scope": "within_each_outer_training_partition",
                "precomputed_at_split_lock": False,
                "outer_validation_outcomes_permitted": False,
            },
        }
        manifest_core: dict[str, Any] = {
            "schema_version": LOCK_SCHEMA_VERSION,
            "lock_id": settings.lock_id,
            "status": "locked",
            "immutable": True,
            "source_files": {
                "curation_manifest": CURATION_MANIFEST_FILENAME,
                "measurement_events": STRICT_EVENTS_FILENAME,
                "temporal_feasibility": FEASIBILITY_FILENAME,
                "protocol": protocol_path.name,
            },
            "source_hashes": source_hashes,
            "source_curation_dataset_sha256": curation_manifest.get(
                "curated_dataset_sha256"
            ),
            "selected_feasibility_row": {
                str(column): _json_value(value) for column, value in selected_row.items()
            },
            "split_definitions": split_definitions,
            "split_definitions_sha256": sha256_json(split_definitions),
            "counts": {
                **recomputed,
                "development_temporal_scaffold_overlap": len(
                    development_scaffolds & temporal_scaffolds
                ),
                "outer_assignment_rows": len(assignments),
            },
            "identity_hashes": {
                "development_ids_sha256": sha256_json(
                    temporal.development[settings.identity_col].astype(str).tolist()
                ),
                "temporal_test_ids_sha256": sha256_json(
                    temporal.temporal_test[settings.identity_col].astype(str).tolist()
                ),
                "cross_cutoff_event_ids_sha256": sha256_json(
                    temporal.cross_cutoff_records[settings.event_id_col].astype(str).tolist()
                ),
            },
            "output_files_sha256": output_hashes,
        }
        manifest = dict(manifest_core)
        manifest["split_lock_sha256"] = sha256_json(manifest_core)
        atomic_write_json(temporary_dir / "split_lock_manifest.json", manifest)
        try:
            os.replace(temporary_dir, lock_dir)
        except OSError:
            if not lock_dir.exists():
                raise
            shutil.rmtree(temporary_dir)
            return lock_dir, _verify_existing_lock(
                lock_dir,
                protocol_path=protocol_path,
                events_path=events_path,
                feasibility_path=feasibility_path,
                curation_manifest_path=curation_manifest_path,
                settings=settings,
            )
    except BaseException:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        raise
    return lock_dir, manifest
