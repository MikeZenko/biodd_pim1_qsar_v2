from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pim1_qsar.artifacts import sha256_file, sha256_json
from pim1_qsar.validation.locking import (
    OUTPUT_FILENAMES,
    lock_splits,
    load_split_lock_settings,
    select_outcome_blind_cutoff,
)
from pim1_qsar.validation.splits import (
    validate_outer_assignment_frame,
    validate_temporal_partitions,
)


IDENTITY = "standardized_parent_inchi_key"
SCAFFOLD = "scaffold_smiles"


def _event_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(800):
        rows.append(
            {
                "event_id": f"dev-event-{index:04d}",
                IDENTITY: f"DEV-{index:04d}",
                "standardized_parent_smiles": f"C{'C' * (index % 4)}N",
                SCAFFOLD: f"dev-scaffold-{index // 5:03d}",
                "document_year": 2018,
                "pIC50": 5.0 + (index % 40) / 10.0,
            }
        )
    # Later measurements for known development compounds are label-drift
    # records, not additions to development labels.
    for index in range(10):
        rows.append(
            {
                "event_id": f"cross-event-{index:04d}",
                IDENTITY: f"DEV-{index:04d}",
                "standardized_parent_smiles": f"C{'C' * (index % 4)}N",
                SCAFFOLD: f"dev-scaffold-{index // 5:03d}",
                "document_year": 2019,
                "pIC50": 99.0,
            }
        )
    for index in range(260):
        rows.append(
            {
                "event_id": f"test-event-{index:04d}",
                IDENTITY: f"TEST-{index:04d}",
                "standardized_parent_smiles": f"N{'C' * (index % 4)}O",
                SCAFFOLD: f"test-scaffold-{index // 2:03d}",
                "document_year": 2019,
                "pIC50": 5.5 + (index % 35) / 10.0,
            }
        )
    return pd.DataFrame(rows)


def _feasibility() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "cutoff_year": 2017,
                "development_compounds": 700,
                "temporal_test_compounds": 360,
                "development_scaffolds": 140,
                "temporal_test_scaffolds": 180,
                "new_temporal_test_scaffolds": 180,
                "temporal_test_compounds_on_new_scaffolds": 360,
                "cross_cutoff_event_count": 10,
                "earliest_temporal_test_year": 2018,
                "latest_temporal_test_year": 2019,
                "selection_status": "unselected",
            },
            {
                "cutoff_year": 2018,
                "development_compounds": 800,
                "temporal_test_compounds": 260,
                "development_scaffolds": 160,
                "temporal_test_scaffolds": 130,
                "new_temporal_test_scaffolds": 130,
                "temporal_test_compounds_on_new_scaffolds": 260,
                "cross_cutoff_event_count": 10,
                "earliest_temporal_test_year": 2019,
                "latest_temporal_test_year": 2019,
                "selection_status": "unselected",
            },
            {
                "cutoff_year": 2019,
                "development_compounds": 900,
                "temporal_test_compounds": 160,
                "development_scaffolds": 180,
                "temporal_test_scaffolds": 80,
                "new_temporal_test_scaffolds": 80,
                "temporal_test_compounds_on_new_scaffolds": 160,
                "cross_cutoff_event_count": 0,
                "earliest_temporal_test_year": 2020,
                "latest_temporal_test_year": 2020,
                "selection_status": "unselected",
            },
        ]
    )


def _write_curated_dir(path: Path) -> Path:
    path.mkdir(parents=True)
    events_path = path / "strict_measurement_events.csv"
    feasibility_path = path / "temporal_feasibility.csv"
    _event_rows().to_csv(events_path, index=False, lineterminator="\n")
    _feasibility().to_csv(feasibility_path, index=False, lineterminator="\n")
    manifest = {
        "schema_version": "2.0",
        "curated_dataset_sha256": "curated-test-fixture",
        "temporal_feasibility": {"selected_cutoff": None},
        "output_files_sha256": {
            events_path.name: sha256_file(events_path),
            feasibility_path.name: sha256_file(feasibility_path),
        },
    }
    (path / "curation_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _refresh_feasibility_hash(curated_dir: Path) -> None:
    manifest_path = curated_dir / "curation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output_files_sha256"]["temporal_feasibility.csv"] = sha256_file(
        curated_dir / "temporal_feasibility.csv"
    )
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def test_protocol_locks_requested_cutoff_splits_and_graph_families():
    settings = load_split_lock_settings(PROJECT_ROOT / "configs" / "protocol_v2.yaml")
    protocol = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "protocol_v2.yaml").read_text(encoding="utf-8")
    )

    assert settings.expected_cutoff == 2018
    assert settings.n_repeats == 3
    assert settings.n_splits == 5
    assert settings.seeds == (1101, 1102, 1103)
    assert settings.inner_folds == 3
    assert protocol["candidate_families"]["graph"] == [
        "chemprop_dmpnn",
        "edge_aware_gine",
    ]


def test_outcome_blind_rule_selects_the_latest_eligible_cutoff():
    selected = select_outcome_blind_cutoff(
        _feasibility(),
        minimum_development_compounds=750,
        minimum_temporal_test_compounds=250,
        minimum_new_temporal_test_scaffolds=100,
        expected_cutoff=2018,
    )
    assert selected["cutoff_year"] == 2018


def test_outcome_blind_rule_fails_on_no_duplicate_or_unexpected_match():
    no_match = _feasibility().assign(temporal_test_compounds=1)
    with pytest.raises(ValueError, match="No temporal cutoff"):
        select_outcome_blind_cutoff(
            no_match,
            minimum_development_compounds=750,
            minimum_temporal_test_compounds=250,
            minimum_new_temporal_test_scaffolds=100,
            expected_cutoff=2018,
        )

    duplicate = pd.concat([_feasibility(), _feasibility().iloc[[1]]], ignore_index=True)
    with pytest.raises(ValueError, match="multiple rows"):
        select_outcome_blind_cutoff(
            duplicate,
            minimum_development_compounds=750,
            minimum_temporal_test_compounds=250,
            minimum_new_temporal_test_scaffolds=100,
            expected_cutoff=2018,
        )

    with pytest.raises(ValueError, match="not locked cutoff"):
        select_outcome_blind_cutoff(
            _feasibility(),
            minimum_development_compounds=750,
            minimum_temporal_test_compounds=250,
            minimum_new_temporal_test_scaffolds=100,
            expected_cutoff=2017,
        )


def test_lock_splits_persists_sealed_labels_shared_folds_and_lineage(tmp_path):
    curated_dir = _write_curated_dir(tmp_path / "curated")
    output_root = tmp_path / "locks"
    lock_dir, manifest = lock_splits(
        protocol_path=PROJECT_ROOT / "configs" / "protocol_v2.yaml",
        curated_dir=curated_dir,
        output_root=output_root,
    )

    assert set(path.name for path in lock_dir.iterdir()) == {
        *OUTPUT_FILENAMES,
        "split_lock_manifest.json",
    }
    assert manifest["status"] == "locked"
    assert manifest["immutable"] is True
    assert manifest["counts"]["cutoff_year"] == 2018
    assert manifest["counts"]["development_compounds"] == 800
    assert manifest["counts"]["temporal_test_compounds"] == 260
    assert manifest["counts"]["new_temporal_test_scaffolds"] == 130
    assert manifest["counts"]["outer_assignment_rows"] == 800 * 3 * 5
    assert manifest["split_definitions"]["inner_tuning"]["precomputed_at_split_lock"] is False
    assert manifest["split_definitions"]["outer_validation"][
        "shared_across_all_model_families"
    ] is True

    manifest_core = dict(manifest)
    manifest_core.pop("split_lock_sha256")
    assert manifest["split_lock_sha256"] == sha256_json(manifest_core)
    for filename, expected in manifest["output_files_sha256"].items():
        assert sha256_file(lock_dir / filename) == expected

    development = pd.read_csv(lock_dir / "development.csv")
    temporal_test = pd.read_csv(lock_dir / "temporal_test_sealed.csv")
    cross_cutoff = pd.read_csv(lock_dir / "cross_cutoff_records.csv")
    assignments = pd.read_csv(lock_dir / "outer_fold_assignments.csv")
    validate_temporal_partitions(
        development,
        temporal_test,
        cross_cutoff,
        cutoff_year=2018,
        identity_col=IDENTITY,
        scaffold_col=SCAFFOLD,
        year_col="document_year",
    )
    validate_outer_assignment_frame(
        assignments,
        development,
        identity_col=IDENTITY,
        scaffold_col=SCAFFOLD,
        n_repeats=3,
        n_splits=5,
        seeds=(1101, 1102, 1103),
    )
    assert set(development[IDENTITY]).isdisjoint(temporal_test[IDENTITY])
    assert development["last_record_year"].max() == 2018
    assert temporal_test["first_record_year"].min() == 2019
    assert set(cross_cutoff["event_id"]) == {
        f"cross-event-{index:04d}" for index in range(10)
    }
    # The future value 99.0 is excluded from the locked development median.
    observed = development.set_index(IDENTITY).loc["DEV-0000", "pIC50"]
    assert observed == pytest.approx(5.0)
    assert "model_family" not in assignments.columns


def test_existing_lock_is_verified_not_overwritten_and_tampering_fails(tmp_path):
    curated_dir = _write_curated_dir(tmp_path / "curated")
    output_root = tmp_path / "locks"
    lock_dir, first = lock_splits(
        protocol_path=PROJECT_ROOT / "configs" / "protocol_v2.yaml",
        curated_dir=curated_dir,
        output_root=output_root,
    )
    manifest_bytes = (lock_dir / "split_lock_manifest.json").read_bytes()
    replay_dir, replay = lock_splits(
        protocol_path=PROJECT_ROOT / "configs" / "protocol_v2.yaml",
        curated_dir=curated_dir,
        output_root=output_root,
    )
    assert replay_dir == lock_dir
    assert replay == first
    assert (lock_dir / "split_lock_manifest.json").read_bytes() == manifest_bytes

    with (lock_dir / "development.csv").open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")
    with pytest.raises(RuntimeError, match="output mismatch"):
        lock_splits(
            protocol_path=PROJECT_ROOT / "configs" / "protocol_v2.yaml",
            curated_dir=curated_dir,
            output_root=output_root,
        )


def test_feasibility_row_must_exactly_match_event_derived_split(tmp_path):
    curated_dir = _write_curated_dir(tmp_path / "curated")
    feasibility = pd.read_csv(curated_dir / "temporal_feasibility.csv")
    feasibility.loc[feasibility["cutoff_year"].eq(2018), "development_scaffolds"] = 159
    feasibility.to_csv(curated_dir / "temporal_feasibility.csv", index=False)
    _refresh_feasibility_hash(curated_dir)

    with pytest.raises(ValueError, match="does not match strict measurement events"):
        lock_splits(
            protocol_path=PROJECT_ROOT / "configs" / "protocol_v2.yaml",
            curated_dir=curated_dir,
            output_root=tmp_path / "locks",
        )


def test_cli_never_falls_back_to_compound_strict(tmp_path):
    curated_dir = tmp_path / "curated"
    curated_dir.mkdir()
    _event_rows().drop_duplicates(IDENTITY).to_csv(
        curated_dir / "compound_strict.csv", index=False
    )
    _feasibility().to_csv(curated_dir / "temporal_feasibility.csv", index=False)
    (curated_dir / "curation_manifest.json").write_text(
        json.dumps(
            {
                "temporal_feasibility": {"selected_cutoff": None},
                "output_files_sha256": {},
            }
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "03_lock_splits.py"),
            "--curated-dir",
            str(curated_dir),
            "--output-root",
            str(tmp_path / "locks"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "strict_measurement_events.csv" in completed.stderr
    assert not (tmp_path / "locks").exists()
