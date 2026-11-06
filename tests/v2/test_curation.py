from __future__ import annotations

import gzip
import importlib.metadata
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pim1_qsar.data.chembl_snapshot import sha256_file, write_json, write_jsonl_gz  # noqa: E402
from pim1_qsar.data.curation import (  # noqa: E402
    CurationConfig,
    StandardizedStructure,
    curate_records,
    curate_snapshot,
    load_chembl_structure_standardizer,
)


def fake_standardizer(smiles: str) -> StandardizedStructure:
    token = smiles.replace("[", "").replace("]", "")
    return StandardizedStructure(smiles, f"KEY-{token}", f"SCAF-{token[0]}")


def activity(
    activity_id: int,
    molecule: str,
    parent: str,
    assay: str,
    record_id: str,
    value: float,
    year: int,
    *,
    duplicate: int = 0,
    validity: str = "",
) -> dict[str, object]:
    return {
        "activity_id": activity_id,
        "record_id": record_id,
        "molecule_chembl_id": molecule,
        "parent_molecule_chembl_id": parent,
        "assay_chembl_id": assay,
        "document_chembl_id": f"DOC-{year}",
        "document_year": year,
        "target_chembl_id": "CHEMBL2147",
        "target_organism": "Homo sapiens",
        "target_tax_id": 9606,
        "standard_type": "IC50",
        "standard_relation": "=",
        "relation": "=",
        "standard_units": "nM",
        "standard_flag": 1,
        "standard_value": value,
        "potential_duplicate": duplicate,
        "data_validity_comment": validity,
        "data_validity_description": "",
        "bao_endpoint": "BAO_END",
        "assay_variant_accession": "",
        "assay_variant_mutation": "",
    }


def assay(
    assay_id: str,
    description: str,
    *,
    assay_type: str = "B",
    bao_format: str = "BAO_0000357",
    confidence: int = 9,
    relationship: str = "D",
) -> dict[str, object]:
    return {
        "assay_chembl_id": assay_id,
        "target_chembl_id": "CHEMBL2147",
        "assay_type": assay_type,
        "bao_format": bao_format,
        "description": description,
        "confidence_score": confidence,
        "relationship_type": relationship,
    }


@pytest.fixture()
def snapshot_tables() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    activities = [
        activity(1, "M1", "P1", "A1", "R1", 10.0, 2018),
        # Same source measurement reissued under an alias assay ID.
        activity(2, "M1", "P1", "A2", "R1", 10.0, 2018),
        activity(3, "M1", "P1", "A3", "R2", 1000.0, 2019),
        activity(4, "M2", "P2", "CHEMBL3705754", "R3", 0.05, 2021),
        activity(5, "M3", "P3", "CELL", "R4", 100.0, 2022),
        activity(6, "M4", "P4", "A1", "R5", 50.0, 2020, duplicate=1),
        activity(7, "M5", "P5", "LOW", "R6", 500.0, 2020),
        activity(8, "M6", "P6", "A1", "R7", 100.0, 2020, validity="warning"),
    ]
    assays = [
        assay("A1", "same assay"),
        assay("A2", "same assay"),
        assay("A3", "independent assay"),
        assay("CHEMBL3705754", "tail assay"),
        assay("CELL", "cell assay", assay_type="F", bao_format="BAO_0000219"),
        assay("LOW", "low confidence", confidence=7),
    ]
    parents = [
        {
            "molecule_chembl_id": f"P{i}",
            "parent_chembl_id": f"P{i}",
            "canonical_smiles": "C" * i,
        }
        for i in range(1, 7)
    ]
    return activities, assays, parents


def test_cohorts_event_dedup_and_sensitivities(snapshot_tables) -> None:
    activities, assays, parents = snapshot_tables
    result = curate_records(
        activities,
        assays,
        parents,
        standardizer=fake_standardizer,
        structure_pipeline_version="test-1",
    )
    counts = result.manifest["cohort_counts"]
    assert counts["broad"] == {
        "eligible_records": 6,
        "measurement_events": 5,
        "duplicate_records_collapsed": 1,
        "compounds": 4,
    }
    assert counts["strict"] == {
        "eligible_records": 4,
        "measurement_events": 3,
        "duplicate_records_collapsed": 1,
        "compounds": 2,
    }
    assert counts["strict_no_high_tail"]["events_removed"] == 1
    assert counts["strict_no_high_tail"]["compounds"] == 1
    assert counts["strict_no_CHEMBL3705754"]["events_removed"] == 1
    assert counts["strict_no_CHEMBL3705754"]["compounds"] == 1
    assert not result.strict_events["event_signature_sha256"].duplicated().any()


def test_lineage_preserves_every_collapsed_activity(snapshot_tables) -> None:
    activities, assays, parents = snapshot_tables
    result = curate_records(
        activities, assays, parents, standardizer=fake_standardizer
    )
    strict_duplicate = result.strict_events.loc[
        result.strict_events["lineage_record_count"].eq(2)
    ].iloc[0]
    assert strict_duplicate["activity_ids"] == "1|2"
    lineage = result.lineage.loc[
        (result.lineage["cohort"] == "strict")
        & (result.lineage["event_id"] == strict_duplicate["event_id"])
    ]
    assert set(lineage["activity_id"]) == {"1", "2"}
    assert lineage["is_event_representative"].sum() == 1
    reasons = result.exclusions.loc[
        result.exclusions["cohort"].eq("strict"), "reason"
    ].tolist()
    assert "duplicate_source_event_collapsed" in reasons


def test_measurement_dispersion_uses_unique_source_events(snapshot_tables) -> None:
    activities, assays, parents = snapshot_tables
    result = curate_records(
        activities, assays, parents, standardizer=fake_standardizer
    )
    m1 = result.strict_compounds.loc[
        result.strict_compounds["standardized_parent_inchi_key"] == "KEY-C"
    ].iloc[0]
    assert m1["measurement_event_count"] == 2
    assert m1["source_record_count"] == 3
    assert m1["pIC50"] == pytest.approx(7.0)
    assert m1["pIC50_range"] == pytest.approx(2.0)
    assert result.manifest["measurement_dispersion"]["strict_compounds_range_gt_1"] == 1


def test_strict_uses_joined_assay_confidence_and_format(snapshot_tables) -> None:
    activities, assays, parents = snapshot_tables
    result = curate_records(
        activities, assays, parents, standardizer=fake_standardizer
    )
    qc = result.record_qc.set_index("activity_id")
    assert bool(qc.loc["5", "broad_eligible"])
    assert not bool(qc.loc["5", "strict_eligible"])
    assert "assay_type_not_biochemical" in qc.loc["5", "strict_exclusion_reasons"]
    assert bool(qc.loc["7", "broad_eligible"])
    assert not bool(qc.loc["7", "strict_eligible"])
    assert "assay_confidence_not_strict" in qc.loc["7", "strict_exclusion_reasons"]
    assert not bool(qc.loc["6", "broad_eligible"])
    assert not bool(qc.loc["8", "broad_eligible"])


def test_temporal_table_reports_candidates_without_selecting(snapshot_tables) -> None:
    activities, assays, parents = snapshot_tables
    result = curate_records(
        activities, assays, parents, standardizer=fake_standardizer
    )
    table = result.temporal_feasibility.set_index("cutoff_year")
    assert table.loc[2019, "development_compounds"] == 1
    assert table.loc[2019, "temporal_test_compounds"] == 1
    assert set(table["selection_status"]) == {"unselected"}
    assert result.manifest["temporal_feasibility"]["selected_cutoff"] is None


def test_missing_assay_metadata_is_fatal(snapshot_tables) -> None:
    activities, assays, parents = snapshot_tables
    with pytest.raises(ValueError, match="Missing joined assay metadata"):
        curate_records(
            activities, assays[:-1], parents, standardizer=fake_standardizer
        )


def test_missing_structure_pipeline_has_clear_error(monkeypatch) -> None:
    def missing(_name: str) -> str:
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(importlib.metadata, "version", missing)
    with pytest.raises(RuntimeError, match="chembl-structure-pipeline==1.2.4"):
        load_chembl_structure_standardizer()


def test_real_structure_pipeline_initializes_ring_information() -> None:
    pytest.importorskip("rdkit")
    try:
        installed = importlib.metadata.version("chembl_structure_pipeline")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("chembl-structure-pipeline is not installed in this environment")
    if installed != "1.2.4":
        pytest.skip(f"requires the protocol-pinned structure pipeline, found {installed}")

    standardizer, used_version = load_chembl_structure_standardizer()
    structure = standardizer("CCOc1ccc(C(=O)N2CCN(CC2)C)cc1")

    assert used_version == "1.2.4"
    assert structure.canonical_parent_smiles
    assert structure.parent_inchi_key
    assert structure.scaffold_smiles != "__ACYCLIC__"


def test_all_structure_failures_produce_auditable_empty_cohorts(
    snapshot_tables,
) -> None:
    activities, assays, parents = snapshot_tables

    def rejected(_smiles: str) -> StandardizedStructure:
        raise ValueError("deliberate structure rejection")

    result = curate_records(
        activities,
        assays,
        parents,
        standardizer=rejected,
        structure_pipeline_version="test-rejection",
    )

    assert result.broad_events.empty
    assert result.strict_events.empty
    assert list(result.strict_events.columns)[0:3] == [
        "event_id",
        "event_signature_sha256",
        "standardized_parent_inchi_key",
    ]
    assert result.manifest["cohort_counts"]["strict"]["compounds"] == 0
    assert result.manifest["input_counts"]["structure_standardization_failures"] == len(
        parents
    )


def test_curate_snapshot_writes_hashed_deterministic_outputs(
    tmp_path: Path, snapshot_tables
) -> None:
    activities, assays, parents = snapshot_tables
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    write_jsonl_gz(snapshot / "activities.jsonl.gz", activities)
    write_jsonl_gz(snapshot / "assays.jsonl.gz", assays)
    write_jsonl_gz(snapshot / "parent_molecules.jsonl.gz", parents)
    files = ["activities.jsonl.gz", "assays.jsonl.gz", "parent_molecules.jsonl.gz"]
    snapshot_manifest = {
        "snapshot_id": "fixture",
        "source": {"release": "fixture", "release_date": "2026-01-01"},
        "files_sha256": {name: sha256_file(snapshot / name) for name in files},
    }
    write_json(snapshot / "snapshot_manifest.json", snapshot_manifest)

    first = curate_snapshot(
        snapshot,
        tmp_path / "first",
        standardizer=fake_standardizer,
        structure_pipeline_version="test-1",
    )
    second = curate_snapshot(
        snapshot,
        tmp_path / "second",
        standardizer=fake_standardizer,
        structure_pipeline_version="test-1",
    )
    assert first["output_files_sha256"] == second["output_files_sha256"]
    assert first["curated_dataset_sha256"] == second["curated_dataset_sha256"]
    assert first["temporal_feasibility"]["selected_cutoff"] is None
    assert pd.read_csv(tmp_path / "first" / "compound_strict.csv").shape[0] == 2
