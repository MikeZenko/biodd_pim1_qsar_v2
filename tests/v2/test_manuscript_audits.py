from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "08_build_manuscript_assets.py"


def _load_manuscript_assets_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_manuscript_assets", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load manuscript asset builder: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def manuscript_assets() -> ModuleType:
    return _load_manuscript_assets_module()


def _synthetic_assay_events() -> pd.DataFrame:
    matched_pic50 = -np.log10(9e-9)
    rows = [
        {
            "event_id": "event-1",
            "representative_assay_chembl_id": "CHEMBL_A1",
            "document_chembl_id": "CHEMBL_D1",
            "document_year": 2018,
            "assay_description": "PIM1 inhibition in the presence of 20 uM ATP",
            "standardized_parent_inchi_key": "KEY-1",
            "standard_value_nm": 10.0,
            "pIC50": 8.0,
        },
        {
            "event_id": "event-2",
            "representative_assay_chembl_id": "CHEMBL_A1",
            "document_chembl_id": "CHEMBL_D1",
            "document_year": 2018,
            "assay_description": "PIM1 inhibition in the presence of 20 uM ATP",
            "standardized_parent_inchi_key": "KEY-2",
            "standard_value_nm": 31.6227766017,
            "pIC50": 7.5,
        },
        {
            "event_id": "event-3",
            "representative_assay_chembl_id": "CHEMBL_A2",
            "document_chembl_id": "CHEMBL_D2",
            "document_year": 2018,
            "assay_description": "Inhibition of PIM1 in the presence of 50 µM ATP",
            "standardized_parent_inchi_key": "MATCHED-KEY",
            "standard_value_nm": 9.0,
            "pIC50": matched_pic50,
        },
        {
            "event_id": "event-4",
            "representative_assay_chembl_id": "CHEMBL_A3",
            "document_chembl_id": "CHEMBL_D2",
            "document_year": 2018,
            "assay_description": "Inhibition of PIM1 in the presence of 5 mM ATP",
            "standardized_parent_inchi_key": "MATCHED-KEY",
            "standard_value_nm": 100.0,
            "pIC50": 7.0,
        },
        {
            "event_id": "event-5",
            "representative_assay_chembl_id": "CHEMBL_A4",
            "document_chembl_id": "CHEMBL_D3",
            "document_year": 2020,
            "assay_description": "PIM1 inhibition at 500 nM ATP",
            "standardized_parent_inchi_key": "KEY-3",
            "standard_value_nm": 1000.0,
            "pIC50": 6.0,
        },
        {
            "event_id": "event-6",
            "representative_assay_chembl_id": "CHEMBL_A5",
            "document_chembl_id": "CHEMBL_D4",
            "document_year": 2020,
            "assay_description": "PIM1 inhibition with ATP at Km",
            "standardized_parent_inchi_key": "KEY-4",
            "standard_value_nm": 100.0,
            "pIC50": 7.0,
        },
        {
            "event_id": "event-7",
            "representative_assay_chembl_id": "CHEMBL_A6",
            "document_chembl_id": "CHEMBL_D5",
            "document_year": 2021,
            "assay_description": "PIM1 biochemical inhibition assay",
            "standardized_parent_inchi_key": "KEY-5",
            "standard_value_nm": 100.0,
            "pIC50": 7.0,
        },
    ]
    frame = pd.DataFrame(rows)
    frame["scaffold_smiles"] = [
        "c1ccccc1",
        "c1ccncc1",
        "c1ccc2ccccc2c1",
        "c1ccc2ccccc2c1",
        "C1CCCCC1",
        "c1cncnc1",
        "[ACYCLIC]",
    ]
    return frame


def _synthetic_partitioned_metrics(manuscript_assets: ModuleType) -> dict[str, object]:
    return manuscript_assets.assay_context_metrics(
        _synthetic_assay_events(),
        development_identities={"KEY-1", "KEY-2", "MATCHED-KEY"},
        temporal_identities={"KEY-3", "KEY-4", "KEY-5"},
        cutoff_year=2018,
    )


def _write_checkpoint(
    path: Path,
    *,
    inner_best_epochs: list[int],
    refit_epochs: int,
) -> None:
    payload = {
        "schema_version": "gnn_nested_v2",
        "temporal_test_consulted": False,
        "metrics": {
            "model": path.stem,
            "repeat": 0,
            "fold": 0,
            "inner_best_epochs": inner_best_epochs,
            "refit_epochs": refit_epochs,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_assay_context_audit_normalizes_atp_units_and_counts_metadata(
    manuscript_assets: ModuleType,
) -> None:
    metrics = _synthetic_partitioned_metrics(manuscript_assets)

    assert metrics["status"] == "post_hoc_descriptive"
    assert metrics["event_count"] == 7
    assert metrics["assay_count"] == 6
    assert metrics["document_count"] == 5
    assert metrics["description_count"] == 6
    assert metrics["atp_mention_assay_count"] == 5
    assert metrics["atp_mention_event_count"] == 6
    assert metrics["numeric_atp_assay_count"] == 4
    assert metrics["numeric_atp_event_count"] == 5
    assert metrics["numeric_atp_min_um"] == pytest.approx(0.5)
    assert metrics["numeric_atp_max_um"] == pytest.approx(5000.0)
    assert metrics["numeric_atp_fold_range"] == pytest.approx(10_000.0)
    assert metrics["development_temporal_assay_intersection_count"] == 0
    assert metrics["development_temporal_document_intersection_count"] == 0


def test_assay_context_audit_reports_the_matched_concentration_pair(
    manuscript_assets: ModuleType,
) -> None:
    metrics = manuscript_assets.assay_context_metrics(_synthetic_assay_events())

    assert metrics["matched_numeric_atp_pair_count"] == 1
    # The defined contrast is higher-ATP minus lower-ATP pIC50.
    matched_pair = metrics["matched_numeric_atp_pairs"][0]
    assert matched_pair["pic50_delta_high_minus_low_atp"] == pytest.approx(
        7.0 - (-np.log10(9e-9)), abs=1e-12
    )


def test_assay_context_audit_rejects_missing_or_invalid_fields(
    manuscript_assets: ModuleType,
) -> None:
    events = _synthetic_assay_events()
    with pytest.raises(ValueError, match="missing|columns"):
        manuscript_assets.assay_context_metrics(
            events.drop(columns=["assay_description"])
        )

    invalid_identifier = events.copy()
    invalid_identifier.loc[0, "representative_assay_chembl_id"] = None
    with pytest.raises(ValueError, match="assay|missing|null|finite"):
        manuscript_assets.assay_context_metrics(invalid_identifier)

    invalid_potency = events.copy()
    invalid_potency.loc[0, "pIC50"] = np.inf
    with pytest.raises(ValueError, match="pIC50|finite|invalid"):
        manuscript_assets.assay_context_metrics(invalid_potency)


def test_gnn_duration_audit_counts_epochs_and_cap_hits(
    manuscript_assets: ModuleType, tmp_path: Path
) -> None:
    _write_checkpoint(
        tmp_path / "model-a.json", inner_best_epochs=[1, 5, 10], refit_epochs=5
    )
    _write_checkpoint(
        tmp_path / "model-b.json", inner_best_epochs=[2, 10, 7], refit_epochs=10
    )

    metrics = manuscript_assets.gnn_training_duration_metrics(tmp_path, max_epochs=10)

    assert metrics["max_epochs"] == 10
    assert metrics["checkpoint_count"] == 2
    assert metrics["inner_epoch_count"] == 6
    assert metrics["inner_epoch_min"] == 1
    assert metrics["inner_epoch_max"] == 10
    assert metrics["inner_epoch_cap_hit_count"] == 2
    assert metrics["refit_epoch_count"] == 2
    assert metrics["refit_epoch_min"] == 5
    assert metrics["refit_epoch_max"] == 10
    assert metrics["refit_epoch_cap_hit_count"] == 1
    assert metrics["total_epoch_cap_hit_count"] == 3


def test_gnn_duration_audit_rejects_malformed_and_incomplete_checkpoints(
    manuscript_assets: ModuleType, tmp_path: Path
) -> None:
    malformed = tmp_path / "malformed"
    malformed.mkdir()
    (malformed / "bad.json").write_text("{not-json", encoding="utf-8")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        manuscript_assets.gnn_training_duration_metrics(malformed, max_epochs=80)

    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / "missing-epochs.json").write_text(
        json.dumps(
            {
                "schema_version": "gnn_nested_v2",
                "temporal_test_consulted": False,
                "metrics": {"refit_epochs": 5},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises((KeyError, ValueError), match="inner_best_epochs|checkpoint"):
        manuscript_assets.gnn_training_duration_metrics(incomplete, max_epochs=80)

    invalid_epoch = tmp_path / "invalid-epoch"
    invalid_epoch.mkdir()
    _write_checkpoint(
        invalid_epoch / "over-cap.json",
        inner_best_epochs=[1, 81, 2],
        refit_epochs=2,
    )
    with pytest.raises(ValueError, match="epoch|cap|max"):
        manuscript_assets.gnn_training_duration_metrics(invalid_epoch, max_epochs=80)


def test_pinned_artifact_audits_match_the_sealed_inventories(
    manuscript_assets: ModuleType,
) -> None:
    events = pd.read_csv(
        PROJECT_ROOT
        / "data"
        / "curated"
        / "chembl37_pim1_v2"
        / "strict_measurement_events.csv"
    )
    split_dir = (
        PROJECT_ROOT
        / "artifacts"
        / "split_locks"
        / "chembl37_strict_temporal2018_scaffold3x5_v1"
    )
    identity = "standardized_parent_inchi_key"
    development_identities = set(pd.read_csv(split_dir / "development.csv")[identity])
    temporal_identities = set(
        pd.read_csv(split_dir / "temporal_test_sealed.csv")[identity]
    )
    assay = manuscript_assets.assay_context_metrics(
        events,
        development_identities=development_identities,
        temporal_identities=temporal_identities,
        cutoff_year=2018,
    )

    assert assay["event_count"] == 1125
    assert assay["assay_count"] == 122
    assert assay["document_count"] == 102
    assert assay["numeric_atp_assay_count"] == 8
    assert assay["numeric_atp_event_count"] == 117
    assert assay["numeric_atp_min_um"] == pytest.approx(20.0)
    assert assay["numeric_atp_max_um"] == pytest.approx(5000.0)
    assert assay["numeric_atp_fold_range"] == pytest.approx(250.0)
    assert assay["development_temporal_assay_intersection_count"] == 0
    assert assay["development_temporal_document_intersection_count"] == 0

    gnn = manuscript_assets.gnn_training_duration_metrics(
        PROJECT_ROOT / "artifacts" / "runs" / "gnn_nested_v2" / "checkpoints",
        max_epochs=80,
    )
    assert gnn["checkpoint_count"] == 60
    assert gnn["inner_epoch_count"] == 180
    assert gnn["inner_epoch_max"] == 63
    assert gnn["refit_epoch_count"] == 60
    assert gnn["refit_epoch_max"] == 43
    assert gnn["total_epoch_cap_hit_count"] == 0

    robustness_dir = (
        PROJECT_ROOT / "artifacts" / "audits" / "temporal_robustness_v2"
    )
    robustness = json.loads(
        (robustness_dir / "audit_metrics.json").read_text(encoding="utf-8")
    )
    assert robustness["status"] == "post_hoc_descriptive"
    assert robustness["broad_record_chronology"]["count"] == 4
    assert robustness["broad_record_chronology"]["sensitivity_point_metrics"][
        "RMSE"
    ] == pytest.approx(0.9761450824700881)
    assert robustness["source_centered_ordering"]["document"][
        "centered_R2"
    ] == pytest.approx(-0.03119815625437994)
    assert robustness["nearest_neighbor_potency_discordance"][
        "point_estimate"
    ] == pytest.approx(0.27124505767166085)
