from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pim1_qsar.audits.temporal_robustness import (
    build_audit_frame,
    group_centered_metrics,
    temporal_degradation_bootstrap,
)


def _frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions = pd.DataFrame(
        {
            "standardized_parent_inchi_key": ["A", "B", "C", "D", "E", "F"],
            "event_ids": ["e1", "e2", "e3", "e4", "e5", "e6"],
            "scaffold_smiles": ["s1", "s1", "s2", "s3", "s4", "s5"],
            "first_record_year": [2019, 2019, 2019, 2024, 2024, 2024],
            "y_true": [5.0, 6.0, 7.0, 5.0, 7.0, 9.0],
            "y_pred": [5.1, 6.2, 6.7, 6.0, 8.0, 10.0],
            "absolute_error": [0.1, 0.2, 0.3, 1.0, 1.0, 1.0],
            "nearest_development_identity": ["T"] * 6,
            "max_tanimoto": np.linspace(0.1, 0.6, 6),
            "mean_5nn_tanimoto": np.linspace(0.05, 0.3, 6),
        }
    )
    development = pd.DataFrame(
        {"standardized_parent_inchi_key": ["T"], "pIC50": [6.0]}
    )
    strict = pd.DataFrame(
        {
            "event_id": [f"e{i}" for i in range(1, 7)],
            "document_chembl_id": ["d1"] * 3 + ["d2"] * 3,
            "representative_assay_chembl_id": ["a1"] * 3 + ["a2"] * 3,
        }
    )
    broad = pd.DataFrame(
        {
            "standardized_parent_inchi_key": ["A", "A", "B", "C", "D", "E", "F"],
            "document_year": [2017, 2019, 2019, 2019, 2024, 2024, 2024],
        }
    )
    return predictions, development, strict, broad


def test_build_audit_frame_attaches_sources_chronology_and_neighbor_gap() -> None:
    frame = build_audit_frame(*_frames(), cutoff_year=2018)
    assert frame["broad_record_by_cutoff"].tolist() == [True, False, False, False, False, False]
    assert frame["document_cluster"].tolist() == ["d1", "d1", "d1", "d2", "d2", "d2"]
    assert frame["nearest_neighbor_potency_gap"].tolist() == [1.0, 0.0, 1.0, 1.0, 1.0, 3.0]


def test_degradation_bootstrap_reports_positive_endpoint_change() -> None:
    frame = build_audit_frame(*_frames(), cutoff_year=2018)
    result = temporal_degradation_bootstrap(
        frame,
        "scaffold_smiles",
        first_year=2019,
        last_year=2024,
        n_bootstrap=200,
        random_state=7,
    )
    assert result["rmse_delta"]["point_estimate"] > 0.7
    assert result["absolute_error_linear_slope_per_year"]["point_estimate"] > 0


def test_group_centering_removes_source_intercepts() -> None:
    frame = build_audit_frame(*_frames(), cutoff_year=2018)
    result = group_centered_metrics(frame, "document_cluster", minimum_group_size=3)
    assert result["n"] == 6
    assert result["n_groups"] == 2
    assert np.isfinite(result["centered_R2"])
    assert np.isfinite(result["centered_spearman_rho"])
