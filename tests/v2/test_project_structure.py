from __future__ import annotations

from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGACY_ROOT = PROJECT_ROOT / "archive" / "legacy_v1"


def test_active_root_does_not_mix_legacy_pipeline() -> None:
    forbidden = {
        "1.Download_Bioactivity_Data.py",
        "paper.tex",
        "paper.pdf",
        "models",
        "plots",
        "results",
        "run_pipeline.sh",
        "scaffold_utils.py",
    }
    present = {path.name for path in PROJECT_ROOT.iterdir()}
    assert forbidden.isdisjoint(present)


def test_canonical_v2_roots_are_present() -> None:
    for relative in (
        "configs",
        "data/releases",
        "data/curated",
        "src/pim1_qsar",
        "scripts",
        "tests/v2",
        "artifacts/runs",
        "manuscript_v2",
    ):
        assert (PROJECT_ROOT / relative).is_dir(), relative


def test_legacy_snapshot_is_self_contained() -> None:
    for relative in (
        "1.Download_Bioactivity_Data.py",
        "7.Model_Interpretability.py",
        "preprocessing_utils.py",
        "scaffold_utils.py",
        "data",
        "models/CHEMBL2147",
        "results/CHEMBL2147",
        "plots/CHEMBL2147",
        "figs",
        "paper.tex",
        "paper.pdf",
        "tests",
    ):
        assert (LEGACY_ROOT / relative).exists(), relative


def test_legacy_archive_manifest_verifies() -> None:
    completed = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts/12_manage_archive_manifest.py")],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
