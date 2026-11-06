from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pim1_qsar.data.chembl_snapshot import (  # noqa: E402
    SnapshotConfig,
    minimize_molecule,
    sha256_file,
    validate_status,
    write_jsonl_gz,
)


def test_release_guard_rejects_wrong_version() -> None:
    config = SnapshotConfig()
    with pytest.raises(RuntimeError, match="release mismatch"):
        validate_status(
            {"chembl_db_version": "ChEMBL_36", "chembl_release_date": "2025-07-01"},
            config,
        )


def test_minimize_molecule_drops_molfile() -> None:
    minimal = minimize_molecule(
        {
            "molecule_chembl_id": "CHEMBL1",
            "molecule_hierarchy": {"parent_chembl_id": "CHEMBL1"},
            "molecule_structures": {
                "canonical_smiles": "CCO",
                "standard_inchi": "InChI=1S/C2H6O",
                "standard_inchi_key": "KEY",
                "molfile": "large payload",
            },
        }
    )
    assert minimal["canonical_smiles"] == "CCO"
    assert "molfile" not in minimal


def test_jsonl_gzip_is_byte_deterministic(tmp_path: Path) -> None:
    records = [{"b": 2, "a": 1}, {"a": 3}]
    first = tmp_path / "first.jsonl.gz"
    second = tmp_path / "second.jsonl.gz"
    write_jsonl_gz(first, records)
    write_jsonl_gz(second, records)
    assert sha256_file(first) == sha256_file(second)
    with gzip.open(first, "rt", encoding="utf-8") as handle:
        decoded = [json.loads(line) for line in handle]
    assert decoded == records
