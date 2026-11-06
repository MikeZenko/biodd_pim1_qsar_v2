from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pim1_qsar.artifacts import (
    ArtifactLineage,
    atomic_write_json,
    build_cache_manifest,
    cache_manifest_matches,
    sha256_named_files,
)


def _files(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "dataset": tmp_path / "dataset.csv",
        "splits": tmp_path / "splits.json",
        "protocol": tmp_path / "protocol.yaml",
        "environment": tmp_path / "requirements.txt",
        "code_a": tmp_path / "a.py",
        "code_b": tmp_path / "b.py",
    }
    for name, path in paths.items():
        path.write_text(f"{name}-v1\n", encoding="utf-8")
    return paths


def _lineage(paths: dict[str, Path], seed: int = 42) -> ArtifactLineage:
    return ArtifactLineage.from_files(
        dataset=paths["dataset"],
        split_manifest=paths["splits"],
        protocol=paths["protocol"],
        environment=paths["environment"],
        code_files={"src/a.py": paths["code_a"], "src/b.py": paths["code_b"]},
        seed=seed,
    )


def test_identical_inputs_produce_identical_cache_keys(tmp_path):
    paths = _files(tmp_path)
    first = _lineage(paths)
    second = _lineage(paths)

    assert first == second
    assert first.cache_key == second.cache_key
    assert first.run_id == second.run_id


def test_every_lineage_input_invalidates_the_cache(tmp_path):
    paths = _files(tmp_path)
    baseline = _lineage(paths)
    mutable_inputs = ["dataset", "splits", "protocol", "environment", "code_a", "code_b"]

    for name in mutable_inputs:
        original = paths[name].read_text(encoding="utf-8")
        paths[name].write_text(original + "changed\n", encoding="utf-8")
        assert _lineage(paths).cache_key != baseline.cache_key
        paths[name].write_text(original, encoding="utf-8")
    assert _lineage(paths, seed=43).cache_key != baseline.cache_key


def test_code_hash_is_independent_of_mapping_order(tmp_path):
    paths = _files(tmp_path)
    forward = sha256_named_files(
        {"src/a.py": paths["code_a"], "src/b.py": paths["code_b"]}
    )
    reverse = sha256_named_files(
        {"src/b.py": paths["code_b"], "src/a.py": paths["code_a"]}
    )
    assert forward == reverse


def test_cache_manifest_requires_exact_complete_lineage(tmp_path):
    paths = _files(tmp_path)
    lineage = _lineage(paths)
    complete = build_cache_manifest(lineage, artifact_type="outer_fold")
    incomplete = build_cache_manifest(
        lineage,
        artifact_type="outer_fold",
        status="incomplete",
    )

    assert cache_manifest_matches(complete, lineage)
    assert not cache_manifest_matches(incomplete, lineage)
    assert cache_manifest_matches(incomplete, lineage, require_complete=False)
    assert not cache_manifest_matches(complete, _lineage(paths, seed=999))


def test_atomic_json_write_leaves_no_temporary_file(tmp_path):
    destination = tmp_path / "nested" / "manifest.json"
    payload = {"z": 2, "a": [1, 2, 3]}

    atomic_write_json(destination, payload)

    assert destination.read_text(encoding="utf-8") == (
        '{\n  "a": [\n    1,\n    2,\n    3\n  ],\n  "z": 2\n}\n'
    )
    assert list(destination.parent.glob("*.tmp")) == []
