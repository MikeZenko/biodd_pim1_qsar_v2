#!/usr/bin/env python3
"""Verify and seal the compiled v2 manuscript without modifying evidence.

The numerical manuscript assets are already generated from the immutable split,
development selection, and one-time temporal evaluation. This final stage only
verifies their hashes and records the exact compiled PDF and source inventory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import tempfile
import os
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANUSCRIPT_DIR = PROJECT_ROOT / "manuscript_v2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def verify_generated_assets(manuscript_dir: Path) -> dict[str, Any]:
    manifest_path = manuscript_dir / "generated" / "asset_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != "manuscript_assets_v2":
        raise RuntimeError("Unexpected manuscript asset manifest schema")
    output_hashes = manifest.get("output_hashes")
    if not isinstance(output_hashes, Mapping) or not output_hashes:
        raise RuntimeError("Asset manifest has no output hash inventory")
    for relative, expected in output_hashes.items():
        path = manuscript_dir / str(relative)
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(f"Generated manuscript asset checksum mismatch: {relative}")
    return manifest


def verify_compile_log(log_path: Path) -> int:
    log = log_path.read_text(encoding="utf-8", errors="replace")
    prohibited = (
        "Citation(s) may have changed",
        "There were undefined citations",
        "There were undefined references",
        "Label(s) may have changed",
        "Rerun to get cross-references right",
        "Overfull \\hbox",
        "Underfull \\hbox",
    )
    found = [phrase for phrase in prohibited if phrase in log]
    if found:
        raise RuntimeError("Compilation log is not clean: " + "; ".join(found))
    match = re.search(r"Output written on main\.pdf \((\d+) pages,", log)
    if match is None:
        raise RuntimeError("Could not determine PDF page count from compilation log")
    return int(match.group(1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manuscript-dir", type=Path, default=DEFAULT_MANUSCRIPT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manuscript_dir = args.manuscript_dir.resolve()
    required = {
        "source": manuscript_dir / "main.tex",
        "bibliography": manuscript_dir / "references.bib",
        "bibliography_output": manuscript_dir / "main.bbl",
        "compile_log": manuscript_dir / "main.log",
        "pdf": manuscript_dir / "main.pdf",
    }
    for path in required.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    assets = verify_generated_assets(manuscript_dir)
    page_count = verify_compile_log(required["compile_log"])
    manifest_core = {
        "schema_version": "sealed_manuscript_v2",
        "selected_model": assets["selected_model"],
        "temporal_evaluation_sha256": assets["temporal_evaluation_sha256"],
        "page_count": page_count,
        "asset_manifest_sha256": sha256_file(manuscript_dir / "generated" / "asset_manifest.json"),
        "files_sha256": {
            label: sha256_file(path) for label, path in sorted(required.items())
        },
    }
    manifest = dict(manifest_core)
    manifest["manuscript_sha256"] = sha256_json(manifest_core)
    atomic_write_json(manuscript_dir / "manuscript_manifest.json", manifest)
    marker = manuscript_dir / "MANUSCRIPT_COMPLETE"
    marker.write_text(manifest["manuscript_sha256"] + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
