#!/usr/bin/env python3
"""Write or verify the preserved legacy-v1 archive inventory.

Only `archive/legacy_v1/` is inventoried. Non-evidentiary `archive/workspaces/`
and `archive/local_state/` caches are not part of the public tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = PROJECT_ROOT / "archive" / "legacy_v1"
MANIFEST_NAME = "ARCHIVE_MANIFEST.json"
# Match root .gitignore so local LaTeX aux, editor files, and logs cannot
# appear in a clone and then fail public verification.
SKIP_NAMES = {".DS_Store", "Thumbs.db"}
SKIP_SUFFIXES = {
    ".aux",
    ".bcf",
    ".blg",
    ".bbl",
    ".fdb_latexmk",
    ".fls",
    ".lof",
    ".log",
    ".lot",
    ".out",
    ".run.xml",
    ".swp",
    ".synctex.gz",
    ".toc",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(archive: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in sorted(archive.rglob("*")):
        if not path.is_file() or path.name == MANIFEST_NAME:
            continue
        if path.name in SKIP_NAMES or path.suffix in SKIP_SUFFIXES:
            continue
        relative = path.relative_to(archive).as_posix()
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": "legacy_archive_manifest_v1",
        "archive_id": "pim1_qsar_legacy_v1_20260720",
        "classification": "historical_noncanonical",
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "inventory_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "files": files,
    }


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument(
        "--write",
        action="store_true",
        help="replace the manifest after an intentional archive change",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive = args.archive.resolve()
    if not archive.is_dir():
        raise FileNotFoundError(archive)
    observed = inventory(archive)
    manifest_path = archive / MANIFEST_NAME
    if args.write:
        atomic_write(manifest_path, observed)
        print(
            f"Wrote {manifest_path}: {observed['file_count']} files, "
            f"{observed['total_bytes']} bytes"
        )
        return
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing {manifest_path}; initialize it explicitly with --write"
        )
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    if observed != expected:
        raise RuntimeError(
            "Legacy archive differs from its manifest; inspect the change before "
            "running with --write"
        )
    print(
        f"Verified {observed['file_count']} archived files, "
        f"{observed['total_bytes']} bytes, inventory "
        f"{observed['inventory_sha256']}"
    )


if __name__ == "__main__":
    main()
