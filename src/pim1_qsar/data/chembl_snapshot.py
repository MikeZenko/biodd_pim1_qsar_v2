"""Create a deterministic, release-verified ChEMBL snapshot for PIM1.

The live ChEMBL API always points at the current database release.  This module
therefore refuses to download unless the API reports the exact release named in
the protocol.  The resulting compact snapshot is the immutable input for every
downstream curation and modelling step; future reproduction starts from these
files rather than querying the live API again.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


API_BASE = "https://www.ebi.ac.uk/chembl/api/data"
FTP_RELEASE_BASE = (
    "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_37"
)


@dataclass(frozen=True)
class SnapshotConfig:
    """Immutable acquisition settings for the primary PIM1 snapshot."""

    target_chembl_id: str = "CHEMBL2147"
    expected_release: str = "ChEMBL_37"
    expected_release_date: str = "2026-05-01"
    standard_type: str = "IC50"
    standard_relation: str = "="
    standard_units: str = "nM"
    standard_flag: int = 1
    page_size: int = 1000
    molecule_chunk_size: int = 40
    timeout_seconds: int = 120
    retries: int = 5

    @property
    def activity_filters(self) -> dict[str, Any]:
        return {
            "target_chembl_id": self.target_chembl_id,
            "standard_type": self.standard_type,
            "standard_relation": self.standard_relation,
            "standard_units": self.standard_units,
            "standard_flag": self.standard_flag,
            "standard_value__isnull": "false",
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _request_bytes(
    url: str,
    *,
    timeout_seconds: int,
    retries: int,
) -> bytes:
    headers = {
        "Accept": "application/json",
        "User-Agent": "pim1-qsar-publication-snapshot/2",
    }
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status} for {url}")
                return response.read()
        except Exception as exc:  # network failures are retried deterministically
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Unable to retrieve {url} after {retries} attempts") from last_error


def _request_json(
    resource: str,
    params: Mapping[str, Any] | None,
    *,
    config: SnapshotConfig,
) -> dict[str, Any]:
    base = resource if resource.startswith("https://") else f"{API_BASE}/{resource}"
    if params:
        query = urllib.parse.urlencode(params, doseq=True, safe=",")
        url = f"{base}?{query}"
    else:
        url = base
    payload = _request_bytes(
        url,
        timeout_seconds=config.timeout_seconds,
        retries=config.retries,
    )
    parsed = json.loads(payload.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise TypeError(f"Expected an object response from {url}")
    return parsed


def validate_status(status: Mapping[str, Any], config: SnapshotConfig) -> None:
    release = status.get("chembl_db_version")
    release_date = status.get("chembl_release_date")
    if release != config.expected_release:
        raise RuntimeError(
            f"ChEMBL release mismatch: expected {config.expected_release}, got {release}. "
            "Use the checked-in snapshot or deliberately update the protocol."
        )
    if release_date != config.expected_release_date:
        raise RuntimeError(
            "ChEMBL release-date mismatch: "
            f"expected {config.expected_release_date}, got {release_date}."
        )


def fetch_paginated(
    resource: str,
    collection_key: str,
    params: Mapping[str, Any],
    *,
    config: SnapshotConfig,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    offset = 0
    total_count: int | None = None
    while total_count is None or offset < total_count:
        page_params = dict(params)
        page_params.update({"limit": config.page_size, "offset": offset})
        page = _request_json(resource, page_params, config=config)
        page_records = page.get(collection_key)
        if not isinstance(page_records, list):
            raise RuntimeError(f"Missing list '{collection_key}' in {resource} response")
        meta = page.get("page_meta", {})
        reported_total = meta.get("total_count")
        if reported_total is None:
            raise RuntimeError(f"Missing total_count in {resource} response")
        if total_count is None:
            total_count = int(reported_total)
        elif total_count != int(reported_total):
            raise RuntimeError(f"total_count changed while paging {resource}")
        records.extend(page_records)
        if not page_records:
            break
        offset += len(page_records)
    if total_count is None or len(records) != total_count:
        raise RuntimeError(
            f"Incomplete {resource} download: expected {total_count}, got {len(records)}"
        )
    return records


def _chunks(values: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def minimize_molecule(record: Mapping[str, Any]) -> dict[str, Any]:
    structures = record.get("molecule_structures") or {}
    hierarchy = record.get("molecule_hierarchy") or {}
    return {
        "molecule_chembl_id": record.get("molecule_chembl_id"),
        "parent_chembl_id": hierarchy.get("parent_chembl_id"),
        "canonical_smiles": structures.get("canonical_smiles"),
        "standard_inchi": structures.get("standard_inchi"),
        "standard_inchi_key": structures.get("standard_inchi_key"),
    }


def fetch_parent_molecules(
    parent_ids: Sequence[str],
    *,
    config: SnapshotConfig,
) -> list[dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for chunk in _chunks(sorted(set(parent_ids)), config.molecule_chunk_size):
        records = fetch_paginated(
            "molecule.json",
            "molecules",
            {
                "molecule_chembl_id__in": ",".join(chunk),
                "order_by": "molecule_chembl_id",
            },
            config=config,
        )
        for record in records:
            minimal = minimize_molecule(record)
            molecule_id = minimal.get("molecule_chembl_id")
            if not molecule_id:
                raise RuntimeError("Molecule response omitted molecule_chembl_id")
            results[str(molecule_id)] = minimal
    missing = sorted(set(parent_ids) - set(results))
    if missing:
        raise RuntimeError(f"Missing {len(missing)} parent molecule records: {missing[:5]}")
    return [results[key] for key in sorted(results)]


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_write_bytes(path, payload)


def write_jsonl_gz(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="", fileobj=raw_handle, mode="wb", mtime=0
        ) as compressed:
            for record in records:
                line = json.dumps(record, sort_keys=True, separators=(",", ":"))
                compressed.write(line.encode("utf-8") + b"\n")
    os.replace(temporary, path)


def build_snapshot(output_dir: Path, config: SnapshotConfig = SnapshotConfig()) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    status = _request_json("status.json", None, config=config)
    validate_status(status, config)

    target = _request_json(f"target/{config.target_chembl_id}.json", None, config=config)
    if target.get("target_chembl_id") != config.target_chembl_id:
        raise RuntimeError("Target response does not match requested ChEMBL target")
    if target.get("target_type") != "SINGLE PROTEIN":
        raise RuntimeError(f"Unexpected target type: {target.get('target_type')}")

    activities = fetch_paginated(
        "activity.json",
        "activities",
        {**config.activity_filters, "order_by": "activity_id"},
        config=config,
    )
    activities.sort(key=lambda record: int(record["activity_id"]))
    activity_ids = [int(record["activity_id"]) for record in activities]
    if len(activity_ids) != len(set(activity_ids)):
        raise RuntimeError("Duplicate activity_id values in ChEMBL response")

    assays = fetch_paginated(
        "assay.json",
        "assays",
        {
            "target_chembl_id": config.target_chembl_id,
            "order_by": "assay_chembl_id",
        },
        config=config,
    )
    assays.sort(key=lambda record: str(record["assay_chembl_id"]))
    assay_ids = {str(record["assay_chembl_id"]) for record in assays}
    activity_assay_ids = {str(record["assay_chembl_id"]) for record in activities}
    missing_assays = sorted(activity_assay_ids - assay_ids)
    if missing_assays:
        raise RuntimeError(f"Assay endpoint omitted referenced assays: {missing_assays[:5]}")

    parent_ids = [
        str(record.get("parent_molecule_chembl_id") or record["molecule_chembl_id"])
        for record in activities
    ]
    parents = fetch_parent_molecules(parent_ids, config=config)

    official_files = {
        "CHEMBL_README.txt": f"{FTP_RELEASE_BASE}/README",
        "CHEMBL_RELEASE_NOTES.txt": f"{FTP_RELEASE_BASE}/chembl_37_release_notes.txt",
        "CHEMBL_FTP_CHECKSUMS.txt": f"{FTP_RELEASE_BASE}/checksums.txt",
    }
    for filename, url in official_files.items():
        payload = _request_bytes(
            url,
            timeout_seconds=config.timeout_seconds,
            retries=config.retries,
        )
        _atomic_write_bytes(output_dir / filename, payload)

    write_json(output_dir / "chembl_status.json", status)
    write_json(output_dir / "target.json", target)
    write_jsonl_gz(output_dir / "activities.jsonl.gz", activities)
    write_jsonl_gz(output_dir / "assays.jsonl.gz", assays)
    write_jsonl_gz(output_dir / "parent_molecules.jsonl.gz", parents)

    data_files = [
        "CHEMBL_README.txt",
        "CHEMBL_RELEASE_NOTES.txt",
        "CHEMBL_FTP_CHECKSUMS.txt",
        "chembl_status.json",
        "target.json",
        "activities.jsonl.gz",
        "assays.jsonl.gz",
        "parent_molecules.jsonl.gz",
    ]
    manifest = {
        "schema_version": 1,
        "snapshot_id": "chembl37_pim1_ic50_20260501",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "api_base": API_BASE,
            "ftp_release_base": FTP_RELEASE_BASE,
            "release": config.expected_release,
            "release_date": config.expected_release_date,
            "release_status": status,
        },
        "query": {
            "resource": "activity",
            "filters": config.activity_filters,
            "target_resource": config.target_chembl_id,
            "assay_resource_filter": {"target_chembl_id": config.target_chembl_id},
            "parent_structure_source": "molecule endpoint for parent_molecule_chembl_id",
        },
        "counts": {
            "activities": len(activities),
            "assays_for_target": len(assays),
            "referenced_assays": len(activity_assay_ids),
            "parent_molecules": len(parents),
        },
        "files_sha256": {
            filename: sha256_file(output_dir / filename) for filename in data_files
        },
    }
    write_json(output_dir / "snapshot_manifest.json", manifest)
    return manifest
