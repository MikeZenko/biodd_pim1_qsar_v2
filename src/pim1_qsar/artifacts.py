"""Content-addressed artifact lineage and atomic manifest helpers.

The v2 pipeline never decides cache validity from a seed or filename alone.
Every reusable artifact is keyed by the frozen dataset, split manifest,
protocol, source code, environment, and stochastic seed that produced it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping


SCHEMA_VERSION = "2.0"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON-compatible data deterministically for hashing."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    """Return the SHA-256 hex digest for *value*."""

    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Return a deterministic SHA-256 digest for JSON-compatible data."""

    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: str | Path) -> str:
    """Hash a file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_named_files(files: Mapping[str, str | Path]) -> str:
    """Hash a labeled collection of files independent of mapping order.

    Labels, rather than absolute paths, are included so that the digest is
    stable when a frozen run is copied to another machine.
    """

    entries = [
        {"label": str(label), "sha256": sha256_file(path)}
        for label, path in sorted(files.items(), key=lambda item: str(item[0]))
    ]
    return sha256_json(entries)


def _validate_digest(name: str, value: str) -> None:
    if not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


@dataclass(frozen=True)
class ArtifactLineage:
    """Complete cache identity for one v2 artifact lineage."""

    dataset_sha256: str
    split_manifest_sha256: str
    protocol_sha256: str
    code_sha256: str
    environment_sha256: str
    seed: int
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "dataset_sha256",
            "split_manifest_sha256",
            "protocol_sha256",
            "code_sha256",
            "environment_sha256",
        ):
            _validate_digest(field_name, getattr(self, field_name))
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported lineage schema: {self.schema_version}")

    @classmethod
    def from_files(
        cls,
        *,
        dataset: str | Path,
        split_manifest: str | Path,
        protocol: str | Path,
        code_files: Mapping[str, str | Path],
        environment: str | Path,
        seed: int,
    ) -> "ArtifactLineage":
        """Construct lineage from the files that define a computation."""

        if not code_files:
            raise ValueError("code_files cannot be empty")
        return cls(
            dataset_sha256=sha256_file(dataset),
            split_manifest_sha256=sha256_file(split_manifest),
            protocol_sha256=sha256_file(protocol),
            code_sha256=sha256_named_files(code_files),
            environment_sha256=sha256_file(environment),
            seed=seed,
        )

    @property
    def cache_key(self) -> str:
        """Content-addressed cache key for this exact computation."""

        return sha256_json(self.as_dict())

    @property
    def run_id(self) -> str:
        """Short human-readable identifier; full hashes remain in manifests."""

        return f"v2-{self.dataset_sha256[:8]}-{self.protocol_sha256[:8]}-{self.cache_key[:12]}"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_cache_manifest(
    lineage: ArtifactLineage,
    *,
    artifact_type: str,
    status: str = "complete",
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a manifest that can be checked before an artifact is reused."""

    if not artifact_type.strip():
        raise ValueError("artifact_type cannot be empty")
    if status not in {"complete", "incomplete", "failed"}:
        raise ValueError(f"Unsupported artifact status: {status}")
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": artifact_type,
        "status": status,
        "cache_key": lineage.cache_key,
        "lineage": lineage.as_dict(),
        "metadata": dict(metadata or {}),
    }


def cache_manifest_matches(
    manifest: Mapping[str, Any],
    lineage: ArtifactLineage,
    *,
    require_complete: bool = True,
) -> bool:
    """Return whether a manifest belongs to the exact requested lineage."""

    if manifest.get("schema_version") != SCHEMA_VERSION:
        return False
    if require_complete and manifest.get("status") != "complete":
        return False
    return (
        manifest.get("cache_key") == lineage.cache_key
        and manifest.get("lineage") == lineage.as_dict()
    )


def atomic_write_bytes(path: str | Path, payload: bytes) -> None:
    """Atomically replace *path* with *payload* on the same filesystem."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_text(path: str | Path, text: str) -> None:
    """Atomically write UTF-8 text."""

    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str | Path, value: Any) -> None:
    """Atomically write stable, human-readable JSON."""

    payload = json.dumps(
        value,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    atomic_write_text(path, payload)
