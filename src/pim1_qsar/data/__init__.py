"""Versioned data acquisition and curation for the publication benchmark."""

from .chembl_snapshot import SnapshotConfig, build_snapshot

__all__ = ["SnapshotConfig", "build_snapshot"]
