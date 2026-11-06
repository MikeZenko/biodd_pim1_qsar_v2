"""Leakage-resistant split and uncertainty utilities."""

from .bootstrap import ClusterBootstrapResult, cluster_bootstrap_metrics
from .locking import SplitLockSettings, lock_splits, select_outcome_blind_cutoff
from .splits import (
    FoldAssignment,
    TemporalSplit,
    fold_assignments_to_frame,
    make_repeated_scaffold_folds,
    make_temporal_split,
    validate_outer_assignment_frame,
    validate_temporal_partitions,
)

__all__ = [
    "ClusterBootstrapResult",
    "FoldAssignment",
    "SplitLockSettings",
    "TemporalSplit",
    "cluster_bootstrap_metrics",
    "fold_assignments_to_frame",
    "lock_splits",
    "make_repeated_scaffold_folds",
    "make_temporal_split",
    "select_outcome_blind_cutoff",
    "validate_outer_assignment_frame",
    "validate_temporal_partitions",
]
