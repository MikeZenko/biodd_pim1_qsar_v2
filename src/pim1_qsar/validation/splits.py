"""Temporal sealing and repeated scaffold-disjoint split manifests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from pim1_qsar.artifacts import sha256_json


SPLIT_SCHEMA_VERSION = "2.0"


def _json_scalar(value: Any) -> str | int | float | bool | None:
    """Convert common numpy scalars to canonical JSON-compatible values."""

    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def _validate_record_values(
    records: pd.DataFrame,
    *,
    identity_col: str,
    year_col: str,
    value_col: str,
) -> pd.DataFrame:
    normalized = records.copy()
    if normalized.empty:
        raise ValueError("records cannot be empty")
    if normalized[identity_col].isna().any():
        raise ValueError(f"{identity_col} contains missing identities")
    identities = normalized[identity_col].astype(str).str.strip()
    if identities.eq("").any():
        raise ValueError(f"{identity_col} contains empty identities")
    normalized[identity_col] = identities

    years = pd.to_numeric(normalized[year_col], errors="coerce")
    if years.isna().any() or not np.isfinite(years.to_numpy(dtype=float)).all():
        raise ValueError(f"{year_col} must contain finite years")
    if not np.equal(years, np.floor(years)).all():
        raise ValueError(f"{year_col} must contain integer years")
    normalized[year_col] = years.astype(int)

    values = pd.to_numeric(normalized[value_col], errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"{value_col} must contain finite numeric values")
    normalized[value_col] = values.astype(float)
    return normalized


def _aggregate_partition(
    records: pd.DataFrame,
    *,
    identity_col: str,
    year_col: str,
    value_col: str,
    scaffold_col: str | None,
    stable_cols: Sequence[str],
    event_id_col: str | None,
) -> pd.DataFrame:
    """Aggregate eligible record-level labels without crossing time boundaries."""

    if records.empty:
        columns = [
            identity_col,
            value_col,
            "first_record_year",
            "last_record_year",
            "measurement_count",
            "measurement_std",
        ]
        if scaffold_col:
            columns.insert(1, scaffold_col)
        for column in reversed(stable_cols):
            columns.insert(1, column)
        if event_id_col:
            columns.append("event_ids")
        return pd.DataFrame(columns=columns)

    invariant_cols = ([scaffold_col] if scaffold_col else []) + list(stable_cols)
    for invariant_col in invariant_cols:
        invariant_counts = records.groupby(identity_col, sort=True)[invariant_col].nunique(
            dropna=False
        )
        inconsistent = invariant_counts[invariant_counts != 1]
        if not inconsistent.empty:
            raise ValueError(
                f"{len(inconsistent)} identities map to multiple {invariant_col} values"
            )

    grouped = records.groupby(identity_col, sort=True, dropna=False)
    output = grouped.agg(
        **{
            value_col: (value_col, "median"),
            "first_record_year": (year_col, "min"),
            "last_record_year": (year_col, "max"),
            "measurement_count": (value_col, "size"),
            "measurement_std": (value_col, lambda values: float(np.std(values, ddof=0))),
        }
    ).reset_index()
    for invariant_col in invariant_cols:
        invariant_values = (
            grouped[invariant_col].first().rename(invariant_col).reset_index()
        )
        output = output.merge(
            invariant_values, on=identity_col, how="left", validate="one_to_one"
        )
    if event_id_col:
        event_ids = grouped[event_id_col].agg(
            lambda values: "|".join(sorted(set(values.astype(str))))
        ).rename("event_ids").reset_index()
        output = output.merge(event_ids, on=identity_col, how="left", validate="one_to_one")
    ordered_invariants = list(stable_cols) + ([scaffold_col] if scaffold_col else [])
    ordered = [identity_col] + ordered_invariants + [
        column
        for column in output.columns
        if column not in {identity_col, *ordered_invariants}
    ]
    output = output[ordered]
    return output.sort_values(identity_col, kind="mergesort").reset_index(drop=True)


@dataclass(frozen=True)
class TemporalSplit:
    """Record-derived development and sealed temporal-test partitions."""

    development: pd.DataFrame
    temporal_test: pd.DataFrame
    cross_cutoff_records: pd.DataFrame
    manifest: dict[str, Any]


def make_temporal_split(
    records: pd.DataFrame,
    *,
    cutoff_year: int,
    identity_col: str = "standardized_parent_inchi_key",
    year_col: str = "document_year",
    value_col: str = "pIC50",
    scaffold_col: str | None = "scaffold",
    stable_cols: Sequence[str] = (),
    event_id_col: str | None = None,
) -> TemporalSplit:
    """Build a publication-year split without contaminating development labels.

    An identity is assigned by its first eligible record year. Development
    labels use only records on or before *cutoff_year*. Identities first seen
    later form the temporal test. Later records for development identities are
    returned separately for label-drift analysis and never update training
    labels.
    """

    if isinstance(cutoff_year, bool) or not isinstance(cutoff_year, int):
        raise TypeError("cutoff_year must be an integer")
    required = [identity_col, year_col, value_col]
    if scaffold_col:
        required.append(scaffold_col)
    required.extend(stable_cols)
    if event_id_col:
        required.append(event_id_col)
    _require_columns(records, required)
    normalized = _validate_record_values(
        records,
        identity_col=identity_col,
        year_col=year_col,
        value_col=value_col,
    )

    first_year = normalized.groupby(identity_col, sort=True)[year_col].transform("min")
    development_mask = first_year.le(cutoff_year) & normalized[year_col].le(cutoff_year)
    temporal_test_mask = first_year.gt(cutoff_year)
    cross_cutoff_mask = first_year.le(cutoff_year) & normalized[year_col].gt(cutoff_year)

    development_records = normalized.loc[development_mask].copy()
    temporal_test_records = normalized.loc[temporal_test_mask].copy()
    cross_cutoff_records = normalized.loc[cross_cutoff_mask].copy()

    development = _aggregate_partition(
        development_records,
        identity_col=identity_col,
        year_col=year_col,
        value_col=value_col,
        scaffold_col=scaffold_col,
        stable_cols=stable_cols,
        event_id_col=event_id_col,
    )
    temporal_test = _aggregate_partition(
        temporal_test_records,
        identity_col=identity_col,
        year_col=year_col,
        value_col=value_col,
        scaffold_col=scaffold_col,
        stable_cols=stable_cols,
        event_id_col=event_id_col,
    )
    cross_cutoff_sort = [identity_col, year_col]
    if event_id_col:
        cross_cutoff_sort.append(event_id_col)
    cross_cutoff_records = cross_cutoff_records.sort_values(
        cross_cutoff_sort, kind="mergesort"
    ).reset_index(drop=True)

    development_ids = development[identity_col].astype(str).tolist()
    test_ids = temporal_test[identity_col].astype(str).tolist()
    overlap = set(development_ids) & set(test_ids)
    if overlap:
        raise RuntimeError(f"Temporal identity overlap detected: {sorted(overlap)[:3]}")
    if not development.empty and development["last_record_year"].gt(cutoff_year).any():
        raise RuntimeError("Development labels contain post-cutoff records")
    if not temporal_test.empty and temporal_test["first_record_year"].le(cutoff_year).any():
        raise RuntimeError("Temporal test contains pre-cutoff identities")

    manifest_core = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "split_type": "first_eligible_document_year",
        "cutoff_year": cutoff_year,
        "identity_column": identity_col,
        "year_column": year_col,
        "value_column": value_col,
        "scaffold_column": scaffold_col,
        "stable_columns": list(stable_cols),
        "event_id_column": event_id_col,
        "aggregation": "median",
        "development_compounds": len(development),
        "temporal_test_compounds": len(temporal_test),
        "cross_cutoff_record_count": len(cross_cutoff_records),
        "development_identity_sha256": sha256_json(development_ids),
        "temporal_test_identity_sha256": sha256_json(test_ids),
        "development_max_record_year": (
            int(development["last_record_year"].max()) if not development.empty else None
        ),
        "temporal_test_min_record_year": (
            int(temporal_test["first_record_year"].min()) if not temporal_test.empty else None
        ),
    }
    manifest = dict(manifest_core)
    manifest["split_sha256"] = sha256_json(manifest_core)
    return TemporalSplit(development, temporal_test, cross_cutoff_records, manifest)


def validate_temporal_partitions(
    development: pd.DataFrame,
    temporal_test: pd.DataFrame,
    cross_cutoff_records: pd.DataFrame,
    *,
    cutoff_year: int,
    identity_col: str,
    scaffold_col: str,
    year_col: str,
) -> None:
    """Validate the identity and time seals without requiring scaffold novelty.

    A temporal test may intentionally contain both known and new scaffolds. The
    scaffold-disjoint guarantee applies to each outer train/validation fold,
    while this guard enforces identity disjointness and time-directionality.
    """

    _require_columns(
        development,
        [identity_col, scaffold_col, "first_record_year", "last_record_year"],
    )
    _require_columns(
        temporal_test,
        [identity_col, scaffold_col, "first_record_year", "last_record_year"],
    )
    _require_columns(cross_cutoff_records, [identity_col, scaffold_col, year_col])
    development_ids = set(development[identity_col].astype(str))
    temporal_ids = set(temporal_test[identity_col].astype(str))
    if development_ids & temporal_ids:
        raise ValueError("Development and temporal-test identities overlap")
    if development[identity_col].duplicated().any() or temporal_test[identity_col].duplicated().any():
        raise ValueError("Compound-level temporal partitions contain duplicate identities")
    if development[scaffold_col].isna().any() or temporal_test[scaffold_col].isna().any():
        raise ValueError("Temporal partitions contain missing scaffold labels")
    if development["last_record_year"].gt(cutoff_year).any():
        raise ValueError("Development labels contain post-cutoff events")
    if temporal_test["first_record_year"].le(cutoff_year).any():
        raise ValueError("Temporal test contains an identity observed before the cutoff")
    cross_ids = set(cross_cutoff_records[identity_col].astype(str))
    if not cross_ids.issubset(development_ids):
        raise ValueError("Cross-cutoff records contain non-development identities")
    if not cross_cutoff_records.empty and cross_cutoff_records[year_col].le(cutoff_year).any():
        raise ValueError("Cross-cutoff table contains pre-cutoff events")


@dataclass(frozen=True)
class FoldAssignment:
    """One outer or inner scaffold-disjoint fold assignment."""

    repeat: int
    fold: int
    seed: int
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    train_groups: tuple[str, ...]
    validation_groups: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "repeat": self.repeat,
            "fold": self.fold,
            "seed": self.seed,
            "train_indices": list(self.train_indices),
            "validation_indices": list(self.validation_indices),
            "train_ids": list(self.train_ids),
            "validation_ids": list(self.validation_ids),
            "train_groups": list(self.train_groups),
            "validation_groups": list(self.validation_groups),
        }


def _regression_activity_bins(y: np.ndarray, n_bins: int) -> np.ndarray:
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    unique_count = len(np.unique(y))
    bins = min(n_bins, unique_count, len(y))
    if bins < 2:
        return np.zeros(len(y), dtype=int)
    # Ranking gives deterministic, near-equal bins even when activity values tie.
    ranks = pd.Series(y).rank(method="first")
    return pd.qcut(ranks, q=bins, labels=False, duplicates="drop").to_numpy(dtype=int)


def validate_repeated_scaffold_folds(
    folds: Sequence[FoldAssignment],
    *,
    n_samples: int,
    n_repeats: int,
    n_splits: int,
) -> None:
    """Raise if folds violate coverage, identity, or scaffold separation."""

    expected_indices = set(range(n_samples))
    if len(folds) != n_repeats * n_splits:
        raise ValueError("Unexpected number of fold assignments")
    for repeat in range(n_repeats):
        repeat_folds = [assignment for assignment in folds if assignment.repeat == repeat]
        if len(repeat_folds) != n_splits:
            raise ValueError(f"Repeat {repeat} does not have {n_splits} folds")
        validation_counts = np.zeros(n_samples, dtype=int)
        for assignment in repeat_folds:
            train_indices = set(assignment.train_indices)
            validation_indices = set(assignment.validation_indices)
            if train_indices & validation_indices:
                raise ValueError("Train/validation index overlap detected")
            if train_indices | validation_indices != expected_indices:
                raise ValueError("A fold does not cover all samples")
            if set(assignment.train_ids) & set(assignment.validation_ids):
                raise ValueError("Train/validation identity overlap detected")
            if set(assignment.train_groups) & set(assignment.validation_groups):
                raise ValueError("Train/validation scaffold overlap detected")
            validation_counts[np.asarray(assignment.validation_indices, dtype=int)] += 1
        if not np.equal(validation_counts, 1).all():
            raise ValueError(f"Every sample must validate exactly once in repeat {repeat}")


def make_repeated_scaffold_folds(
    sample_ids: Sequence[Any],
    scaffold_groups: Sequence[Any],
    y: Sequence[float],
    *,
    n_splits: int = 5,
    n_repeats: int = 5,
    base_seed: int = 1101,
    seeds: Sequence[int] | None = None,
    n_bins: int = 5,
) -> list[FoldAssignment]:
    """Create deterministic repeated, activity-balanced scaffold folds."""

    ids = np.asarray([str(value) for value in sample_ids], dtype=object)
    groups = np.asarray([str(value) for value in scaffold_groups], dtype=object)
    targets = np.asarray(y, dtype=float)
    if not (len(ids) == len(groups) == len(targets)):
        raise ValueError("sample_ids, scaffold_groups, and y must have equal length")
    if len(ids) == 0:
        raise ValueError("split inputs cannot be empty")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("sample_ids must be unique")
    if not np.isfinite(targets).all():
        raise ValueError("y must contain only finite values")
    if n_splits < 2 or n_repeats < 1:
        raise ValueError("n_splits must be >=2 and n_repeats must be >=1")
    if len(np.unique(groups)) < n_splits:
        raise ValueError("The number of scaffold groups must be at least n_splits")

    if seeds is None:
        repeat_seeds = [base_seed + repeat for repeat in range(n_repeats)]
    else:
        repeat_seeds = list(seeds)
        if len(repeat_seeds) != n_repeats:
            raise ValueError("seeds must contain exactly n_repeats values")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in repeat_seeds):
        raise TypeError("all seeds must be integers")

    activity_bins = _regression_activity_bins(targets, n_bins=n_bins)
    assignments: list[FoldAssignment] = []
    dummy_features = np.zeros((len(ids), 1), dtype=np.uint8)
    for repeat, seed in enumerate(repeat_seeds):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )
        for fold, (train_indices, validation_indices) in enumerate(
            splitter.split(dummy_features, activity_bins, groups=groups)
        ):
            train_indices = np.sort(train_indices)
            validation_indices = np.sort(validation_indices)
            assignments.append(
                FoldAssignment(
                    repeat=repeat,
                    fold=fold,
                    seed=seed,
                    train_indices=tuple(int(index) for index in train_indices),
                    validation_indices=tuple(int(index) for index in validation_indices),
                    train_ids=tuple(str(ids[index]) for index in train_indices),
                    validation_ids=tuple(str(ids[index]) for index in validation_indices),
                    train_groups=tuple(sorted(set(str(groups[index]) for index in train_indices))),
                    validation_groups=tuple(
                        sorted(set(str(groups[index]) for index in validation_indices))
                    ),
                )
            )
    validate_repeated_scaffold_folds(
        assignments,
        n_samples=len(ids),
        n_repeats=n_repeats,
        n_splits=n_splits,
    )
    return assignments


def fold_assignments_to_frame(
    folds: Sequence[FoldAssignment],
    compounds: pd.DataFrame,
    *,
    identity_col: str,
    scaffold_col: str,
) -> pd.DataFrame:
    """Materialize shared outer folds as a model-agnostic long-form table."""

    _require_columns(compounds, [identity_col, scaffold_col])
    if compounds[identity_col].duplicated().any():
        raise ValueError("compounds must contain one row per identity")
    identities = compounds[identity_col].astype(str).to_numpy()
    scaffolds = compounds[scaffold_col].astype(str).to_numpy()
    rows: list[dict[str, Any]] = []
    for assignment in folds:
        for role, indices in (
            ("train", assignment.train_indices),
            ("validation", assignment.validation_indices),
        ):
            for row_index in indices:
                rows.append(
                    {
                        "repeat": assignment.repeat,
                        "fold": assignment.fold,
                        "seed": assignment.seed,
                        "role": role,
                        "row_index": int(row_index),
                        identity_col: identities[row_index],
                        scaffold_col: scaffolds[row_index],
                    }
                )
    output = pd.DataFrame(
        rows,
        columns=[
            "repeat",
            "fold",
            "seed",
            "role",
            "row_index",
            identity_col,
            scaffold_col,
        ],
    )
    role_order = pd.Categorical(output["role"], categories=["train", "validation"], ordered=True)
    output = output.assign(_role_order=role_order).sort_values(
        ["repeat", "fold", "_role_order", "row_index"], kind="mergesort"
    ).drop(columns="_role_order").reset_index(drop=True)
    return output


def validate_outer_assignment_frame(
    assignments: pd.DataFrame,
    compounds: pd.DataFrame,
    *,
    identity_col: str,
    scaffold_col: str,
    n_repeats: int,
    n_splits: int,
    seeds: Sequence[int],
) -> None:
    """Guard long-form outer folds against identity/scaffold leakage."""

    required = [
        "repeat",
        "fold",
        "seed",
        "role",
        "row_index",
        identity_col,
        scaffold_col,
    ]
    _require_columns(assignments, required)
    _require_columns(compounds, [identity_col, scaffold_col])
    expected_ids = set(compounds[identity_col].astype(str))
    if len(expected_ids) != len(compounds):
        raise ValueError("compounds must contain unique identities")
    if len(seeds) != n_repeats:
        raise ValueError("seeds must contain exactly n_repeats values")
    if set(assignments["role"]) != {"train", "validation"}:
        raise ValueError("Outer assignments must contain train and validation roles")
    expected_rows = len(compounds) * n_repeats * n_splits
    if len(assignments) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} long-form assignment rows, found {len(assignments)}"
        )
    for repeat in range(n_repeats):
        repeat_rows = assignments.loc[assignments["repeat"].eq(repeat)]
        if set(repeat_rows["seed"].astype(int)) != {int(seeds[repeat])}:
            raise ValueError(f"Repeat {repeat} does not use its locked seed")
        validation_counts = (
            repeat_rows.loc[repeat_rows["role"].eq("validation"), identity_col]
            .astype(str)
            .value_counts()
        )
        if set(validation_counts.index) != expected_ids or not validation_counts.eq(1).all():
            raise ValueError(f"Every identity must validate once in repeat {repeat}")
        for fold in range(n_splits):
            fold_rows = repeat_rows.loc[repeat_rows["fold"].eq(fold)]
            train = fold_rows.loc[fold_rows["role"].eq("train")]
            validation = fold_rows.loc[fold_rows["role"].eq("validation")]
            train_ids = set(train[identity_col].astype(str))
            validation_ids = set(validation[identity_col].astype(str))
            if train_ids & validation_ids:
                raise ValueError("Outer train/validation identities overlap")
            if train_ids | validation_ids != expected_ids:
                raise ValueError("An outer fold does not cover every development identity")
            train_scaffolds = set(train[scaffold_col].astype(str))
            validation_scaffolds = set(validation[scaffold_col].astype(str))
            if train_scaffolds & validation_scaffolds:
                raise ValueError("Outer train/validation scaffolds overlap")


def repeated_folds_manifest(
    folds: Sequence[FoldAssignment],
    *,
    n_samples: int,
    n_repeats: int,
    n_splits: int,
    n_bins: int,
) -> dict[str, Any]:
    """Create a content-addressed manifest for persisted fold assignments."""

    core = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "splitter": "StratifiedGroupKFold",
        "group": "Bemis-Murcko scaffold",
        "n_samples": int(n_samples),
        "n_repeats": int(n_repeats),
        "n_splits": int(n_splits),
        "n_bins": int(n_bins),
        "folds": [assignment.as_dict() for assignment in folds],
    }
    manifest = dict(core)
    manifest["split_sha256"] = sha256_json(core)
    return manifest
