"""Publication-grade curation of the frozen PIM1 ChEMBL snapshot.

This module is deliberately separate from the legacy notebook-style pipeline.
It converts the release-verified JSONL snapshot into record-, measurement-, and
compound-level tables with complete exclusion and source-event lineage.  No
model split is chosen here: the temporal table only reports feasibility for
every possible document-year cutoff.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
import gzip
import importlib.metadata
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from pim1_qsar.artifacts import atomic_write_json, atomic_write_text, sha256_file, sha256_json


CURATION_SCHEMA_VERSION = "2.0"
REQUIRED_STRUCTURE_PIPELINE_VERSION = "1.2.4"
SINGLE_PROTEIN_BAO_FORMAT = "BAO_0000357"
HIGH_TAIL_ASSAY_ID = "CHEMBL3705754"

EVENT_COLUMNS = [
    "event_id",
    "event_signature_sha256",
    "standardized_parent_inchi_key",
    "standardized_parent_smiles",
    "scaffold_smiles",
    "target_chembl_id",
    "standard_type",
    "standard_relation",
    "standard_units",
    "standard_value_nm",
    "pIC50",
    "record_id",
    "document_chembl_id",
    "document_year",
    "representative_assay_chembl_id",
    "assay_chembl_ids",
    "assay_type",
    "assay_description",
    "bao_endpoint",
    "bao_format",
    "assay_confidence_score",
    "assay_relationship_type",
    "source_molecule_chembl_ids",
    "source_parent_chembl_ids",
    "activity_ids",
    "lineage_record_count",
]


@dataclass(frozen=True)
class CurationConfig:
    """Prespecified cohort rules; none of these are fitted to model results."""

    target_chembl_id: str = "CHEMBL2147"
    target_organism: str = "Homo sapiens"
    target_tax_id: int = 9606
    standard_type: str = "IC50"
    standard_relation: str = "="
    standard_units: str = "nM"
    standard_flag: int = 1
    strict_assay_type: str = "B"
    strict_bao_format: str = SINGLE_PROTEIN_BAO_FORMAT
    strict_relationship_type: str = "D"
    strict_confidence_score: int = 9
    high_potency_threshold: float = 10.0
    high_tail_assay_id: str = HIGH_TAIL_ASSAY_ID


@dataclass(frozen=True)
class StandardizedStructure:
    """The structure identity used for event and compound aggregation."""

    canonical_parent_smiles: str
    parent_inchi_key: str
    scaffold_smiles: str


@dataclass(frozen=True)
class CurationResult:
    record_qc: pd.DataFrame
    broad_events: pd.DataFrame
    strict_events: pd.DataFrame
    broad_compounds: pd.DataFrame
    strict_compounds: pd.DataFrame
    strict_no_high_tail_compounds: pd.DataFrame
    strict_no_high_tail_events: pd.DataFrame
    strict_no_tail_assay_compounds: pd.DataFrame
    strict_no_tail_assay_events: pd.DataFrame
    strict_measurement_dispersion: pd.DataFrame
    lineage: pd.DataFrame
    exclusions: pd.DataFrame
    temporal_feasibility: pd.DataFrame
    manifest: dict[str, Any]


StructureStandardizer = Callable[[str], StandardizedStructure]


def _blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _normalized_description(value: Any) -> str:
    return " ".join(_text(value).split()).casefold()


def _flag_is_one(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _text(value).casefold() in {"1", "true"}


def _flag_is_zero(value: Any) -> bool:
    if isinstance(value, bool):
        return not value
    return _text(value).casefold() in {"0", "false"}


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _integer_year(value: Any) -> int | None:
    parsed = _finite_float(value)
    if parsed is None or parsed != math.floor(parsed):
        return None
    year = int(parsed)
    return year if 1900 <= year <= 2200 else None


def _canonical_number(value: Any) -> str:
    try:
        decimal = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return _text(value)
    if not decimal.is_finite():
        return _text(value)
    normalized = format(decimal.normalize(), "f")
    return "0" if normalized in {"-0", "-0.0"} else normalized


def _sorted_tokens(values: Iterable[Any]) -> str:
    tokens = {_text(value) for value in values if not _blank(value)}

    def key(token: str) -> tuple[int, Any]:
        try:
            return (0, int(token))
        except ValueError:
            return (1, token)

    return "|".join(sorted(tokens, key=key))


def _unique_map(
    records: Sequence[Mapping[str, Any]], key: str, label: str
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for record in records:
        identifier = _text(record.get(key))
        if not identifier:
            raise ValueError(f"{label} record omitted {key}")
        materialized = dict(record)
        if identifier in output and output[identifier] != materialized:
            raise ValueError(f"Conflicting duplicate {label} identifier: {identifier}")
        output[identifier] = materialized
    return output


def load_chembl_structure_standardizer(
    required_version: str = REQUIRED_STRUCTURE_PIPELINE_VERSION,
) -> tuple[StructureStandardizer, str]:
    """Load the exact ChEMBL Structure Pipeline required by the protocol.

    Importing is deferred so unit tests and read-only manifest tools do not
    silently fall back to a different standardizer.  Production curation fails
    with an actionable message if the pinned package is unavailable or wrong.
    """

    try:
        installed = importlib.metadata.version("chembl_structure_pipeline")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "Curation requires chembl-structure-pipeline==1.2.4. "
            "Install that exact version in the VM environment; no fallback is permitted."
        ) from exc
    if installed != required_version:
        raise RuntimeError(
            "ChEMBL Structure Pipeline version mismatch: "
            f"required {required_version}, found {installed}."
        )

    try:
        from chembl_structure_pipeline import standardizer as chembl_standardizer
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError as exc:  # pragma: no cover - metadata normally implies importability
        raise RuntimeError(
            "chembl-structure-pipeline==1.2.4 is installed but could not be imported"
        ) from exc

    def standardize(smiles: str) -> StandardizedStructure:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError("RDKit could not parse the parent SMILES")
        standardized = chembl_standardizer.standardize_mol(molecule)
        parent_result = chembl_standardizer.get_parent_mol(standardized)
        if isinstance(parent_result, tuple):
            parent, exclusion_flag = parent_result
        else:  # retained for API clarity if a patch release changes return wrapping
            parent, exclusion_flag = parent_result, False
        if parent is None:
            raise ValueError("ChEMBL Structure Pipeline returned no parent molecule")
        if exclusion_flag:
            raise ValueError("ChEMBL Structure Pipeline marked the molecule for exclusion")

        # chembl-structure-pipeline 1.2.4 can return a chemically valid Mol whose
        # derived RDKit ring cache is not populated when used with newer RDKit
        # releases.  MurckoScaffold then raises "RingInfo not initialized" even
        # though the structure itself is valid.  Sanitize and explicitly build
        # the ring information before serializing; reparsing the canonical
        # parent also prevents pipeline-owned transient properties from becoming
        # part of the identity used downstream.
        Chem.SanitizeMol(parent)
        Chem.GetSymmSSSR(parent)
        parent_smiles = Chem.MolToSmiles(
            parent, canonical=True, isomericSmiles=True
        )
        if not parent_smiles:
            raise ValueError("Unable to generate standardized parent SMILES")
        parent = Chem.MolFromSmiles(parent_smiles, sanitize=True)
        if parent is None:
            raise ValueError("Unable to reparse standardized parent SMILES")
        Chem.GetSymmSSSR(parent)
        parent_smiles = Chem.MolToSmiles(
            parent, canonical=True, isomericSmiles=True
        )
        inchi_key = Chem.MolToInchiKey(parent)
        if not inchi_key:
            raise ValueError("Unable to generate standardized SMILES/InChIKey")
        scaffold_mol = MurckoScaffold.GetScaffoldForMol(parent)
        scaffold = Chem.MolToSmiles(
            scaffold_mol, canonical=True, isomericSmiles=False
        )
        if not scaffold:
            scaffold = "__ACYCLIC__"
        return StandardizedStructure(parent_smiles, inchi_key, scaffold)

    return standardize, installed


def _assay_value(assay: Mapping[str, Any], activity: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in assay and not _blank(assay.get(name)):
            return assay.get(name)
        if name in activity and not _blank(activity.get(name)):
            return activity.get(name)
    return None


def _standardize_parent_structures(
    parent_ids: Sequence[str],
    parent_map: Mapping[str, Mapping[str, Any]],
    standardizer: StructureStandardizer,
) -> tuple[dict[str, StandardizedStructure], dict[str, str]]:
    structures: dict[str, StandardizedStructure] = {}
    errors: dict[str, str] = {}
    for parent_id in sorted(set(parent_ids)):
        parent = parent_map.get(parent_id)
        if parent is None:
            errors[parent_id] = "missing_parent_molecule_metadata"
            continue
        smiles = _text(parent.get("canonical_smiles"))
        if not smiles:
            errors[parent_id] = "missing_parent_smiles"
            continue
        try:
            result = standardizer(smiles)
            if not result.canonical_parent_smiles or not result.parent_inchi_key:
                raise ValueError("standardizer returned an incomplete identity")
            structures[parent_id] = result
        except Exception as exc:
            errors[parent_id] = f"structure_standardization_failed:{type(exc).__name__}:{exc}"
    return structures, errors


def _event_signature(record: Mapping[str, Any]) -> dict[str, Any]:
    record_id = _text(record.get("record_id"))
    if not record_id:
        record_id = f"activity:{_text(record.get('activity_id'))}"
    return {
        "standardized_parent_inchi_key": record["standardized_parent_inchi_key"],
        "target_chembl_id": record["target_chembl_id"],
        "record_id": record_id,
        "document_chembl_id": _text(record.get("document_chembl_id")),
        "assay_description": _normalized_description(record.get("assay_description")),
        "bao_endpoint": _text(record.get("bao_endpoint")),
        "bao_format": _text(record.get("bao_format")),
        "assay_type": _text(record.get("assay_type")),
        "standard_type": _text(record.get("standard_type")),
        "standard_relation": _text(record.get("standard_relation")),
        "standard_value": _canonical_number(record.get("standard_value_nm")),
        "standard_units": _text(record.get("standard_units")),
    }


def _activity_sort_key(record: Mapping[str, Any]) -> tuple[int, Any]:
    identifier = _text(record.get("activity_id"))
    try:
        return (0, int(identifier))
    except ValueError:
        return (1, identifier)


def _collapse_events(
    records: Sequence[Mapping[str, Any]], cohort: str
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    signatures: dict[str, dict[str, Any]] = {}
    for record in records:
        signature = _event_signature(record)
        digest = sha256_json(signature)
        grouped[digest].append(dict(record))
        signatures[digest] = signature

    events: list[dict[str, Any]] = []
    lineage: list[dict[str, Any]] = []
    duplicate_exclusions: list[dict[str, Any]] = []
    for digest in sorted(grouped):
        members = sorted(grouped[digest], key=_activity_sort_key)
        representative = members[0]
        event_id = f"event-{digest[:24]}"
        activity_ids = _sorted_tokens(member.get("activity_id") for member in members)
        assay_ids = _sorted_tokens(member.get("assay_chembl_id") for member in members)
        source_molecule_ids = _sorted_tokens(
            member.get("molecule_chembl_id") for member in members
        )
        parent_ids = _sorted_tokens(
            member.get("source_parent_chembl_id") for member in members
        )
        events.append(
            {
                "event_id": event_id,
                "event_signature_sha256": digest,
                "standardized_parent_inchi_key": representative[
                    "standardized_parent_inchi_key"
                ],
                "standardized_parent_smiles": representative[
                    "standardized_parent_smiles"
                ],
                "scaffold_smiles": representative["scaffold_smiles"],
                "target_chembl_id": representative["target_chembl_id"],
                "standard_type": representative["standard_type"],
                "standard_relation": representative["standard_relation"],
                "standard_units": representative["standard_units"],
                "standard_value_nm": float(representative["standard_value_nm"]),
                "pIC50": float(representative["pIC50"]),
                "record_id": signatures[digest]["record_id"],
                "document_chembl_id": representative["document_chembl_id"],
                "document_year": int(representative["document_year"]),
                "representative_assay_chembl_id": representative["assay_chembl_id"],
                "assay_chembl_ids": assay_ids,
                "assay_type": representative["assay_type"],
                "assay_description": representative["assay_description"],
                "bao_endpoint": representative["bao_endpoint"],
                "bao_format": representative["bao_format"],
                "assay_confidence_score": representative["assay_confidence_score"],
                "assay_relationship_type": representative["assay_relationship_type"],
                "source_molecule_chembl_ids": source_molecule_ids,
                "source_parent_chembl_ids": parent_ids,
                "activity_ids": activity_ids,
                "lineage_record_count": len(members),
            }
        )
        for index, member in enumerate(members):
            lineage.append(
                {
                    "cohort": cohort,
                    "event_id": event_id,
                    "event_signature_sha256": digest,
                    "activity_id": _text(member.get("activity_id")),
                    "record_id": _text(member.get("record_id")),
                    "assay_chembl_id": _text(member.get("assay_chembl_id")),
                    "document_chembl_id": _text(member.get("document_chembl_id")),
                    "molecule_chembl_id": _text(member.get("molecule_chembl_id")),
                    "source_parent_chembl_id": _text(
                        member.get("source_parent_chembl_id")
                    ),
                    "standardized_parent_inchi_key": member[
                        "standardized_parent_inchi_key"
                    ],
                    "is_event_representative": index == 0,
                }
            )
            if index:
                duplicate_exclusions.append(
                    {
                        "cohort": cohort,
                        "activity_id": _text(member.get("activity_id")),
                        "reason": "duplicate_source_event_collapsed",
                        "detail": event_id,
                    }
                )
    # Preserve the public table contract even when every record is excluded.
    # This lets curation finish with an auditable zero-row cohort and prevents
    # opaque KeyErrors in prespecified sensitivity filters.
    event_frame = pd.DataFrame(events, columns=EVENT_COLUMNS)
    if not event_frame.empty:
        event_frame = event_frame.sort_values(
            ["standardized_parent_inchi_key", "document_year", "event_id"],
            kind="mergesort",
        ).reset_index(drop=True)
        if event_frame["event_signature_sha256"].duplicated().any():
            raise RuntimeError("Duplicate event signatures remained after collapse")
    return event_frame, lineage, duplicate_exclusions


def _pipe_union(values: Iterable[Any]) -> str:
    tokens: list[str] = []
    for value in values:
        tokens.extend(token for token in _text(value).split("|") if token)
    return _sorted_tokens(tokens)


def _aggregate_compounds(events: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "standardized_parent_inchi_key",
        "standardized_parent_smiles",
        "scaffold_smiles",
        "pIC50",
        "pIC50_mean",
        "pIC50_std",
        "pIC50_q1",
        "pIC50_q3",
        "pIC50_iqr",
        "pIC50_mad",
        "pIC50_min",
        "pIC50_max",
        "pIC50_range",
        "measurement_event_count",
        "source_record_count",
        "unique_assay_count",
        "unique_document_count",
        "first_document_year",
        "last_document_year",
        "event_ids",
        "activity_ids",
        "assay_chembl_ids",
        "document_chembl_ids",
        "source_molecule_chembl_ids",
        "source_parent_chembl_ids",
    ]
    if events.empty:
        return pd.DataFrame(columns=columns)

    output: list[dict[str, Any]] = []
    for identity, group in events.groupby(
        "standardized_parent_inchi_key", sort=True, dropna=False
    ):
        smiles = set(group["standardized_parent_smiles"].astype(str))
        scaffolds = set(group["scaffold_smiles"].astype(str))
        if len(smiles) != 1 or len(scaffolds) != 1:
            raise RuntimeError(f"Inconsistent standardized structure for {identity}")
        values = group["pIC50"].to_numpy(dtype=float)
        median = float(np.median(values))
        q1, q3 = np.quantile(values, [0.25, 0.75], method="linear")
        documents = _sorted_tokens(group["document_chembl_id"])
        assay_ids = _pipe_union(group["assay_chembl_ids"])
        output.append(
            {
                "standardized_parent_inchi_key": str(identity),
                "standardized_parent_smiles": next(iter(smiles)),
                "scaffold_smiles": next(iter(scaffolds)),
                "pIC50": median,
                "pIC50_mean": float(np.mean(values)),
                "pIC50_std": float(np.std(values, ddof=0)),
                "pIC50_q1": float(q1),
                "pIC50_q3": float(q3),
                "pIC50_iqr": float(q3 - q1),
                "pIC50_mad": float(np.median(np.abs(values - median))),
                "pIC50_min": float(np.min(values)),
                "pIC50_max": float(np.max(values)),
                "pIC50_range": float(np.max(values) - np.min(values)),
                "measurement_event_count": int(len(group)),
                "source_record_count": int(group["lineage_record_count"].sum()),
                "unique_assay_count": len([x for x in assay_ids.split("|") if x]),
                "unique_document_count": len([x for x in documents.split("|") if x]),
                "first_document_year": int(group["document_year"].min()),
                "last_document_year": int(group["document_year"].max()),
                "event_ids": _sorted_tokens(group["event_id"]),
                "activity_ids": _pipe_union(group["activity_ids"]),
                "assay_chembl_ids": assay_ids,
                "document_chembl_ids": documents,
                "source_molecule_chembl_ids": _pipe_union(
                    group["source_molecule_chembl_ids"]
                ),
                "source_parent_chembl_ids": _pipe_union(
                    group["source_parent_chembl_ids"]
                ),
            }
        )
    return pd.DataFrame(output, columns=columns).sort_values(
        "standardized_parent_inchi_key", kind="mergesort"
    ).reset_index(drop=True)


def _temporal_feasibility(events: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "cutoff_year",
        "development_compounds",
        "temporal_test_compounds",
        "development_scaffolds",
        "temporal_test_scaffolds",
        "new_temporal_test_scaffolds",
        "temporal_test_compounds_on_new_scaffolds",
        "cross_cutoff_event_count",
        "earliest_temporal_test_year",
        "latest_temporal_test_year",
        "selection_status",
    ]
    if events.empty:
        return pd.DataFrame(columns=columns)
    first_year = events.groupby("standardized_parent_inchi_key")["document_year"].min()
    scaffold_by_identity = events.groupby("standardized_parent_inchi_key")[
        "scaffold_smiles"
    ].agg(lambda values: sorted(set(values.astype(str))))
    inconsistent = scaffold_by_identity[scaffold_by_identity.map(len) != 1]
    if not inconsistent.empty:
        raise RuntimeError("One identity maps to multiple scaffolds")
    scaffold_by_identity = scaffold_by_identity.map(lambda values: values[0])
    minimum, maximum = int(events["document_year"].min()), int(events["document_year"].max())
    rows: list[dict[str, Any]] = []
    for cutoff in range(minimum, maximum):
        development_ids = set(first_year[first_year <= cutoff].index.astype(str))
        test_ids = set(first_year[first_year > cutoff].index.astype(str))
        development_scaffolds = set(scaffold_by_identity.loc[list(development_ids)])
        test_scaffolds = set(scaffold_by_identity.loc[list(test_ids)])
        new_scaffolds = test_scaffolds - development_scaffolds
        new_scaffold_compounds = sum(
            scaffold_by_identity.loc[identity] in new_scaffolds for identity in test_ids
        )
        identity_series = events["standardized_parent_inchi_key"].astype(str)
        cross_cutoff = identity_series.isin(development_ids) & events["document_year"].gt(
            cutoff
        )
        test_years = events.loc[identity_series.isin(test_ids), "document_year"]
        rows.append(
            {
                "cutoff_year": cutoff,
                "development_compounds": len(development_ids),
                "temporal_test_compounds": len(test_ids),
                "development_scaffolds": len(development_scaffolds),
                "temporal_test_scaffolds": len(test_scaffolds),
                "new_temporal_test_scaffolds": len(new_scaffolds),
                "temporal_test_compounds_on_new_scaffolds": int(new_scaffold_compounds),
                "cross_cutoff_event_count": int(cross_cutoff.sum()),
                "earliest_temporal_test_year": (
                    int(test_years.min()) if not test_years.empty else None
                ),
                "latest_temporal_test_year": (
                    int(test_years.max()) if not test_years.empty else None
                ),
                "selection_status": "unselected",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def curate_records(
    activities: Sequence[Mapping[str, Any]],
    assays: Sequence[Mapping[str, Any]],
    parent_molecules: Sequence[Mapping[str, Any]],
    *,
    config: CurationConfig = CurationConfig(),
    standardizer: StructureStandardizer | None = None,
    structure_pipeline_version: str | None = None,
) -> CurationResult:
    """Curate already-loaded snapshot tables without performing network I/O."""

    if standardizer is None:
        standardizer, loaded_version = load_chembl_structure_standardizer()
        structure_pipeline_version = loaded_version
    elif structure_pipeline_version is None:
        structure_pipeline_version = "injected-test-standardizer"

    activities = [dict(record) for record in activities]
    assays = [dict(record) for record in assays]
    parent_molecules = [dict(record) for record in parent_molecules]
    assay_map = _unique_map(assays, "assay_chembl_id", "assay")
    parent_map = _unique_map(parent_molecules, "molecule_chembl_id", "parent molecule")

    referenced_assays = {_text(record.get("assay_chembl_id")) for record in activities}
    missing_assays = sorted(referenced_assays - set(assay_map))
    if missing_assays:
        raise ValueError(f"Missing joined assay metadata: {missing_assays[:5]}")
    parent_ids = [
        _text(record.get("parent_molecule_chembl_id") or record.get("molecule_chembl_id"))
        for record in activities
    ]
    structures, structure_errors = _standardize_parent_structures(
        parent_ids, parent_map, standardizer
    )

    qc_rows: list[dict[str, Any]] = []
    broad_records: list[dict[str, Any]] = []
    strict_records: list[dict[str, Any]] = []
    exclusion_rows: list[dict[str, Any]] = []

    for activity in sorted(activities, key=_activity_sort_key):
        activity_id = _text(activity.get("activity_id"))
        assay_id = _text(activity.get("assay_chembl_id"))
        assay = assay_map[assay_id]
        parent_id = _text(
            activity.get("parent_molecule_chembl_id") or activity.get("molecule_chembl_id")
        )
        value = _finite_float(activity.get("standard_value"))
        year = _integer_year(activity.get("document_year"))
        broad_reasons: list[str] = []
        if _text(activity.get("target_chembl_id")) != config.target_chembl_id:
            broad_reasons.append("target_chembl_id_mismatch")
        if _text(activity.get("target_organism")) != config.target_organism:
            broad_reasons.append("target_organism_mismatch")
        if _text(activity.get("target_tax_id")) != str(config.target_tax_id):
            broad_reasons.append("target_tax_id_mismatch")
        if _text(activity.get("standard_type")) != config.standard_type:
            broad_reasons.append("standard_type_not_ic50")
        if _text(activity.get("standard_relation")) != config.standard_relation:
            broad_reasons.append("standard_relation_not_exact")
        relation = activity.get("relation")
        if not _blank(relation) and _text(relation) != config.standard_relation:
            broad_reasons.append("relation_not_exact")
        if _text(activity.get("standard_units")) != config.standard_units:
            broad_reasons.append("standard_units_not_nm")
        if not _flag_is_one(activity.get("standard_flag")):
            broad_reasons.append("standard_flag_not_one")
        if value is None or value <= 0:
            broad_reasons.append("standard_value_not_positive_finite")
        if not _flag_is_zero(activity.get("potential_duplicate")):
            broad_reasons.append("potential_duplicate")
        if not _blank(activity.get("data_validity_comment")) or not _blank(
            activity.get("data_validity_description")
        ):
            broad_reasons.append("data_validity_warning")
        if year is None:
            broad_reasons.append("document_year_missing_or_invalid")
        assay_target = _text(assay.get("target_chembl_id"))
        if assay_target != config.target_chembl_id:
            broad_reasons.append("assay_target_mismatch")
        if parent_id in structure_errors:
            broad_reasons.append(structure_errors[parent_id])

        assay_type = _text(_assay_value(assay, activity, "assay_type"))
        bao_format = _text(_assay_value(assay, activity, "bao_format"))
        bao_endpoint = _text(activity.get("bao_endpoint"))
        assay_description = _text(
            _assay_value(assay, activity, "description", "assay_description")
        )
        relationship = _text(assay.get("relationship_type"))
        confidence_float = _finite_float(assay.get("confidence_score"))
        confidence: int | None = None
        if confidence_float is not None and confidence_float == math.floor(confidence_float):
            confidence = int(confidence_float)

        strict_reasons: list[str] = []
        if assay_type != config.strict_assay_type:
            strict_reasons.append("assay_type_not_biochemical")
        if bao_format != config.strict_bao_format:
            strict_reasons.append("bao_format_not_single_protein")
        if relationship != config.strict_relationship_type:
            strict_reasons.append("assay_relationship_not_direct")
        if confidence != config.strict_confidence_score:
            strict_reasons.append("assay_confidence_not_strict")
        if not _blank(activity.get("assay_variant_accession")) or not _blank(
            activity.get("assay_variant_mutation")
        ):
            strict_reasons.append("assay_variant_not_wild_type")

        broad_eligible = not broad_reasons
        strict_eligible = broad_eligible and not strict_reasons
        structure = structures.get(parent_id)
        pic50 = 9.0 - math.log10(value) if value is not None and value > 0 else None
        qc_rows.append(
            {
                "activity_id": activity_id,
                "record_id": _text(activity.get("record_id")),
                "molecule_chembl_id": _text(activity.get("molecule_chembl_id")),
                "source_parent_chembl_id": parent_id,
                "assay_chembl_id": assay_id,
                "document_chembl_id": _text(activity.get("document_chembl_id")),
                "document_year": year,
                "standard_value_nm": value,
                "pIC50": pic50,
                "standardized_parent_inchi_key": (
                    structure.parent_inchi_key if structure else ""
                ),
                "broad_eligible": broad_eligible,
                "strict_eligible": strict_eligible,
                "broad_exclusion_reasons": "|".join(sorted(broad_reasons)),
                "strict_exclusion_reasons": "|".join(sorted(strict_reasons)),
                "assay_type": assay_type,
                "bao_format": bao_format,
                "assay_confidence_score": confidence,
                "assay_relationship_type": relationship,
            }
        )
        for reason in broad_reasons:
            exclusion_rows.append(
                {"cohort": "broad", "activity_id": activity_id, "reason": reason, "detail": ""}
            )
        if broad_eligible:
            for reason in strict_reasons:
                exclusion_rows.append(
                    {
                        "cohort": "strict",
                        "activity_id": activity_id,
                        "reason": reason,
                        "detail": "",
                    }
                )
            assert structure is not None and value is not None and year is not None
            enriched = {
                "activity_id": activity_id,
                "record_id": _text(activity.get("record_id")),
                "molecule_chembl_id": _text(activity.get("molecule_chembl_id")),
                "source_parent_chembl_id": parent_id,
                "standardized_parent_inchi_key": structure.parent_inchi_key,
                "standardized_parent_smiles": structure.canonical_parent_smiles,
                "scaffold_smiles": structure.scaffold_smiles,
                "target_chembl_id": config.target_chembl_id,
                "standard_type": config.standard_type,
                "standard_relation": config.standard_relation,
                "standard_units": config.standard_units,
                "standard_value_nm": value,
                "pIC50": 9.0 - math.log10(value),
                "assay_chembl_id": assay_id,
                "assay_type": assay_type,
                "assay_description": assay_description,
                "bao_endpoint": bao_endpoint,
                "bao_format": bao_format,
                "assay_confidence_score": confidence,
                "assay_relationship_type": relationship,
                "document_chembl_id": _text(activity.get("document_chembl_id")),
                "document_year": year,
            }
            broad_records.append(enriched)
            if strict_eligible:
                strict_records.append(enriched)

    broad_events, broad_lineage, broad_duplicate_exclusions = _collapse_events(
        broad_records, "broad"
    )
    strict_events, strict_lineage, strict_duplicate_exclusions = _collapse_events(
        strict_records, "strict"
    )
    exclusion_rows.extend(broad_duplicate_exclusions)
    exclusion_rows.extend(strict_duplicate_exclusions)

    strict_no_high_tail_events = strict_events.loc[
        strict_events["pIC50"].le(config.high_potency_threshold)
    ].copy().reset_index(drop=True)
    removed_tail = strict_events.loc[
        strict_events["pIC50"].gt(config.high_potency_threshold)
    ]
    for activity_ids, event_id in zip(removed_tail["activity_ids"], removed_tail["event_id"]):
        for activity_id in _text(activity_ids).split("|"):
            exclusion_rows.append(
                {
                    "cohort": "strict_no_high_tail",
                    "activity_id": activity_id,
                    "reason": "pIC50_above_prespecified_threshold",
                    "detail": event_id,
                }
            )

    tail_assay = config.high_tail_assay_id
    assay_contains_tail = strict_events["assay_chembl_ids"].map(
        lambda value: tail_assay in _text(value).split("|")
    )
    strict_no_tail_assay_events = strict_events.loc[~assay_contains_tail].copy().reset_index(
        drop=True
    )
    for activity_ids, event_id in zip(
        strict_events.loc[assay_contains_tail, "activity_ids"],
        strict_events.loc[assay_contains_tail, "event_id"],
    ):
        for activity_id in _text(activity_ids).split("|"):
            exclusion_rows.append(
                {
                    "cohort": "strict_no_CHEMBL3705754",
                    "activity_id": activity_id,
                    "reason": "tail_assay_excluded",
                    "detail": event_id,
                }
            )

    broad_compounds = _aggregate_compounds(broad_events)
    strict_compounds = _aggregate_compounds(strict_events)
    strict_no_high_tail_compounds = _aggregate_compounds(strict_no_high_tail_events)
    strict_no_tail_assay_compounds = _aggregate_compounds(strict_no_tail_assay_events)
    dispersion_columns = [
        "standardized_parent_inchi_key",
        "measurement_event_count",
        "source_record_count",
        "unique_assay_count",
        "unique_document_count",
        "pIC50_std",
        "pIC50_iqr",
        "pIC50_mad",
        "pIC50_min",
        "pIC50_max",
        "pIC50_range",
        "first_document_year",
        "last_document_year",
        "event_ids",
        "activity_ids",
        "assay_chembl_ids",
        "document_chembl_ids",
    ]
    strict_measurement_dispersion = strict_compounds[dispersion_columns].copy()
    temporal_feasibility = _temporal_feasibility(strict_events)

    record_qc = pd.DataFrame(qc_rows).sort_values("activity_id", kind="mergesort").reset_index(
        drop=True
    )
    lineage = pd.DataFrame(broad_lineage + strict_lineage)
    if not lineage.empty:
        lineage = lineage.sort_values(
            ["cohort", "event_id", "is_event_representative", "activity_id"],
            ascending=[True, True, False, True],
            kind="mergesort",
        ).reset_index(drop=True)
    exclusions = pd.DataFrame(
        exclusion_rows, columns=["cohort", "activity_id", "reason", "detail"]
    )
    if not exclusions.empty:
        exclusions = exclusions.sort_values(
            ["cohort", "activity_id", "reason"], kind="mergesort"
        ).reset_index(drop=True)

    reason_counts = Counter(
        f"{row['cohort']}:{row['reason']}" for row in exclusion_rows
    )
    manifest = {
        "schema_version": CURATION_SCHEMA_VERSION,
        "curation_protocol": "pim1_publication_curation_v2",
        "structure_pipeline": {
            "package": "chembl-structure-pipeline",
            "required_version": REQUIRED_STRUCTURE_PIPELINE_VERSION,
            "used_version": structure_pipeline_version,
            "identity": "ChEMBL parent followed by structure-pipeline parent InChIKey",
        },
        "config": asdict(config),
        "input_counts": {
            "activities": len(activities),
            "assays": len(assays),
            "parent_molecules": len(parent_molecules),
            "referenced_parent_molecules": len(set(parent_ids)),
            "structure_standardization_failures": len(structure_errors),
        },
        "cohort_counts": {
            "broad": {
                "eligible_records": len(broad_records),
                "measurement_events": len(broad_events),
                "duplicate_records_collapsed": len(broad_records) - len(broad_events),
                "compounds": len(broad_compounds),
            },
            "strict": {
                "eligible_records": len(strict_records),
                "measurement_events": len(strict_events),
                "duplicate_records_collapsed": len(strict_records) - len(strict_events),
                "compounds": len(strict_compounds),
            },
            "strict_no_high_tail": {
                "measurement_events": len(strict_no_high_tail_events),
                "events_removed": len(strict_events) - len(strict_no_high_tail_events),
                "compounds": len(strict_no_high_tail_compounds),
            },
            "strict_no_CHEMBL3705754": {
                "measurement_events": len(strict_no_tail_assay_events),
                "events_removed": len(strict_events) - len(strict_no_tail_assay_events),
                "compounds": len(strict_no_tail_assay_compounds),
            },
        },
        "measurement_dispersion": {
            "strict_compounds_with_multiple_events": int(
                strict_compounds["measurement_event_count"].gt(1).sum()
            ),
            "strict_compounds_range_gt_0_5": int(
                strict_compounds["pIC50_range"].gt(0.5).sum()
            ),
            "strict_compounds_range_gt_1": int(
                strict_compounds["pIC50_range"].gt(1.0).sum()
            ),
        },
        "temporal_feasibility": {
            "basis": "first eligible strict measurement-event document year",
            "candidate_cutoffs_reported": temporal_feasibility["cutoff_year"].tolist(),
            "selected_cutoff": None,
        },
        "exclusion_reason_counts": dict(sorted(reason_counts.items())),
    }
    return CurationResult(
        record_qc=record_qc,
        broad_events=broad_events,
        strict_events=strict_events,
        broad_compounds=broad_compounds,
        strict_compounds=strict_compounds,
        strict_no_high_tail_compounds=strict_no_high_tail_compounds,
        strict_no_high_tail_events=strict_no_high_tail_events,
        strict_no_tail_assay_compounds=strict_no_tail_assay_compounds,
        strict_no_tail_assay_events=strict_no_tail_assay_events,
        strict_measurement_dispersion=strict_measurement_dispersion,
        lineage=lineage,
        exclusions=exclusions,
        temporal_feasibility=temporal_feasibility,
        manifest=manifest,
    )


def _read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    text = frame.to_csv(index=False, lineterminator="\n", float_format="%.12g")
    atomic_write_text(path, text)


def _verify_snapshot(snapshot_dir: Path) -> tuple[dict[str, Any], Path]:
    manifest_path = snapshot_dir / "snapshot_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing snapshot manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for filename, expected in manifest.get("files_sha256", {}).items():
        path = snapshot_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Snapshot manifest references missing file: {path}")
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(f"Snapshot checksum mismatch for {filename}")
    return manifest, manifest_path


def curate_snapshot(
    snapshot_dir: Path,
    output_dir: Path,
    *,
    config: CurationConfig = CurationConfig(),
    standardizer: StructureStandardizer | None = None,
    structure_pipeline_version: str | None = None,
) -> dict[str, Any]:
    """Curate a verified snapshot and atomically write all auditable tables."""

    snapshot_dir = snapshot_dir.resolve()
    output_dir = output_dir.resolve()
    snapshot_manifest, snapshot_manifest_path = _verify_snapshot(snapshot_dir)
    result = curate_records(
        _read_jsonl_gz(snapshot_dir / "activities.jsonl.gz"),
        _read_jsonl_gz(snapshot_dir / "assays.jsonl.gz"),
        _read_jsonl_gz(snapshot_dir / "parent_molecules.jsonl.gz"),
        config=config,
        standardizer=standardizer,
        structure_pipeline_version=structure_pipeline_version,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "record_qc.csv": result.record_qc,
        "broad_measurement_events.csv": result.broad_events,
        "strict_measurement_events.csv": result.strict_events,
        "compound_broad.csv": result.broad_compounds,
        "compound_strict.csv": result.strict_compounds,
        "strict_no_high_tail_events.csv": result.strict_no_high_tail_events,
        "compound_strict_no_high_tail.csv": result.strict_no_high_tail_compounds,
        "strict_no_CHEMBL3705754_events.csv": result.strict_no_tail_assay_events,
        "compound_strict_no_CHEMBL3705754.csv": result.strict_no_tail_assay_compounds,
        "strict_measurement_dispersion.csv": result.strict_measurement_dispersion,
        "event_lineage.csv": result.lineage,
        "exclusion_log.csv": result.exclusions,
        "temporal_feasibility.csv": result.temporal_feasibility,
    }
    for filename, frame in tables.items():
        _write_csv(output_dir / filename, frame)

    manifest = dict(result.manifest)
    manifest["source_snapshot"] = {
        "snapshot_id": snapshot_manifest.get("snapshot_id"),
        "snapshot_manifest_sha256": sha256_file(snapshot_manifest_path),
        "release": snapshot_manifest.get("source", {}).get("release"),
        "release_date": snapshot_manifest.get("source", {}).get("release_date"),
    }
    manifest["output_files_sha256"] = {
        filename: sha256_file(output_dir / filename) for filename in sorted(tables)
    }
    manifest["curated_dataset_sha256"] = sha256_json(manifest["output_files_sha256"])
    atomic_write_json(output_dir / "curation_manifest.json", manifest)
    return manifest
