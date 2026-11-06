"""Deterministic molecular features for the v2 classical and GNN ablations.

Feature calculation is deliberately label-free.  The complete standardized
development set may therefore be featurized once, while every learned
preprocessing operation (imputation, scaling, selection) remains inside the
nested-validation estimator pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, GraphDescriptors, Lipinski, MolSurf, rdFingerprintGenerator

from pim1_qsar.artifacts import sha256_json


FEATURE_SCHEMA_VERSION = "classical_features_v2"
MORGAN_RADIUS = 3
MORGAN_BITS = 2048

# This order exactly matches the 23-descriptor legacy comparison while fixing
# its computation to the release-standardized parent structures.
PHYSICOCHEMICAL_DESCRIPTOR_NAMES: tuple[str, ...] = (
    "MolWt",
    "LogP",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "NumAromaticRings",
    "NumHeteroatoms",
    "FractionCSP3",
    "TPSA",
    "LabuteASA",
    "BalabanJ",
    "BertzCT",
    "Chi0v",
    "Chi1v",
    "Chi2v",
    "Chi3v",
    "Chi4v",
    "QED",
    "NumRings",
    "HeavyAtomCount",
    "FormalCharge",
    "NHOHCount",
    "NOCount",
)


def _formal_charge(molecule: Chem.Mol) -> float:
    return float(sum(atom.GetFormalCharge() for atom in molecule.GetAtoms()))


PHYSICOCHEMICAL_DESCRIPTOR_FUNCTIONS = (
    Descriptors.MolWt,
    Descriptors.MolLogP,
    Lipinski.NumHDonors,
    Lipinski.NumHAcceptors,
    Lipinski.NumRotatableBonds,
    Lipinski.NumAromaticRings,
    Lipinski.NumHeteroatoms,
    Lipinski.FractionCSP3,
    MolSurf.TPSA,
    MolSurf.LabuteASA,
    GraphDescriptors.BalabanJ,
    GraphDescriptors.BertzCT,
    GraphDescriptors.Chi0v,
    GraphDescriptors.Chi1v,
    GraphDescriptors.Chi2v,
    GraphDescriptors.Chi3v,
    GraphDescriptors.Chi4v,
    Descriptors.qed,
    Descriptors.RingCount,
    Descriptors.HeavyAtomCount,
    _formal_charge,
    Descriptors.NHOHCount,
    Descriptors.NOCount,
)


@dataclass(frozen=True)
class MolecularFeatures:
    """Aligned, immutable label-free feature matrices."""

    row_ids: tuple[str, ...]
    canonical_smiles: tuple[str, ...]
    ecfp6: np.ndarray
    physicochemical: np.ndarray
    manifest: dict[str, Any]

    def combined(self) -> np.ndarray:
        return np.concatenate(
            [self.ecfp6.astype(np.float64, copy=False), self.physicochemical],
            axis=1,
        )


def feature_schema() -> dict[str, Any]:
    """Return the immutable JSON-compatible feature definition."""

    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "standardized_structure_input": "canonical_parent_smiles",
        "ecfp": {
            "implementation": "RDKit MorganGenerator",
            "radius": MORGAN_RADIUS,
            "n_bits": MORGAN_BITS,
            "use_chirality": False,
            "use_bond_types": True,
            "count_simulation": False,
        },
        "physicochemical_descriptors": list(PHYSICOCHEMICAL_DESCRIPTOR_NAMES),
        "learned_preprocessing": "none_at_feature_generation",
    }


def _canonical_molecule(smiles: str) -> tuple[str, Chem.Mol]:
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("Every standardized SMILES must be a non-empty string")
    molecule = Chem.MolFromSmiles(smiles, sanitize=True)
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ValueError(f"RDKit could not parse standardized SMILES: {smiles!r}")
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    reparsed = Chem.MolFromSmiles(canonical, sanitize=True)
    if reparsed is None:
        raise ValueError(f"Canonical SMILES failed to round-trip: {smiles!r}")
    return canonical, reparsed


def compute_molecular_features(
    row_ids: Sequence[Any],
    smiles: Sequence[str],
) -> MolecularFeatures:
    """Calculate aligned ECFP6 and 23 physicochemical descriptors.

    Invalid molecules are fatal because silently dropping a molecule would
    invalidate the stored split assignments.  Non-finite descriptor values are
    retained as NaN and must be imputed inside a training-only pipeline.
    """

    ids = tuple(str(value).strip() for value in row_ids)
    smiles_values = tuple(smiles)
    if not ids or len(ids) != len(smiles_values):
        raise ValueError("row_ids and smiles must be non-empty and equally sized")
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("row_ids must be unique non-empty values")

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=MORGAN_RADIUS,
        fpSize=MORGAN_BITS,
        countSimulation=False,
        includeChirality=False,
        useBondTypes=True,
    )
    ecfp = np.empty((len(ids), MORGAN_BITS), dtype=np.uint8)
    descriptors = np.empty(
        (len(ids), len(PHYSICOCHEMICAL_DESCRIPTOR_NAMES)), dtype=np.float64
    )
    canonical_smiles: list[str] = []
    failures: list[str] = []

    for row_index, (row_id, value) in enumerate(zip(ids, smiles_values)):
        try:
            canonical, molecule = _canonical_molecule(value)
            fingerprint = generator.GetFingerprint(molecule)
            bit_array = np.zeros((MORGAN_BITS,), dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(fingerprint, bit_array)
            ecfp[row_index] = bit_array
            descriptor_values: list[float] = []
            for function in PHYSICOCHEMICAL_DESCRIPTOR_FUNCTIONS:
                try:
                    result = float(function(molecule))
                except Exception:
                    result = float("nan")
                descriptor_values.append(result if np.isfinite(result) else float("nan"))
            descriptors[row_index] = descriptor_values
            canonical_smiles.append(canonical)
        except Exception as exc:
            failures.append(f"{row_id}:{type(exc).__name__}:{exc}")

    if failures:
        raise ValueError(
            "Feature generation failed; split alignment forbids row dropping: "
            + "; ".join(failures[:5])
        )

    schema = feature_schema()
    manifest = {
        **schema,
        "schema_sha256": sha256_json(schema),
        "n_rows": len(ids),
        "row_ids_sha256": sha256_json(list(ids)),
        "canonical_smiles_sha256": sha256_json(canonical_smiles),
        "ecfp_shape": list(ecfp.shape),
        "physicochemical_shape": list(descriptors.shape),
        "physicochemical_nonfinite_count": int((~np.isfinite(descriptors)).sum()),
    }
    ecfp.setflags(write=False)
    descriptors.setflags(write=False)
    return MolecularFeatures(
        row_ids=ids,
        canonical_smiles=tuple(canonical_smiles),
        ecfp6=ecfp,
        physicochemical=descriptors,
        manifest=manifest,
    )


def features_from_frame(
    frame: pd.DataFrame,
    *,
    identity_col: str = "standardized_parent_inchi_key",
    smiles_col: str = "standardized_parent_smiles",
) -> MolecularFeatures:
    missing = sorted({identity_col, smiles_col} - set(frame.columns))
    if missing:
        raise ValueError(f"Missing feature columns: {', '.join(missing)}")
    return compute_molecular_features(frame[identity_col].tolist(), frame[smiles_col].tolist())

