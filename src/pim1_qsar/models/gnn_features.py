"""Deterministic two-dimensional molecular graph featurization.

The v2 benchmark materializes graph tensors from the already-standardized
canonical SMILES. This module deliberately performs no salt stripping,
tautomer conversion, or activity-dependent transformation; those operations
belong to the frozen curation stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdchem


ATOMIC_NUM_CHOICES: tuple[int, ...] = tuple(range(1, 119))
DEGREE_CHOICES: tuple[int, ...] = tuple(range(0, 7))
FORMAL_CHARGE_CHOICES: tuple[int, ...] = tuple(range(-3, 4))
TOTAL_H_CHOICES: tuple[int, ...] = tuple(range(0, 5))
CHIRAL_TAG_CHOICES: tuple[rdchem.ChiralType, ...] = (
    rdchem.ChiralType.CHI_UNSPECIFIED,
    rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    rdchem.ChiralType.CHI_OTHER,
)
HYBRIDIZATION_CHOICES: tuple[rdchem.HybridizationType, ...] = (
    rdchem.HybridizationType.S,
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
    rdchem.HybridizationType.UNSPECIFIED,
    rdchem.HybridizationType.OTHER,
)
BOND_TYPE_CHOICES: tuple[rdchem.BondType, ...] = (
    rdchem.BondType.SINGLE,
    rdchem.BondType.DOUBLE,
    rdchem.BondType.TRIPLE,
    rdchem.BondType.AROMATIC,
)
BOND_STEREO_CHOICES: tuple[rdchem.BondStereo, ...] = (
    rdchem.BondStereo.STEREONONE,
    rdchem.BondStereo.STEREOANY,
    rdchem.BondStereo.STEREOZ,
    rdchem.BondStereo.STEREOE,
    rdchem.BondStereo.STEREOCIS,
    rdchem.BondStereo.STEREOTRANS,
)


def _one_hot_with_unknown(value: Any, choices: Sequence[Any]) -> list[float]:
    """One-hot encode against a fixed vocabulary with a final unknown bin."""

    encoded = [0.0] * (len(choices) + 1)
    try:
        encoded[choices.index(value)] = 1.0
    except ValueError:
        encoded[-1] = 1.0
    return encoded


ATOM_FEATURE_DIM = (
    len(ATOMIC_NUM_CHOICES)
    + 1
    + len(DEGREE_CHOICES)
    + 1
    + len(FORMAL_CHARGE_CHOICES)
    + 1
    + len(CHIRAL_TAG_CHOICES)
    + 1
    + len(TOTAL_H_CHOICES)
    + 1
    + len(HYBRIDIZATION_CHOICES)
    + 1
    + 3  # aromatic, ring membership, and mass / 100
)
BOND_FEATURE_DIM = (
    len(BOND_TYPE_CHOICES)
    + 1
    + len(BOND_STEREO_CHOICES)
    + 1
    + 2  # conjugation and ring membership
)


@dataclass(frozen=True)
class MoleculeGraphFeatures:
    """Framework-neutral graph tensors for one standardized molecule."""

    canonical_smiles: str
    atom_features: np.ndarray
    edge_index: np.ndarray
    bond_features: np.ndarray

    def validate(self) -> None:
        """Raise when a graph violates the stable feature contract."""

        if not self.canonical_smiles:
            raise ValueError("canonical_smiles cannot be empty")
        if self.atom_features.ndim != 2:
            raise ValueError("atom_features must be a two-dimensional array")
        if self.atom_features.shape[1] != ATOM_FEATURE_DIM:
            raise ValueError(
                f"Expected {ATOM_FEATURE_DIM} atom features, "
                f"found {self.atom_features.shape[1]}"
            )
        if self.atom_features.shape[0] == 0:
            raise ValueError("A molecular graph must contain at least one atom")
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, n_directed_edges)")
        if self.bond_features.ndim != 2:
            raise ValueError("bond_features must be a two-dimensional array")
        if self.bond_features.shape[1] != BOND_FEATURE_DIM:
            raise ValueError(
                f"Expected {BOND_FEATURE_DIM} bond features, "
                f"found {self.bond_features.shape[1]}"
            )
        if self.edge_index.shape[1] != self.bond_features.shape[0]:
            raise ValueError("edge_index and bond_features are not aligned")
        if self.edge_index.size:
            if self.edge_index.min() < 0:
                raise ValueError("edge_index contains a negative atom index")
            if self.edge_index.max() >= self.atom_features.shape[0]:
                raise ValueError("edge_index refers to a missing atom")
        if not np.isfinite(self.atom_features).all():
            raise ValueError("atom_features contains a non-finite value")
        if not np.isfinite(self.bond_features).all():
            raise ValueError("bond_features contains a non-finite value")


def canonicalize_smiles(smiles: str) -> str:
    """Parse and canonically serialize one isomeric SMILES string."""

    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("SMILES must be a non-empty string")
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ValueError(f"RDKit could not parse a molecular graph: {smiles!r}")
    canonical = Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=True,
    )
    if not canonical:
        raise ValueError(f"RDKit produced an empty canonical SMILES: {smiles!r}")
    return canonical


def atom_features(atom: rdchem.Atom) -> np.ndarray:
    """Return the fixed, documented feature vector for one atom."""

    values = (
        _one_hot_with_unknown(atom.GetAtomicNum(), ATOMIC_NUM_CHOICES)
        + _one_hot_with_unknown(atom.GetTotalDegree(), DEGREE_CHOICES)
        + _one_hot_with_unknown(atom.GetFormalCharge(), FORMAL_CHARGE_CHOICES)
        + _one_hot_with_unknown(atom.GetChiralTag(), CHIRAL_TAG_CHOICES)
        + _one_hot_with_unknown(atom.GetTotalNumHs(), TOTAL_H_CHOICES)
        + _one_hot_with_unknown(atom.GetHybridization(), HYBRIDIZATION_CHOICES)
        + [
            float(atom.GetIsAromatic()),
            float(atom.IsInRing()),
            float(atom.GetMass()) / 100.0,
        ]
    )
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (ATOM_FEATURE_DIM,):
        raise AssertionError("Atom feature schema changed unexpectedly")
    return result


def bond_features(bond: rdchem.Bond) -> np.ndarray:
    """Return the fixed, documented feature vector for one directed bond."""

    values = (
        _one_hot_with_unknown(bond.GetBondType(), BOND_TYPE_CHOICES)
        + _one_hot_with_unknown(bond.GetStereo(), BOND_STEREO_CHOICES)
        + [float(bond.GetIsConjugated()), float(bond.IsInRing())]
    )
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (BOND_FEATURE_DIM,):
        raise AssertionError("Bond feature schema changed unexpectedly")
    return result


def featurize_smiles(smiles: str) -> MoleculeGraphFeatures:
    """Create a deterministic bidirectional graph from canonical SMILES."""

    canonical = canonicalize_smiles(smiles)
    molecule = Chem.MolFromSmiles(canonical)
    if molecule is None:
        raise AssertionError("Canonical SMILES failed to round-trip through RDKit")

    atoms = np.stack(
        [atom_features(atom) for atom in molecule.GetAtoms()],
        axis=0,
    ).astype(np.float32, copy=False)

    directed_edges: list[tuple[int, int, np.ndarray]] = []
    for bond in molecule.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        features = bond_features(bond)
        directed_edges.append((begin, end, features))
        directed_edges.append((end, begin, features.copy()))
    directed_edges.sort(key=lambda item: (item[0], item[1]))

    if directed_edges:
        edge_index = np.asarray(
            [(source, target) for source, target, _ in directed_edges],
            dtype=np.int64,
        ).T
        bonds = np.stack(
            [features for _, _, features in directed_edges],
            axis=0,
        ).astype(np.float32, copy=False)
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
        bonds = np.empty((0, BOND_FEATURE_DIM), dtype=np.float32)

    graph = MoleculeGraphFeatures(
        canonical_smiles=canonical,
        atom_features=atoms,
        edge_index=edge_index,
        bond_features=bonds,
    )
    graph.validate()
    return graph


def feature_schema() -> dict[str, Any]:
    """Return a JSON-compatible description of the immutable feature layout."""

    return {
        "schema_version": "gnn_features_v1",
        "atom_feature_dim": ATOM_FEATURE_DIM,
        "bond_feature_dim": BOND_FEATURE_DIM,
        "directed_edges": True,
        "atom_fields": [
            "atomic_number",
            "total_degree",
            "formal_charge",
            "chiral_tag",
            "total_hydrogens",
            "hybridization",
            "is_aromatic",
            "is_in_ring",
            "mass_div_100",
        ],
        "bond_fields": [
            "bond_type",
            "stereochemistry",
            "is_conjugated",
            "is_in_ring",
        ],
        "vocabularies": {
            "atomic_number": list(ATOMIC_NUM_CHOICES),
            "total_degree": list(DEGREE_CHOICES),
            "formal_charge": list(FORMAL_CHARGE_CHOICES),
            "chiral_tag": [str(value) for value in CHIRAL_TAG_CHOICES],
            "total_hydrogens": list(TOTAL_H_CHOICES),
            "hybridization": [str(value) for value in HYBRIDIZATION_CHOICES],
            "bond_type": [str(value) for value in BOND_TYPE_CHOICES],
            "bond_stereochemistry": [
                str(value) for value in BOND_STEREO_CHOICES
            ],
            "unknown_bin": "last_position_of_each_categorical_field",
        },
    }


def to_pyg_data(
    graph: MoleculeGraphFeatures,
    *,
    target: float | None = None,
    descriptors: Sequence[float] | np.ndarray | None = None,
    row_id: str | None = None,
):
    """Convert framework-neutral features to a PyG ``Data`` object.

    Torch and PyG are imported lazily, so curation and contract tests can use
    this module in the lightweight classical-model environment.
    """

    try:
        import torch
        from torch_geometric.data import Data
    except (ImportError, OSError) as exc:  # pragma: no cover - environment path
        raise ImportError(
            "to_pyg_data requires the isolated GNN dependencies; "
            "run scripts/setup_gnn_vm.sh on the compute VM"
        ) from exc

    graph.validate()
    data = Data(
        x=torch.as_tensor(graph.atom_features, dtype=torch.float32),
        edge_index=torch.as_tensor(graph.edge_index, dtype=torch.long),
        edge_attr=torch.as_tensor(graph.bond_features, dtype=torch.float32),
    )
    data.canonical_smiles = graph.canonical_smiles
    if row_id is not None:
        if not isinstance(row_id, str) or not row_id:
            raise ValueError("row_id must be a non-empty string")
        data.row_id = row_id
    if target is not None:
        if not np.isfinite(target):
            raise ValueError("target must be finite")
        data.y = torch.tensor([float(target)], dtype=torch.float32)
    if descriptors is not None:
        descriptor_array = np.asarray(descriptors, dtype=np.float32)
        if descriptor_array.ndim != 1 or descriptor_array.size == 0:
            raise ValueError("descriptors must be a non-empty one-dimensional array")
        if not np.isfinite(descriptor_array).all():
            raise ValueError("descriptors contains a non-finite value")
        data.descriptors = torch.as_tensor(
            descriptor_array.reshape(1, -1),
            dtype=torch.float32,
        )
    return data
