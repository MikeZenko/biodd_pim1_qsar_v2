"""Optional graph-learning models for the PIM1 benchmark."""

from .chemprop_adapter import (
    ChempropAdapter,
    ChempropConfig,
    ChempropInputBatch,
    EXPECTED_CHEMPROP_VERSION,
    installed_chemprop_version,
    require_chemprop,
)
from .gine import (
    GINEConfig,
    GINERegressor,
    gnn_dependencies_available,
    require_gnn_dependencies,
    set_global_determinism,
)
from .gnn_features import (
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    MoleculeGraphFeatures,
    canonicalize_smiles,
    feature_schema,
    featurize_smiles,
    to_pyg_data,
)

__all__ = [
    "ATOM_FEATURE_DIM",
    "BOND_FEATURE_DIM",
    "ChempropAdapter",
    "ChempropConfig",
    "ChempropInputBatch",
    "EXPECTED_CHEMPROP_VERSION",
    "GINEConfig",
    "GINERegressor",
    "MoleculeGraphFeatures",
    "canonicalize_smiles",
    "feature_schema",
    "featurize_smiles",
    "gnn_dependencies_available",
    "installed_chemprop_version",
    "require_chemprop",
    "require_gnn_dependencies",
    "set_global_determinism",
    "to_pyg_data",
]
