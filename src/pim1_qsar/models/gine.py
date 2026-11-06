"""Edge-aware Graph Isomorphism Network baseline for molecular regression."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Any

import numpy as np

from .gnn_features import ATOM_FEATURE_DIM, BOND_FEATURE_DIM


try:  # Optional in the validated classical-model environment.
    import torch
    from torch import nn
    import torch.nn.functional as functional
    from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool

    _GNN_IMPORT_ERROR: BaseException | None = None
except (ImportError, OSError) as exc:  # pragma: no cover - environment path
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    functional = None  # type: ignore[assignment]
    GINEConv = None  # type: ignore[assignment]
    global_add_pool = None  # type: ignore[assignment]
    global_mean_pool = None  # type: ignore[assignment]
    _GNN_IMPORT_ERROR = exc


@dataclass(frozen=True)
class GINEConfig:
    """Fully serializable architecture contract for one GINE regressor."""

    atom_feature_dim: int = ATOM_FEATURE_DIM
    bond_feature_dim: int = BOND_FEATURE_DIM
    descriptor_dim: int = 0
    hidden_dim: int = 256
    num_layers: int = 4
    ffn_hidden_dim: int = 256
    ffn_num_layers: int = 2
    dropout: float = 0.1
    pooling: str = "add"
    residual: bool = True
    batch_norm: bool = True

    def __post_init__(self) -> None:
        for name in (
            "atom_feature_dim",
            "bond_feature_dim",
            "hidden_dim",
            "num_layers",
            "ffn_hidden_dim",
            "ffn_num_layers",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.descriptor_dim, bool)
            or not isinstance(self.descriptor_dim, int)
            or self.descriptor_dim < 0
        ):
            raise ValueError("descriptor_dim must be a non-negative integer")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.pooling not in {"add", "mean"}:
            raise ValueError("pooling must be 'add' or 'mean'")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def gnn_dependencies_available() -> bool:
    """Return whether Torch and PyG imported successfully."""

    return _GNN_IMPORT_ERROR is None


def require_gnn_dependencies() -> None:
    """Raise an actionable error when the isolated runtime is unavailable."""

    if _GNN_IMPORT_ERROR is not None:
        raise ImportError(
            "GINERegressor requires torch==2.12.1 and "
            "torch-geometric==2.8.0 in the isolated .venv-gnn environment"
        ) from _GNN_IMPORT_ERROR


def set_global_determinism(seed: int, *, strict: bool = True) -> None:
    """Seed Python, NumPy, and Torch and request deterministic Torch kernels."""

    require_gnn_dependencies()
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=not strict)


if _GNN_IMPORT_ERROR is None:

    class GINERegressor(nn.Module):
        """Graph-level pIC50 regressor with explicit edge attributes."""

        def __init__(self, config: GINEConfig | None = None) -> None:
            super().__init__()
            self.config = config or GINEConfig()
            hidden = self.config.hidden_dim

            self.atom_projection = nn.Linear(
                self.config.atom_feature_dim,
                hidden,
            )
            self.convolutions = nn.ModuleList()
            self.normalizations = nn.ModuleList()
            for _ in range(self.config.num_layers):
                update_network = nn.Sequential(
                    nn.Linear(hidden, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, hidden),
                )
                self.convolutions.append(
                    GINEConv(
                        nn=update_network,
                        edge_dim=self.config.bond_feature_dim,
                        train_eps=True,
                    )
                )
                self.normalizations.append(
                    nn.BatchNorm1d(hidden)
                    if self.config.batch_norm
                    else nn.Identity()
                )

            ffn_input = hidden + self.config.descriptor_dim
            ffn_layers: list[nn.Module] = []
            current_dim = ffn_input
            for _ in range(self.config.ffn_num_layers - 1):
                ffn_layers.extend(
                    [
                        nn.Linear(current_dim, self.config.ffn_hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(self.config.dropout),
                    ]
                )
                current_dim = self.config.ffn_hidden_dim
            ffn_layers.append(nn.Linear(current_dim, 1))
            self.predictor = nn.Sequential(*ffn_layers)

        def forward(self, data):
            """Predict one pIC50 value for each graph in a PyG batch."""

            x = self.atom_projection(data.x.float())
            edge_attr = data.edge_attr.float()
            for convolution, normalization in zip(
                self.convolutions,
                self.normalizations,
            ):
                update = convolution(x, data.edge_index, edge_attr)
                update = normalization(update)
                update = functional.relu(update)
                update = functional.dropout(
                    update,
                    p=self.config.dropout,
                    training=self.training,
                )
                x = x + update if self.config.residual else update

            batch = getattr(data, "batch", None)
            if batch is None:
                batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            if self.config.pooling == "add":
                graph_embedding = global_add_pool(x, batch)
            else:
                graph_embedding = global_mean_pool(x, batch)

            if self.config.descriptor_dim:
                if not hasattr(data, "descriptors"):
                    raise ValueError(
                        "This GINEConfig requires molecule descriptors, but the "
                        "PyG batch has no 'descriptors' attribute"
                    )
                descriptors = data.descriptors.float()
                if descriptors.ndim == 1:
                    descriptors = descriptors.reshape(graph_embedding.shape[0], -1)
                expected = (graph_embedding.shape[0], self.config.descriptor_dim)
                if tuple(descriptors.shape) != expected:
                    raise ValueError(
                        f"Expected descriptor tensor with shape {expected}, "
                        f"found {tuple(descriptors.shape)}"
                    )
                graph_embedding = torch.cat(
                    [graph_embedding, descriptors],
                    dim=1,
                )

            return self.predictor(graph_embedding).squeeze(-1)

else:

    class GINERegressor:  # pragma: no cover - exercised without optional deps
        """Import-safe placeholder that explains how to enable the model."""

        def __init__(self, config: GINEConfig | None = None) -> None:
            require_gnn_dependencies()

