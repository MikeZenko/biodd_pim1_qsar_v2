"""Configurable edge-aware GINE used by the post-hoc tuned benchmark.

This module is intentionally separate from :mod:`pim1_qsar.models.gine`, whose
fixed architecture belongs to the sealed v2 analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

from .gnn_features import ATOM_FEATURE_DIM, BOND_FEATURE_DIM


try:  # Optional dependencies live in the isolated GNN environment.
    import torch
    from torch import nn
    import torch.nn.functional as functional
    from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool

    _IMPORT_ERROR: BaseException | None = None
except (ImportError, OSError) as exc:  # pragma: no cover - environment path
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    functional = None  # type: ignore[assignment]
    GINEConv = None  # type: ignore[assignment]
    global_add_pool = None  # type: ignore[assignment]
    global_mean_pool = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc


@dataclass(frozen=True)
class TunedGINEConfig:
    descriptor_dim: int
    hidden_dim: int
    depth: int
    ffn_hidden_dim: int
    ffn_layers: int
    dropout: float
    pooling: str
    normalization: str
    residual: bool
    initialization: str

    def __post_init__(self) -> None:
        if self.descriptor_dim < 0:
            raise ValueError("descriptor_dim must be non-negative")
        for name in ("hidden_dim", "depth", "ffn_hidden_dim", "ffn_layers"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.pooling not in {"mean", "add"}:
            raise ValueError("pooling must be mean or add")
        if self.normalization not in {"none", "batch", "layer"}:
            raise ValueError("normalization must be none, batch, or layer")
        if self.initialization not in {"xavier_uniform", "kaiming_uniform"}:
            raise ValueError("unsupported initialization")


def require_tuned_gine() -> None:
    if _IMPORT_ERROR is not None:  # pragma: no cover - environment path
        raise ImportError(
            "The tuned GINE requires torch and torch-geometric in the isolated "
            "GNN environment"
        ) from _IMPORT_ERROR


if _IMPORT_ERROR is None:

    class TunedGINERegressor(nn.Module):
        """Edge-aware residual GINE with tunable readout and normalization."""

        def __init__(self, config: TunedGINEConfig) -> None:
            super().__init__()
            self.config = config
            hidden = config.hidden_dim
            self.atom_projection = nn.Linear(ATOM_FEATURE_DIM, hidden)
            self.convolutions = nn.ModuleList()
            self.normalizations = nn.ModuleList()
            for _ in range(config.depth):
                update_network = nn.Sequential(
                    nn.Linear(hidden, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, hidden),
                )
                self.convolutions.append(
                    GINEConv(update_network, edge_dim=BOND_FEATURE_DIM, train_eps=True)
                )
                if config.normalization == "batch":
                    normalization = nn.BatchNorm1d(hidden)
                elif config.normalization == "layer":
                    normalization = nn.LayerNorm(hidden)
                else:
                    normalization = nn.Identity()
                self.normalizations.append(normalization)

            current = hidden + config.descriptor_dim
            head: list[nn.Module] = []
            for _ in range(config.ffn_layers - 1):
                head.extend(
                    [
                        nn.Linear(current, config.ffn_hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(config.dropout),
                    ]
                )
                current = config.ffn_hidden_dim
            head.append(nn.Linear(current, 1))
            self.predictor = nn.Sequential(*head)
            self.reset_parameters()

        def reset_parameters(self) -> None:
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    if self.config.initialization == "xavier_uniform":
                        nn.init.xavier_uniform_(module.weight)
                    else:
                        nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        def forward(self, data):
            x = self.atom_projection(data.x.float())
            edge_attr = data.edge_attr.float()
            for convolution, normalization in zip(
                self.convolutions, self.normalizations
            ):
                update = convolution(x, data.edge_index, edge_attr)
                update = functional.relu(normalization(update))
                update = functional.dropout(
                    update, p=self.config.dropout, training=self.training
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
                descriptors = data.descriptors.float().reshape(
                    graph_embedding.shape[0], self.config.descriptor_dim
                )
                graph_embedding = torch.cat([graph_embedding, descriptors], dim=1)
            return self.predictor(graph_embedding).squeeze(-1)

else:

    class TunedGINERegressor:  # pragma: no cover - environment path
        def __init__(self, config: TunedGINEConfig) -> None:
            require_tuned_gine()

