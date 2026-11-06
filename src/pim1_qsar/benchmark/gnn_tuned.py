"""Post-hoc, development-only nested tuning for molecular graph models.

The sealed v2 temporal analysis is deliberately out of scope.  This runner
accepts only the development table and its frozen outer-fold assignments.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from pim1_qsar.artifacts import atomic_write_json, atomic_write_text, sha256_file, sha256_json
from pim1_qsar.benchmark.classical import (
    IDENTITY_COLUMN,
    SCAFFOLD_COLUMN,
    SMILES_COLUMN,
    TARGET_COLUMN,
    _normalize_assignments,
    _validate_outer_fold,
    make_shared_inner_splits,
    regression_metrics,
)
from pim1_qsar.benchmark.gnn import (
    _chemprop_datapoints,
    _configure_torch,
    _fit_descriptor_transform,
    _precompute_graphs,
    _pyg_dataset,
    _target_scale,
)
from pim1_qsar.features import features_from_frame
from pim1_qsar.models.chemprop_adapter import require_chemprop
from pim1_qsar.models.gine_tuned import TunedGINEConfig, TunedGINERegressor


SCHEMA_VERSION = "gnn_tuned_posthoc_v1"
FAMILIES = ("edge_aware_gine", "chemprop_dmpnn", "chemprop_chemeleon")
CHEMELEON_SHA256 = "c376624d3407204e780a0ed13a9ac097cc9bb1c13ef89cdbc633c1715c183651"


@dataclass(frozen=True)
class GraphCandidate:
    family: str
    representation: str
    depth: int
    hidden_dim: int
    ffn_hidden_dim: int
    ffn_layers: int
    dropout: float
    learning_rate: float
    weight_decay: float
    batch_size: int
    pooling: str
    normalization: str
    residual: bool
    initialization: str
    activation: str
    featurization: str
    encoder_training: str

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"Unsupported graph family: {self.family}")
        if self.representation not in {"graph", "graph_plus_rdkit2d"}:
            raise ValueError("Unsupported representation")
        if min(self.depth, self.hidden_dim, self.ffn_hidden_dim, self.ffn_layers) <= 0:
            raise ValueError("Architecture dimensions must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer settings")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

    @property
    def candidate_id(self) -> str:
        return self.family + "-" + sha256_json(asdict(self))[:12]

    @property
    def use_descriptors(self) -> bool:
        return self.representation == "graph_plus_rdkit2d"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SearchProtocol:
    candidate_budget: int
    inner_folds: int
    rerank_top: int
    rerank_seeds: tuple[int, ...]
    refit_seeds: tuple[int, ...]
    max_epochs: int
    patience: int
    min_delta: float
    torch_threads: int


def load_search_protocol(path: Path) -> tuple[dict[str, Any], SearchProtocol]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != "gnn_search_v1":
        raise ValueError("Unsupported tuned-GNN search schema")
    if raw.get("status") != "post_hoc_development_only":
        raise ValueError("The tuned-GNN search must remain development-only")
    protocol = SearchProtocol(
        candidate_budget=int(raw["candidate_budget_per_family"]),
        inner_folds=int(raw["inner_scaffold_folds"]),
        rerank_top=int(raw["rerank_top_candidates"]),
        rerank_seeds=tuple(int(value) for value in raw["rerank_initialization_seeds"]),
        refit_seeds=tuple(
            int(value) for value in raw["outer_refit_initialization_seeds"]
        ),
        max_epochs=int(raw["max_epochs"]),
        patience=int(raw["patience"]),
        min_delta=float(raw["min_delta_rmse"]),
        torch_threads=int(raw["torch_threads_per_process"]),
    )
    if protocol.candidate_budget < protocol.rerank_top:
        raise ValueError("Candidate budget is smaller than reranking shortlist")
    if len(protocol.rerank_seeds) < 2 or len(protocol.refit_seeds) < 2:
        raise ValueError("At least two initialization seeds are required")
    return raw, protocol


def _choice(rng: np.random.Generator, values: Sequence[Any]) -> Any:
    return values[int(rng.integers(0, len(values)))]


def _log_uniform(rng: np.random.Generator, lower: float, upper: float) -> float:
    return float(math.exp(rng.uniform(math.log(lower), math.log(upper))))


def generate_candidates(
    family: str, protocol: SearchProtocol, *, seed: int = 62017
) -> tuple[GraphCandidate, ...]:
    """Generate a deterministic random-search design with a documented baseline."""

    if family not in FAMILIES:
        raise ValueError(f"Unsupported graph family: {family}")
    family_seed = seed + FAMILIES.index(family) * 1009
    rng = np.random.default_rng(family_seed)
    candidates: list[GraphCandidate] = []
    if family == "edge_aware_gine":
        baseline = GraphCandidate(
            family=family,
            representation="graph_plus_rdkit2d",
            depth=3,
            hidden_dim=128,
            ffn_hidden_dim=128,
            ffn_layers=2,
            dropout=0.1,
            learning_rate=3e-4,
            weight_decay=1e-5,
            batch_size=64,
            pooling="mean",
            normalization="batch",
            residual=True,
            initialization="xavier_uniform",
            activation="relu",
            featurization="rich_atom_bond_v1",
            encoder_training="random",
        )
    elif family == "chemprop_dmpnn":
        baseline = GraphCandidate(
            family=family,
            representation="graph",
            depth=3,
            hidden_dim=128,
            ffn_hidden_dim=128,
            ffn_layers=2,
            dropout=0.1,
            learning_rate=3e-4,
            weight_decay=0.0,
            batch_size=64,
            pooling="mean",
            normalization="batch",
            residual=False,
            initialization="chemprop_default",
            activation="relu",
            featurization="chemprop_v2",
            encoder_training="random",
        )
    else:
        baseline = GraphCandidate(
            family=family,
            representation="graph",
            depth=6,
            hidden_dim=2048,
            ffn_hidden_dim=128,
            ffn_layers=2,
            dropout=0.1,
            learning_rate=3e-4,
            weight_decay=1e-5,
            batch_size=64,
            pooling="mean",
            normalization="batch",
            residual=False,
            initialization="pretrained_chemeleon",
            activation="relu",
            featurization="chemprop_v2",
            encoder_training="frozen",
        )
    candidates.append(baseline)

    while len(candidates) < protocol.candidate_budget:
        if family == "edge_aware_gine":
            candidate = GraphCandidate(
                family=family,
                representation=_choice(rng, ["graph", "graph_plus_rdkit2d"]),
                depth=int(_choice(rng, [2, 3, 4, 5, 6])),
                hidden_dim=int(_choice(rng, [64, 128, 256, 512])),
                ffn_hidden_dim=int(_choice(rng, [64, 128, 256, 512])),
                ffn_layers=int(_choice(rng, [1, 2, 3])),
                dropout=float(rng.uniform(0.0, 0.4)),
                learning_rate=_log_uniform(rng, 1e-5, 3e-3),
                weight_decay=float(
                    _choice(rng, [0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3])
                ),
                batch_size=int(_choice(rng, [32, 64, 128])),
                pooling=_choice(rng, ["mean", "add"]),
                normalization=_choice(rng, ["none", "batch", "layer"]),
                residual=bool(_choice(rng, [False, True])),
                initialization=_choice(rng, ["xavier_uniform", "kaiming_uniform"]),
                activation="relu",
                featurization="rich_atom_bond_v1",
                encoder_training="random",
            )
        elif family == "chemprop_dmpnn":
            candidate = GraphCandidate(
                family=family,
                representation=_choice(rng, ["graph", "graph_plus_rdkit2d"]),
                depth=int(_choice(rng, [2, 3, 4, 5, 6])),
                hidden_dim=int(_choice(rng, [64, 128, 256, 512])),
                ffn_hidden_dim=int(_choice(rng, [64, 128, 256, 512])),
                ffn_layers=int(_choice(rng, [1, 2, 3])),
                dropout=float(rng.uniform(0.0, 0.4)),
                learning_rate=_log_uniform(rng, 1e-5, 3e-3),
                weight_decay=float(
                    _choice(rng, [0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3])
                ),
                batch_size=int(_choice(rng, [32, 64, 128])),
                pooling=_choice(rng, ["mean", "sum", "norm"]),
                normalization=_choice(rng, ["none", "batch"]),
                residual=False,
                initialization="chemprop_default",
                activation=_choice(rng, ["relu", "leakyrelu", "elu"]),
                featurization=_choice(
                    rng, ["chemprop_v1", "chemprop_v2", "chemprop_organic"]
                ),
                encoder_training="random",
            )
        else:
            candidate = GraphCandidate(
                family=family,
                representation=_choice(rng, ["graph", "graph_plus_rdkit2d"]),
                depth=6,
                hidden_dim=2048,
                ffn_hidden_dim=int(_choice(rng, [64, 128, 256, 512])),
                ffn_layers=int(_choice(rng, [1, 2, 3])),
                dropout=float(rng.uniform(0.0, 0.4)),
                learning_rate=_log_uniform(rng, 1e-5, 3e-3),
                weight_decay=float(
                    _choice(rng, [0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3])
                ),
                batch_size=int(_choice(rng, [32, 64, 128])),
                pooling="mean",
                normalization=_choice(rng, ["none", "batch"]),
                residual=False,
                initialization="pretrained_chemeleon",
                activation=_choice(rng, ["relu", "leakyrelu", "elu"]),
                featurization="chemprop_v2",
                encoder_training="frozen",
            )
        if candidate.candidate_id not in {value.candidate_id for value in candidates}:
            candidates.append(candidate)
    return tuple(candidates)


def _descriptor_map(data: pd.DataFrame) -> tuple[dict[str, np.ndarray], str]:
    features = features_from_frame(data)
    mapping = {
        identity: features.physicochemical[index].copy()
        for index, identity in enumerate(features.row_ids)
    }
    return mapping, sha256_json(features.manifest)


def _safe_training_batch_size(n_rows: int, requested: int) -> int:
    """Avoid singleton final batches, which are invalid for batch normalization."""

    batch_size = min(n_rows, requested)
    while batch_size > 2 and n_rows % batch_size == 1:
        batch_size -= 1
    return batch_size


def _gine_model(candidate: GraphCandidate) -> TunedGINERegressor:
    return TunedGINERegressor(
        TunedGINEConfig(
            descriptor_dim=23 if candidate.use_descriptors else 0,
            hidden_dim=candidate.hidden_dim,
            depth=candidate.depth,
            ffn_hidden_dim=candidate.ffn_hidden_dim,
            ffn_layers=candidate.ffn_layers,
            dropout=candidate.dropout,
            pooling=candidate.pooling,
            normalization=candidate.normalization,
            residual=candidate.residual,
            initialization=candidate.initialization,
        )
    )


def _gine_data(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    graphs: Mapping[str, Any],
    descriptors: Mapping[str, np.ndarray],
    candidate: GraphCandidate,
):
    y_train = train[TARGET_COLUMN].to_numpy(dtype=float)
    y_validation = validation[TARGET_COLUMN].to_numpy(dtype=float)
    target_mean, target_scale = _target_scale(y_train)
    train_desc = validation_desc = None
    if candidate.use_descriptors:
        raw_train = np.stack([descriptors[value] for value in train[IDENTITY_COLUMN]])
        raw_validation = np.stack(
            [descriptors[value] for value in validation[IDENTITY_COLUMN]]
        )
        train_desc, validation_desc = _fit_descriptor_transform(raw_train, raw_validation)
    training = _pyg_dataset(
        train[IDENTITY_COLUMN].astype(str).tolist(),
        (y_train - target_mean) / target_scale,
        graphs,
        train_desc,
    )
    validating = _pyg_dataset(
        validation[IDENTITY_COLUMN].astype(str).tolist(),
        np.zeros(len(validation), dtype=float),
        graphs,
        validation_desc,
    )
    return training, validating, y_validation, target_mean, target_scale


def _gine_inner_score(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    graphs: Mapping[str, Any],
    descriptors: Mapping[str, np.ndarray],
    candidate: GraphCandidate,
    protocol: SearchProtocol,
    seed: int,
) -> tuple[float, int, int]:
    import torch
    from torch_geometric.loader import DataLoader

    _configure_torch(seed, protocol.torch_threads)
    training, validating, y_validation, target_mean, target_scale = _gine_data(
        train,
        validation,
        graphs=graphs,
        descriptors=descriptors,
        candidate=candidate,
    )
    loader = DataLoader(
        training,
        batch_size=_safe_training_batch_size(len(training), candidate.batch_size),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    validation_loader = DataLoader(
        validating, batch_size=len(validating), shuffle=False, num_workers=0
    )
    model = _gine_model(candidate)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=candidate.learning_rate, weight_decay=candidate.weight_decay
    )
    best_rmse = float("inf")
    best_epoch = 1
    stale = 0
    for epoch in range(1, protocol.max_epochs + 1):
        model.train()
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch)
            loss = torch.nn.functional.mse_loss(prediction, batch.y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        predictions: list[np.ndarray] = []
        with torch.no_grad():
            for batch in validation_loader:
                values = model(batch).detach().cpu().numpy()
                predictions.append(values * target_scale + target_mean)
        rmse = regression_metrics(y_validation, np.concatenate(predictions))["rmse"]
        if best_rmse - rmse > protocol.min_delta:
            best_rmse, best_epoch, stale = float(rmse), epoch, 0
        else:
            stale += 1
        if stale >= protocol.patience:
            break
    parameters = sum(value.numel() for value in model.parameters())
    return best_rmse, best_epoch, int(parameters)


def _gine_refit_predict(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    graphs: Mapping[str, Any],
    descriptors: Mapping[str, np.ndarray],
    candidate: GraphCandidate,
    protocol: SearchProtocol,
    seed: int,
    epochs: int,
) -> np.ndarray:
    import torch
    from torch_geometric.loader import DataLoader

    _configure_torch(seed, protocol.torch_threads)
    training, validating, _, target_mean, target_scale = _gine_data(
        train,
        validation,
        graphs=graphs,
        descriptors=descriptors,
        candidate=candidate,
    )
    loader = DataLoader(
        training,
        batch_size=_safe_training_batch_size(len(training), candidate.batch_size),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    validation_loader = DataLoader(
        validating, batch_size=len(validating), shuffle=False, num_workers=0
    )
    model = _gine_model(candidate)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=candidate.learning_rate, weight_decay=candidate.weight_decay
    )
    for _ in range(epochs):
        model.train()
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch)
            loss = torch.nn.functional.mse_loss(prediction, batch.y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    model.eval()
    output: list[np.ndarray] = []
    with torch.no_grad():
        for batch in validation_loader:
            output.append(model(batch).detach().cpu().numpy() * target_scale + target_mean)
    return np.concatenate(output)


def _chemprop_featurizer(name: str):
    from chemprop import featurizers

    mode = name.removeprefix("chemprop_")
    atom_featurizer = getattr(featurizers.MultiHotAtomFeaturizer, mode)()
    return featurizers.SimpleMoleculeMolGraphFeaturizer(
        atom_featurizer=atom_featurizer
    )


def _chemprop_datasets_tuned(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    descriptors: Mapping[str, np.ndarray],
    candidate: GraphCandidate,
):
    from chemprop import data

    descriptor_source = descriptors if candidate.use_descriptors else None
    featurizer = _chemprop_featurizer(candidate.featurization)
    train_dataset = data.MoleculeDataset(
        _chemprop_datapoints(train, descriptor_source), featurizer
    )
    validation_dataset = data.MoleculeDataset(
        _chemprop_datapoints(validation, descriptor_source), featurizer
    )
    target_scaler = train_dataset.normalize_targets()
    validation_dataset.normalize_targets(target_scaler)
    if candidate.use_descriptors:
        descriptor_scaler = train_dataset.normalize_inputs("X_d")
        validation_dataset.normalize_inputs("X_d", descriptor_scaler)
    else:
        descriptor_scaler = None
    return train_dataset, validation_dataset, target_scaler, descriptor_scaler


def _precompute_chemeleon_embeddings(
    data_frame: pd.DataFrame,
    *,
    checkpoint_path: Path,
    torch_threads: int,
) -> dict[str, np.ndarray]:
    """Materialize the frozen CheMeleon representation once per worker."""

    require_chemprop()
    import torch
    from chemprop import data, nn

    if sha256_file(checkpoint_path) != CHEMELEON_SHA256:
        raise RuntimeError("CheMeleon checkpoint hash mismatch")
    _configure_torch(7001, torch_threads)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    message_passing = nn.BondMessagePassing(**payload["hyper_parameters"])
    message_passing.load_state_dict(payload["state_dict"], strict=True)
    message_passing.eval()
    aggregation = nn.MeanAggregation()
    dataset = data.MoleculeDataset(
        _chemprop_datapoints(data_frame, None), _chemprop_featurizer("chemprop_v2")
    )
    loader = data.build_dataloader(
        dataset, batch_size=64, num_workers=0, shuffle=False
    )
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            bmg, atom_descriptors, _, _, _, _, _ = batch
            atom_embeddings = message_passing(bmg, atom_descriptors)
            chunks.append(
                aggregation(atom_embeddings, bmg.batch).detach().cpu().numpy()
            )
    values = np.concatenate(chunks, axis=0)
    if values.shape != (len(data_frame), 2048) or not np.isfinite(values).all():
        raise RuntimeError(f"Unexpected CheMeleon embedding matrix: {values.shape}")
    return {
        str(identity): values[index].copy()
        for index, identity in enumerate(data_frame[IDENTITY_COLUMN])
    }


def _pretrained_arrays(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    embeddings: Mapping[str, np.ndarray],
    descriptors: Mapping[str, np.ndarray],
    candidate: GraphCandidate,
) -> tuple[np.ndarray, np.ndarray]:
    train_values = np.stack([embeddings[value] for value in train[IDENTITY_COLUMN]])
    validation_values = np.stack(
        [embeddings[value] for value in validation[IDENTITY_COLUMN]]
    )
    if candidate.use_descriptors:
        raw_train = np.stack([descriptors[value] for value in train[IDENTITY_COLUMN]])
        raw_validation = np.stack(
            [descriptors[value] for value in validation[IDENTITY_COLUMN]]
        )
        train_desc, validation_desc = _fit_descriptor_transform(raw_train, raw_validation)
        train_values = np.concatenate([train_values, train_desc], axis=1)
        validation_values = np.concatenate([validation_values, validation_desc], axis=1)
    return train_values.astype(np.float32), validation_values.astype(np.float32)


def _pretrained_head(candidate: GraphCandidate, input_dim: int):
    import torch

    activation_types = {
        "relu": torch.nn.ReLU,
        "leakyrelu": torch.nn.LeakyReLU,
        "elu": torch.nn.ELU,
    }
    layers: list[torch.nn.Module] = []
    if candidate.normalization == "batch":
        layers.append(torch.nn.BatchNorm1d(input_dim))
    current = input_dim
    for _ in range(candidate.ffn_layers - 1):
        layers.extend(
            [
                torch.nn.Linear(current, candidate.ffn_hidden_dim),
                activation_types[candidate.activation](),
                torch.nn.Dropout(candidate.dropout),
            ]
        )
        current = candidate.ffn_hidden_dim
    layers.append(torch.nn.Linear(current, 1))
    model = torch.nn.Sequential(*layers)
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    return model


def _pretrained_inner_score(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    embeddings: Mapping[str, np.ndarray],
    descriptors: Mapping[str, np.ndarray],
    candidate: GraphCandidate,
    protocol: SearchProtocol,
    seed: int,
) -> tuple[float, int, int]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    _configure_torch(seed, protocol.torch_threads)
    train_x, validation_x = _pretrained_arrays(
        train,
        validation,
        embeddings=embeddings,
        descriptors=descriptors,
        candidate=candidate,
    )
    y_train = train[TARGET_COLUMN].to_numpy(dtype=float)
    y_validation = validation[TARGET_COLUMN].to_numpy(dtype=float)
    target_mean, target_scale = _target_scale(y_train)
    dataset = TensorDataset(
        torch.as_tensor(train_x),
        torch.as_tensor((y_train - target_mean) / target_scale, dtype=torch.float32),
    )
    loader = DataLoader(
        dataset,
        batch_size=_safe_training_batch_size(len(dataset), candidate.batch_size),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    validation_tensor = torch.as_tensor(validation_x)
    model = _pretrained_head(candidate, train_x.shape[1])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=candidate.learning_rate, weight_decay=candidate.weight_decay
    )
    best_rmse, best_epoch, stale = float("inf"), 1, 0
    for epoch in range(1, protocol.max_epochs + 1):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_x).reshape(-1)
            loss = torch.nn.functional.mse_loss(prediction, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            prediction = (
                model(validation_tensor).reshape(-1).cpu().numpy() * target_scale
                + target_mean
            )
        rmse = regression_metrics(y_validation, prediction)["rmse"]
        if best_rmse - rmse > protocol.min_delta:
            best_rmse, best_epoch, stale = float(rmse), epoch, 0
        else:
            stale += 1
        if stale >= protocol.patience:
            break
    parameters = sum(value.numel() for value in model.parameters())
    return best_rmse, best_epoch, int(parameters)


def _pretrained_refit_predict(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    embeddings: Mapping[str, np.ndarray],
    descriptors: Mapping[str, np.ndarray],
    candidate: GraphCandidate,
    protocol: SearchProtocol,
    seed: int,
    epochs: int,
) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    _configure_torch(seed, protocol.torch_threads)
    train_x, validation_x = _pretrained_arrays(
        train,
        validation,
        embeddings=embeddings,
        descriptors=descriptors,
        candidate=candidate,
    )
    y_train = train[TARGET_COLUMN].to_numpy(dtype=float)
    target_mean, target_scale = _target_scale(y_train)
    dataset = TensorDataset(
        torch.as_tensor(train_x),
        torch.as_tensor((y_train - target_mean) / target_scale, dtype=torch.float32),
    )
    loader = DataLoader(
        dataset,
        batch_size=_safe_training_batch_size(len(dataset), candidate.batch_size),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    model = _pretrained_head(candidate, train_x.shape[1])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=candidate.learning_rate, weight_decay=candidate.weight_decay
    )
    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_x).reshape(-1)
            loss = torch.nn.functional.mse_loss(prediction, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    model.eval()
    with torch.no_grad():
        return (
            model(torch.as_tensor(validation_x)).reshape(-1).cpu().numpy() * target_scale
            + target_mean
        )


def _make_chemprop_model(
    candidate: GraphCandidate,
    *,
    target_scaler: Any,
    descriptor_scaler: Any | None,
    checkpoint_path: Path,
):
    import torch
    from chemprop import models, nn
    from chemprop.schedulers import build_NoamLike_LRSched

    class AdamWMPNN(models.MPNN):
        def configure_optimizers(self):
            parameters = [value for value in self.parameters() if value.requires_grad]
            optimizer = torch.optim.AdamW(
                parameters,
                lr=self.init_lr,
                weight_decay=candidate.weight_decay,
            )
            if self.trainer.train_dataloader is None:
                self.trainer.estimated_stepping_batches
            steps = self.trainer.num_training_batches
            warmup_steps = self.warmup_epochs * steps
            cooldown_steps = max(1, self.trainer.max_epochs - self.warmup_epochs) * steps
            scheduler = build_NoamLike_LRSched(
                optimizer,
                warmup_steps,
                cooldown_steps,
                self.init_lr,
                self.max_lr,
                self.final_lr,
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    if candidate.family == "chemprop_chemeleon":
        if sha256_file(checkpoint_path) != CHEMELEON_SHA256:
            raise RuntimeError("CheMeleon checkpoint hash mismatch")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        message_passing = nn.BondMessagePassing(**payload["hyper_parameters"])
        message_passing.load_state_dict(payload["state_dict"], strict=True)
        message_passing.requires_grad_(candidate.encoder_training == "fine_tuned")
    else:
        message_passing = nn.BondMessagePassing(
            d_v=len(_chemprop_featurizer(candidate.featurization).atom_featurizer),
            d_h=candidate.hidden_dim,
            depth=candidate.depth,
            dropout=candidate.dropout,
            activation=candidate.activation,
        )
    if candidate.pooling == "sum":
        aggregation = nn.SumAggregation()
    elif candidate.pooling == "norm":
        aggregation = nn.NormAggregation(norm=100.0)
    else:
        aggregation = nn.MeanAggregation()
    descriptor_dim = 23 if candidate.use_descriptors else 0
    predictor = nn.RegressionFFN(
        input_dim=message_passing.output_dim + descriptor_dim,
        hidden_dim=candidate.ffn_hidden_dim,
        n_layers=candidate.ffn_layers,
        dropout=candidate.dropout,
        activation=candidate.activation,
        output_transform=nn.UnscaleTransform.from_standard_scaler(target_scaler),
    )
    descriptor_transform = (
        nn.ScaleTransform.from_standard_scaler(descriptor_scaler)
        if descriptor_scaler is not None
        else None
    )
    return AdamWMPNN(
        message_passing,
        aggregation,
        predictor,
        batch_norm=candidate.normalization == "batch",
        metrics=[nn.metrics.RMSE()],
        warmup_epochs=2,
        init_lr=candidate.learning_rate / 10,
        max_lr=candidate.learning_rate,
        final_lr=candidate.learning_rate / 10,
        X_d_transform=descriptor_transform,
    )


def _chemprop_inner_score(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    descriptors: Mapping[str, np.ndarray],
    candidate: GraphCandidate,
    protocol: SearchProtocol,
    checkpoint_path: Path,
    seed: int,
) -> tuple[float, int, int]:
    require_chemprop()
    import lightning.pytorch as pl
    import torch
    from chemprop import data
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    _configure_torch(seed, protocol.torch_threads)
    pl.seed_everything(seed, workers=True, verbose=False)
    train_dataset, validation_dataset, target_scaler, descriptor_scaler = (
        _chemprop_datasets_tuned(
            train, validation, descriptors=descriptors, candidate=candidate
        )
    )
    train_loader = data.build_dataloader(
        train_dataset,
        batch_size=_safe_training_batch_size(len(train_dataset), candidate.batch_size),
        num_workers=0,
        shuffle=True,
        seed=seed,
    )
    validation_loader = data.build_dataloader(
        validation_dataset,
        batch_size=len(validation_dataset),
        num_workers=0,
        shuffle=False,
    )
    model = _make_chemprop_model(
        candidate,
        target_scaler=target_scaler,
        descriptor_scaler=descriptor_scaler,
        checkpoint_path=checkpoint_path,
    )
    parameters = sum(value.numel() for value in model.parameters())
    with tempfile.TemporaryDirectory(prefix="gnn-tuned-") as directory:
        checkpoint = ModelCheckpoint(
            dirpath=directory,
            monitor="val/rmse",
            mode="min",
            save_top_k=1,
            filename="best-{epoch:03d}",
        )
        early_stop = EarlyStopping(
            monitor="val/rmse",
            mode="min",
            patience=protocol.patience,
            min_delta=protocol.min_delta / float(target_scaler.scale_[0]),
        )
        trainer = pl.Trainer(
            accelerator="cpu",
            devices=1,
            max_epochs=protocol.max_epochs,
            deterministic=True,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[checkpoint, early_stop],
            gradient_clip_val=5.0,
        )
        trainer.fit(model, train_loader, validation_loader)
        if not checkpoint.best_model_path or checkpoint.best_model_score is None:
            raise RuntimeError("Chemprop tuning produced no validation checkpoint")
        payload = torch.load(checkpoint.best_model_path, map_location="cpu", weights_only=False)
        epoch = int(payload["epoch"]) + 1
        rmse = float(checkpoint.best_model_score.detach().cpu()) * float(
            target_scaler.scale_[0]
        )
    return rmse, epoch, int(parameters)


def _chemprop_refit_predict(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    descriptors: Mapping[str, np.ndarray],
    candidate: GraphCandidate,
    protocol: SearchProtocol,
    checkpoint_path: Path,
    seed: int,
    epochs: int,
) -> np.ndarray:
    require_chemprop()
    import lightning.pytorch as pl
    from chemprop import data

    _configure_torch(seed, protocol.torch_threads)
    pl.seed_everything(seed, workers=True, verbose=False)
    train_dataset, validation_dataset, target_scaler, descriptor_scaler = (
        _chemprop_datasets_tuned(
            train, validation, descriptors=descriptors, candidate=candidate
        )
    )
    train_loader = data.build_dataloader(
        train_dataset,
        batch_size=_safe_training_batch_size(len(train_dataset), candidate.batch_size),
        num_workers=0,
        shuffle=True,
        seed=seed,
    )
    model = _make_chemprop_model(
        candidate,
        target_scaler=target_scaler,
        descriptor_scaler=descriptor_scaler,
        checkpoint_path=checkpoint_path,
    )
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=epochs,
        deterministic=True,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        gradient_clip_val=5.0,
    )
    trainer.fit(model, train_loader)
    # Chemprop validation runs with dataset values pre-scaled and its stored
    # transforms disabled. Prediction does the inverse: it expects raw values
    # and applies the training-fold transforms held by the model. Reusing the
    # scaled validation dataset would therefore scale X_d twice.
    validation_dataset.reset()
    validation_loader = data.build_dataloader(
        validation_dataset,
        batch_size=len(validation_dataset),
        num_workers=0,
        shuffle=False,
    )
    batches = trainer.predict(model, dataloaders=validation_loader)
    prediction = np.concatenate(
        [batch.detach().cpu().numpy().reshape(-1) for batch in batches]
    )
    if not np.isfinite(prediction).all():
        raise RuntimeError("Chemprop produced non-finite predictions")
    return prediction


def pooled_rmse(rows: Sequence[Mapping[str, Any]]) -> float:
    count = sum(int(row["n_validation"]) for row in rows)
    if count <= 0:
        raise ValueError("Cannot pool empty validation results")
    squared_error = sum(
        int(row["n_validation"]) * float(row["rmse"]) ** 2 for row in rows
    )
    return float(math.sqrt(squared_error / count))


def _trial_path(
    output_dir: Path, family: str, repeat: int, fold: int, candidate_id: str, seed: int
) -> Path:
    return (
        output_dir
        / "trials"
        / f"{family}__repeat-{repeat}__fold-{fold}__{candidate_id}__seed-{seed}.json"
    )


def _evaluate_candidate(
    candidate: GraphCandidate,
    *,
    outer_training: pd.DataFrame,
    inner_splits: Sequence[tuple[np.ndarray, np.ndarray]],
    graphs: Mapping[str, Any],
    pretrained_embeddings: Mapping[str, np.ndarray],
    descriptors: Mapping[str, np.ndarray],
    protocol: SearchProtocol,
    checkpoint_path: Path,
    output_dir: Path,
    repeat: int,
    fold: int,
    seed: int,
    contract_sha256: str,
) -> dict[str, Any]:
    path = _trial_path(output_dir, candidate.family, repeat, fold, candidate.candidate_id, seed)
    if path.is_file():
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("contract_sha256") != contract_sha256:
            raise RuntimeError(f"Stale tuned-GNN trial: {path.name}")
        expected_fields = {
            "family": candidate.family,
            "candidate_id": candidate.candidate_id,
            "candidate": candidate.as_dict(),
            "repeat": repeat,
            "fold": fold,
            "seed": seed,
            "temporal_test_consulted": False,
        }
        if any(result.get(key) != value for key, value in expected_fields.items()):
            raise RuntimeError(f"Misassigned tuned-GNN trial: {path.name}")
        if len(result.get("inner_folds", [])) != protocol.inner_folds:
            raise RuntimeError(f"Incomplete tuned-GNN trial: {path.name}")
        return result
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for inner_fold, (train_index, validation_index) in enumerate(inner_splits):
        train = outer_training.iloc[train_index].reset_index(drop=True)
        validation = outer_training.iloc[validation_index].reset_index(drop=True)
        fit_seed = int(seed + inner_fold * 10000)
        if candidate.family == "edge_aware_gine":
            rmse, epoch, parameters = _gine_inner_score(
                train,
                validation,
                graphs=graphs,
                descriptors=descriptors,
                candidate=candidate,
                protocol=protocol,
                seed=fit_seed,
            )
        elif candidate.family == "chemprop_dmpnn":
            rmse, epoch, parameters = _chemprop_inner_score(
                train,
                validation,
                descriptors=descriptors,
                candidate=candidate,
                protocol=protocol,
                checkpoint_path=checkpoint_path,
                seed=fit_seed,
            )
        else:
            rmse, epoch, parameters = _pretrained_inner_score(
                train,
                validation,
                embeddings=pretrained_embeddings,
                descriptors=descriptors,
                candidate=candidate,
                protocol=protocol,
                seed=fit_seed,
            )
        rows.append(
            {
                "inner_fold": inner_fold,
                "n_train": len(train),
                "n_validation": len(validation),
                "rmse": rmse,
                "selected_epoch": epoch,
                "parameter_count": parameters,
            }
        )
    result = {
        "schema_version": SCHEMA_VERSION,
        "contract_sha256": contract_sha256,
        "family": candidate.family,
        "candidate_id": candidate.candidate_id,
        "candidate": candidate.as_dict(),
        "repeat": repeat,
        "fold": fold,
        "seed": seed,
        "inner_folds": rows,
        "pooled_inner_rmse": pooled_rmse(rows),
        "median_epoch": int(np.median([row["selected_epoch"] for row in rows])),
        "runtime_seconds": float(time.perf_counter() - started),
        "temporal_test_consulted": False,
    }
    atomic_write_json(path, result)
    return result


def run_outer_fold_family(
    development: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    family: str,
    repeat: int,
    fold: int,
    protocol: SearchProtocol,
    output_dir: Path,
    checkpoint_path: Path,
    contract_sha256: str,
) -> None:
    """Tune and assess one family in one immutable outer fold."""

    output_dir.mkdir(parents=True, exist_ok=True)
    normalized = _normalize_assignments(assignments, identity_col=IDENTITY_COLUMN)
    assignment = normalized.loc[
        normalized["repeat"].eq(repeat) & normalized["fold"].eq(fold)
    ]
    train_ids, validation_ids = _validate_outer_fold(
        development,
        assignment,
        identity_col=IDENTITY_COLUMN,
        scaffold_col=SCAFFOLD_COLUMN,
    )
    lookup = development.set_index(IDENTITY_COLUMN, drop=False)
    outer_training = lookup.loc[train_ids].reset_index(drop=True)
    outer_validation = lookup.loc[validation_ids].reset_index(drop=True)
    descriptors, _ = _descriptor_map(development)
    graphs = _precompute_graphs(development) if family == "edge_aware_gine" else {}
    pretrained_embeddings = (
        _precompute_chemeleon_embeddings(
            development,
            checkpoint_path=checkpoint_path,
            torch_threads=protocol.torch_threads,
        )
        if family == "chemprop_chemeleon"
        else {}
    )
    inner_splits = make_shared_inner_splits(
        outer_training,
        seed=9101 + repeat * 100 + fold,
        n_splits=protocol.inner_folds,
    )
    candidates = generate_candidates(family, protocol)
    screening: list[dict[str, Any]] = []
    first_seed = protocol.rerank_seeds[0]
    for candidate in candidates:
        screening.append(
            _evaluate_candidate(
                candidate,
                outer_training=outer_training,
                inner_splits=inner_splits,
                graphs=graphs,
                pretrained_embeddings=pretrained_embeddings,
                descriptors=descriptors,
                protocol=protocol,
                checkpoint_path=checkpoint_path,
                output_dir=output_dir,
                repeat=repeat,
                fold=fold,
                seed=first_seed,
                contract_sha256=contract_sha256,
            )
        )
    shortlist = sorted(
        screening, key=lambda row: (row["pooled_inner_rmse"], row["candidate_id"])
    )[: protocol.rerank_top]
    reranked: list[dict[str, Any]] = []
    candidate_lookup = {candidate.candidate_id: candidate for candidate in candidates}
    for item in shortlist:
        candidate = candidate_lookup[item["candidate_id"]]
        seed_results = [item]
        for seed in protocol.rerank_seeds[1:]:
            seed_results.append(
                _evaluate_candidate(
                    candidate,
                    outer_training=outer_training,
                    inner_splits=inner_splits,
                    graphs=graphs,
                    pretrained_embeddings=pretrained_embeddings,
                    descriptors=descriptors,
                    protocol=protocol,
                    checkpoint_path=checkpoint_path,
                    output_dir=output_dir,
                    repeat=repeat,
                    fold=fold,
                    seed=seed,
                    contract_sha256=contract_sha256,
                )
            )
        reranked.append(
            {
                "candidate_id": candidate.candidate_id,
                "seed_mean_inner_rmse": float(
                    np.mean([row["pooled_inner_rmse"] for row in seed_results])
                ),
                "seed_sd_inner_rmse": float(
                    np.std([row["pooled_inner_rmse"] for row in seed_results], ddof=0)
                ),
                "seed_results": seed_results,
            }
        )
    selected = min(
        reranked, key=lambda row: (row["seed_mean_inner_rmse"], row["candidate_id"])
    )
    selected_candidate = candidate_lookup[selected["candidate_id"]]
    epochs_by_seed = {
        int(row["seed"]): int(row["median_epoch"])
        for row in selected["seed_results"]
    }
    predictions: list[pd.DataFrame] = []
    for seed_index, seed in enumerate(protocol.refit_seeds):
        epoch_seed = protocol.rerank_seeds[seed_index % len(protocol.rerank_seeds)]
        epochs = epochs_by_seed[epoch_seed]
        if family == "edge_aware_gine":
            values = _gine_refit_predict(
                outer_training,
                outer_validation,
                graphs=graphs,
                descriptors=descriptors,
                candidate=selected_candidate,
                protocol=protocol,
                seed=seed,
                epochs=epochs,
            )
        elif family == "chemprop_dmpnn":
            values = _chemprop_refit_predict(
                outer_training,
                outer_validation,
                descriptors=descriptors,
                candidate=selected_candidate,
                protocol=protocol,
                checkpoint_path=checkpoint_path,
                seed=seed,
                epochs=epochs,
            )
        else:
            values = _pretrained_refit_predict(
                outer_training,
                outer_validation,
                embeddings=pretrained_embeddings,
                descriptors=descriptors,
                candidate=selected_candidate,
                protocol=protocol,
                seed=seed,
                epochs=epochs,
            )
        predictions.append(
            pd.DataFrame(
                {
                    "family": family,
                    "candidate_id": selected_candidate.candidate_id,
                    "repeat": repeat,
                    "fold": fold,
                    "seed": seed,
                    IDENTITY_COLUMN: outer_validation[IDENTITY_COLUMN].astype(str),
                    SCAFFOLD_COLUMN: outer_validation[SCAFFOLD_COLUMN].astype(str),
                    "y_true": outer_validation[TARGET_COLUMN].astype(float),
                    "y_pred": values.astype(float),
                    "refit_epochs": epochs,
                }
            )
        )
    frame = pd.concat(predictions, ignore_index=True).sort_values(
        ["seed", IDENTITY_COLUMN], kind="mergesort"
    )
    stem = f"{family}__repeat-{repeat}__fold-{fold}"
    prediction_path = output_dir / "outer_predictions" / f"{stem}.csv"
    atomic_write_text(
        prediction_path,
        frame.to_csv(index=False, lineterminator="\n", float_format="%.12g"),
    )
    selection = {
        "schema_version": SCHEMA_VERSION,
        "contract_sha256": contract_sha256,
        "family": family,
        "repeat": repeat,
        "fold": fold,
        "selected_candidate": selected_candidate.as_dict(),
        "selected_candidate_id": selected_candidate.candidate_id,
        "screening": [
            {
                "candidate_id": row["candidate_id"],
                "pooled_inner_rmse": row["pooled_inner_rmse"],
            }
            for row in screening
        ],
        "reranking": reranked,
        "outer_prediction_sha256": sha256_file(prediction_path),
        "n_outer_train": len(outer_training),
        "n_outer_validation": len(outer_validation),
        "temporal_test_consulted": False,
    }
    atomic_write_json(output_dir / "selections" / f"{stem}.json", selection)


def aggregate_results(
    development: pd.DataFrame,
    *,
    output_dir: Path,
    protocol: SearchProtocol,
    contract_sha256: str,
    source_hashes: Mapping[str, str],
    search_config_sha256: str,
    checkpoint_sha256: str,
    reference_predictions_path: Path,
) -> dict[str, Any]:
    """Validate all expected fold artifacts and build publication summaries."""

    selection_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for family in FAMILIES:
        for repeat in range(3):
            for fold in range(5):
                stem = f"{family}__repeat-{repeat}__fold-{fold}"
                selection_path = output_dir / "selections" / f"{stem}.json"
                prediction_path = output_dir / "outer_predictions" / f"{stem}.csv"
                if not selection_path.is_file() or not prediction_path.is_file():
                    raise FileNotFoundError(f"Missing tuned-GNN fold result: {stem}")
                selection = json.loads(selection_path.read_text(encoding="utf-8"))
                if selection.get("contract_sha256") != contract_sha256:
                    raise RuntimeError(f"Stale tuned-GNN selection: {stem}")
                if selection.get("outer_prediction_sha256") != sha256_file(prediction_path):
                    raise RuntimeError(f"Tampered tuned-GNN predictions: {stem}")
                candidate = selection["selected_candidate"]
                selection_rows.append(
                    {
                        "family": family,
                        "repeat": repeat,
                        "fold": fold,
                        "candidate_id": selection["selected_candidate_id"],
                        "inner_rmse_mean": min(
                            row["seed_mean_inner_rmse"] for row in selection["reranking"]
                        ),
                        **candidate,
                    }
                )
                prediction_frames.append(pd.read_csv(prediction_path))
    trial_rows: list[dict[str, Any]] = []
    trial_paths = sorted((output_dir / "trials").glob("*.json"))
    expected_trials = len(FAMILIES) * 3 * 5 * (
        protocol.candidate_budget
        + protocol.rerank_top * (len(protocol.rerank_seeds) - 1)
    )
    if len(trial_paths) != expected_trials:
        raise RuntimeError(
            f"Expected {expected_trials} tuned-GNN trials, found {len(trial_paths)}"
        )
    for path in trial_paths:
        trial = json.loads(path.read_text(encoding="utf-8"))
        if trial.get("contract_sha256") != contract_sha256 or trial.get(
            "temporal_test_consulted"
        ) is not False:
            raise RuntimeError(f"Invalid tuned-GNN trial contract: {path.name}")
        if len(trial.get("inner_folds", [])) != protocol.inner_folds:
            raise RuntimeError(f"Incomplete tuned-GNN inner folds: {path.name}")
        for inner in trial["inner_folds"]:
            trial_rows.append(
                {
                    "family": trial["family"],
                    "candidate_id": trial["candidate_id"],
                    "repeat": int(trial["repeat"]),
                    "fold": int(trial["fold"]),
                    "seed": int(trial["seed"]),
                    "inner_fold": int(inner["inner_fold"]),
                    "n_train": int(inner["n_train"]),
                    "n_validation": int(inner["n_validation"]),
                    "rmse": float(inner["rmse"]),
                    "selected_epoch": int(inner["selected_epoch"]),
                    "parameter_count": int(inner["parameter_count"]),
                    "pooled_inner_rmse": float(trial["pooled_inner_rmse"]),
                    "runtime_seconds": float(trial["runtime_seconds"]),
                    "candidate_json": json.dumps(
                        trial["candidate"], sort_keys=True, separators=(",", ":")
                    ),
                }
            )
    if len(trial_rows) != expected_trials * protocol.inner_folds:
        raise RuntimeError("Tuned-GNN inner trial inventory is incomplete")
    seed_predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["family", "repeat", "fold", "seed", IDENTITY_COLUMN], kind="mergesort"
    )
    expected = len(development) * 3 * len(protocol.refit_seeds) * len(FAMILIES)
    if len(seed_predictions) != expected:
        raise RuntimeError(f"Expected {expected} seed OOF rows, found {len(seed_predictions)}")
    ensemble = (
        seed_predictions.groupby(
            ["family", "repeat", "fold", IDENTITY_COLUMN, SCAFFOLD_COLUMN, "y_true"],
            as_index=False,
            sort=True,
        )["y_pred"]
        .agg(y_pred="mean", y_pred_seed_sd=lambda values: float(np.std(values, ddof=0)))
    )
    repeat_rows: list[dict[str, Any]] = []
    seed_metric_rows: list[dict[str, Any]] = []
    for (family, repeat, seed), group in seed_predictions.groupby(
        ["family", "repeat", "seed"], sort=True
    ):
        seed_metric_rows.append(
            {
                "family": family,
                "repeat": int(repeat),
                "seed": int(seed),
                "n_oof": len(group),
                **regression_metrics(group["y_true"], group["y_pred"]),
            }
        )
    for (family, repeat), group in ensemble.groupby(["family", "repeat"], sort=True):
        if len(group) != len(development) or group[IDENTITY_COLUMN].duplicated().any():
            raise RuntimeError("Tuned-GNN ensemble OOF inventory is incomplete")
        repeat_rows.append(
            {
                "family": family,
                "repeat": int(repeat),
                "n_oof": len(group),
                **regression_metrics(group["y_true"], group["y_pred"]),
                "mean_seed_prediction_sd": float(group["y_pred_seed_sd"].mean()),
            }
        )
    repeat_metrics = pd.DataFrame(repeat_rows)
    model_rows: list[dict[str, Any]] = []
    for family, group in repeat_metrics.groupby("family", sort=True):
        model_rows.append(
            {
                "family": family,
                "n_repeats": len(group),
                "candidate_budget_per_outer_fold": protocol.candidate_budget,
                "outer_refit_seeds": len(protocol.refit_seeds),
                "rmse_mean": float(group["rmse"].mean()),
                "rmse_sd": float(group["rmse"].std(ddof=0)),
                "mae_mean": float(group["mae"].mean()),
                "mae_sd": float(group["mae"].std(ddof=0)),
                "r2_mean": float(group["r2"].mean()),
                "r2_sd": float(group["r2"].std(ddof=0)),
                "mean_seed_prediction_sd": float(group["mean_seed_prediction_sd"].mean()),
            }
        )
    model_summary = pd.DataFrame(model_rows).sort_values(
        ["rmse_mean", "family"], kind="mergesort"
    )
    reference = pd.read_csv(reference_predictions_path)
    reference = reference.loc[reference["model"].eq("ridge_combined")].copy()
    if len(reference) != len(development):
        raise RuntimeError("Ridge reference OOF inventory is incomplete")
    tuned_mean = (
        ensemble.groupby(
            ["family", IDENTITY_COLUMN, SCAFFOLD_COLUMN, "y_true"],
            as_index=False,
            sort=True,
        )["y_pred"]
        .mean()
    )
    bootstrap_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(44001)
    for family, family_predictions in tuned_mean.groupby("family", sort=True):
        paired = family_predictions.merge(
            reference[[IDENTITY_COLUMN, "y_true", "y_pred"]].rename(
                columns={"y_true": "reference_y_true", "y_pred": "reference_y_pred"}
            ),
            on=IDENTITY_COLUMN,
            how="inner",
            validate="one_to_one",
        )
        if len(paired) != len(development) or not np.allclose(
            paired["y_true"], paired["reference_y_true"], atol=1e-10
        ):
            raise RuntimeError("Tuned and Ridge OOF predictions are not aligned")
        clusters = sorted(paired[SCAFFOLD_COLUMN].astype(str).unique())
        cluster_stats: list[tuple[int, float, float, float, float]] = []
        for cluster in clusters:
            group = paired.loc[paired[SCAFFOLD_COLUMN].astype(str).eq(cluster)]
            tuned_error = group["y_pred"].to_numpy() - group["y_true"].to_numpy()
            reference_error = (
                group["reference_y_pred"].to_numpy() - group["y_true"].to_numpy()
            )
            cluster_stats.append(
                (
                    len(group),
                    float(np.square(tuned_error).sum()),
                    float(np.square(reference_error).sum()),
                    float(np.abs(tuned_error).sum()),
                    float(np.abs(reference_error).sum()),
                )
            )
        stats = np.asarray(cluster_stats, dtype=float)
        counts = rng.multinomial(
            len(clusters), np.full(len(clusters), 1.0 / len(clusters)), size=10000
        )
        n_boot = counts @ stats[:, 0]
        tuned_rmse = np.sqrt((counts @ stats[:, 1]) / n_boot)
        reference_rmse = np.sqrt((counts @ stats[:, 2]) / n_boot)
        tuned_mae = (counts @ stats[:, 3]) / n_boot
        reference_mae = (counts @ stats[:, 4]) / n_boot
        point_tuned = regression_metrics(paired["y_true"], paired["y_pred"])
        point_reference = regression_metrics(
            paired["y_true"], paired["reference_y_pred"]
        )
        for metric, samples in {
            "RMSE": tuned_rmse - reference_rmse,
            "MAE": tuned_mae - reference_mae,
        }.items():
            key = metric.lower()
            bootstrap_rows.append(
                {
                    "family": family,
                    "reference_model": "ridge_combined",
                    "metric": metric,
                    "point_difference": point_tuned[key] - point_reference[key],
                    "ci_lower": float(np.quantile(samples, 0.025)),
                    "ci_upper": float(np.quantile(samples, 0.975)),
                    "bootstrap_replicates": 10000,
                    "cluster_count": len(clusters),
                }
            )
    paired_bootstrap = pd.DataFrame(bootstrap_rows)
    outputs = {
        "inner_trial_metrics.csv": pd.DataFrame(trial_rows),
        "selected_hyperparameters.csv": pd.DataFrame(selection_rows),
        "seed_oof_predictions.csv": seed_predictions,
        "ensemble_oof_predictions.csv": ensemble,
        "seed_metrics.csv": pd.DataFrame(seed_metric_rows),
        "repeat_metrics.csv": repeat_metrics,
        "model_summary.csv": model_summary,
        "paired_scaffold_bootstrap.csv": paired_bootstrap,
    }
    output_hashes: dict[str, str] = {}
    for filename, frame in outputs.items():
        path = output_dir / filename
        atomic_write_text(
            path, frame.to_csv(index=False, lineterminator="\n", float_format="%.12g")
        )
        output_hashes[filename] = sha256_file(path)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "post_hoc_development_only",
        "contract_sha256": contract_sha256,
        "source_hashes": dict(source_hashes),
        "search_config_sha256": search_config_sha256,
        "pretrained_checkpoint_sha256": checkpoint_sha256,
        "reference_predictions_sha256": sha256_file(reference_predictions_path),
        "families": list(FAMILIES),
        "candidate_budget_per_family_per_outer_fold": protocol.candidate_budget,
        "trial_count": expected_trials,
        "inner_fit_count": len(trial_rows),
        "rerank_top_candidates": protocol.rerank_top,
        "rerank_seeds": list(protocol.rerank_seeds),
        "outer_refit_seeds": list(protocol.refit_seeds),
        "output_hashes": output_hashes,
        "temporal_test_consulted": False,
    }
    atomic_write_json(output_dir / "run_manifest.json", manifest)
    atomic_write_text(output_dir / "RUN_COMPLETE", contract_sha256 + "\n")
    return manifest
