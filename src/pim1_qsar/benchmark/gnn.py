"""Nested, resumable GINE and Chemprop D-MPNN benchmarks.

Every outer validation partition is used exactly once for scoring.  Epoch
selection happens on scaffold-disjoint inner folds drawn exclusively from the
outer-training partition; the median selected epoch is then used for a fresh
fit on the complete outer-training partition without validation monitoring.
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
from pim1_qsar.features import MolecularFeatures, features_from_frame
from pim1_qsar.models.chemprop_adapter import require_chemprop
from pim1_qsar.models.gine import GINEConfig, GINERegressor, require_gnn_dependencies
from pim1_qsar.models.gnn_features import MoleculeGraphFeatures, featurize_smiles, to_pyg_data


GNN_BENCHMARK_SCHEMA_VERSION = "gnn_nested_v2"
GRAPH_VARIANT = "graph"
GRAPH_DESCRIPTOR_VARIANT = "graph_plus_rdkit2d"
SUPPORTED_BACKENDS = ("edge_aware_gine", "chemprop_dmpnn")
SUPPORTED_VARIANTS = (GRAPH_VARIANT, GRAPH_DESCRIPTOR_VARIANT)


@dataclass(frozen=True)
class GNNTrainingConfig:
    hidden_dim: int = 128
    depth: int = 3
    ffn_hidden_dim: int = 128
    ffn_layers: int = 2
    dropout: float = 0.1
    batch_size: int = 64
    max_epochs: int = 80
    patience: int = 10
    min_delta: float = 0.001
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip: float = 5.0
    inner_folds: int = 3
    seed_base: int = 5101
    torch_threads: int = 4

    def __post_init__(self) -> None:
        for name in (
            "hidden_dim",
            "depth",
            "ffn_hidden_dim",
            "ffn_layers",
            "batch_size",
            "max_epochs",
            "patience",
            "inner_folds",
            "torch_threads",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.min_delta < 0:
            raise ValueError("min_delta must be non-negative")
        if min(self.learning_rate, self.grad_clip) <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer configuration")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GNNBenchmarkResult:
    predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    repeat_metrics: pd.DataFrame
    model_summary: pd.DataFrame
    run_manifest: dict[str, Any]


def model_name(backend: str, variant: str) -> str:
    if backend not in SUPPORTED_BACKENDS or variant not in SUPPORTED_VARIANTS:
        raise ValueError(f"Unsupported GNN backend/variant: {backend}/{variant}")
    return backend + ("_graph" if variant == GRAPH_VARIANT else "_graph_plus_rdkit2d")


def _fit_descriptor_transform(
    training: np.ndarray,
    *others: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Median-impute and standardize descriptors using training rows only."""

    train = np.asarray(training, dtype=float)
    if train.ndim != 2 or not len(train):
        raise ValueError("Descriptor training matrix must be non-empty and two-dimensional")
    medians = np.nanmedian(train, axis=0)
    if not np.isfinite(medians).all():
        raise ValueError("A descriptor is missing for every training molecule")

    def impute(matrix: np.ndarray) -> np.ndarray:
        values = np.asarray(matrix, dtype=float).copy()
        if values.ndim != 2 or values.shape[1] != train.shape[1]:
            raise ValueError("Descriptor matrices are not aligned")
        rows, columns = np.where(~np.isfinite(values))
        values[rows, columns] = medians[columns]
        return values

    imputed_train = impute(train)
    means = imputed_train.mean(axis=0)
    scales = imputed_train.std(axis=0, ddof=0)
    scales[scales == 0] = 1.0
    return tuple((impute(matrix) - means) / scales for matrix in (train, *others))


def _target_scale(training: np.ndarray) -> tuple[float, float]:
    values = np.asarray(training, dtype=float)
    mean = float(values.mean())
    scale = float(values.std(ddof=0))
    return mean, scale if scale > 0 else 1.0


def _precompute_graphs(data: pd.DataFrame) -> dict[str, MoleculeGraphFeatures]:
    graphs: dict[str, MoleculeGraphFeatures] = {}
    for row in data[[IDENTITY_COLUMN, SMILES_COLUMN]].itertuples(index=False, name=None):
        identity, smiles = str(row[0]), str(row[1])
        graphs[identity] = featurize_smiles(smiles)
    if len(graphs) != len(data):
        raise ValueError("Graph featurization lost or duplicated a development identity")
    return graphs


def _pyg_dataset(
    identities: Sequence[str],
    y_scaled: np.ndarray,
    graphs: Mapping[str, MoleculeGraphFeatures],
    descriptors: np.ndarray | None,
):
    if len(identities) != len(y_scaled):
        raise ValueError("GINE identities and targets are not aligned")
    return [
        to_pyg_data(
            graphs[identity],
            target=float(y_scaled[index]),
            descriptors=None if descriptors is None else descriptors[index],
            row_id=identity,
        )
        for index, identity in enumerate(identities)
    ]


def _configure_torch(seed: int, threads: int) -> None:
    require_gnn_dependencies()
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def _gine_model(config: GNNTrainingConfig, descriptor_dim: int):
    return GINERegressor(
        GINEConfig(
            descriptor_dim=descriptor_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.depth,
            ffn_hidden_dim=config.ffn_hidden_dim,
            ffn_num_layers=config.ffn_layers,
            dropout=config.dropout,
            pooling="mean",
            residual=True,
            batch_norm=True,
        )
    )


def _gine_early_stop_epoch(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    graphs: Mapping[str, MoleculeGraphFeatures],
    descriptor_map: Mapping[str, np.ndarray],
    use_descriptors: bool,
    config: GNNTrainingConfig,
    seed: int,
) -> int:
    import torch
    from torch_geometric.loader import DataLoader

    _configure_torch(seed, config.torch_threads)
    y_train = train[TARGET_COLUMN].to_numpy(dtype=float)
    y_validation = validation[TARGET_COLUMN].to_numpy(dtype=float)
    target_mean, target_scale = _target_scale(y_train)
    train_desc: np.ndarray | None = None
    validation_desc: np.ndarray | None = None
    if use_descriptors:
        raw_train = np.stack([descriptor_map[value] for value in train[IDENTITY_COLUMN]])
        raw_validation = np.stack(
            [descriptor_map[value] for value in validation[IDENTITY_COLUMN]]
        )
        train_desc, validation_desc = _fit_descriptor_transform(raw_train, raw_validation)
    train_graphs = _pyg_dataset(
        train[IDENTITY_COLUMN].astype(str).tolist(),
        (y_train - target_mean) / target_scale,
        graphs,
        train_desc,
    )
    validation_graphs = _pyg_dataset(
        validation[IDENTITY_COLUMN].astype(str).tolist(),
        (y_validation - target_mean) / target_scale,
        graphs,
        validation_desc,
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_graphs,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_graphs,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    model = _gine_model(config, 23 if use_descriptors else 0)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best_epoch = 1
    best_rmse = float("inf")
    stale_epochs = 0
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch)
            loss = torch.nn.functional.mse_loss(prediction, batch.y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
        model.eval()
        predictions: list[np.ndarray] = []
        with torch.no_grad():
            for batch in validation_loader:
                prediction = model(batch).detach().cpu().numpy()
                predictions.append(prediction * target_scale + target_mean)
        observed_rmse = regression_metrics(
            y_validation, np.concatenate(predictions)
        )["rmse"]
        if best_rmse - observed_rmse > config.min_delta:
            best_rmse = observed_rmse
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= config.patience:
            break
    return best_epoch


def _gine_refit_predict(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    graphs: Mapping[str, MoleculeGraphFeatures],
    descriptor_map: Mapping[str, np.ndarray],
    use_descriptors: bool,
    config: GNNTrainingConfig,
    seed: int,
    epochs: int,
) -> np.ndarray:
    import torch
    from torch_geometric.loader import DataLoader

    _configure_torch(seed, config.torch_threads)
    y_train = train[TARGET_COLUMN].to_numpy(dtype=float)
    target_mean, target_scale = _target_scale(y_train)
    train_desc: np.ndarray | None = None
    validation_desc: np.ndarray | None = None
    if use_descriptors:
        raw_train = np.stack([descriptor_map[value] for value in train[IDENTITY_COLUMN]])
        raw_validation = np.stack(
            [descriptor_map[value] for value in validation[IDENTITY_COLUMN]]
        )
        train_desc, validation_desc = _fit_descriptor_transform(raw_train, raw_validation)
    train_graphs = _pyg_dataset(
        train[IDENTITY_COLUMN].astype(str).tolist(),
        (y_train - target_mean) / target_scale,
        graphs,
        train_desc,
    )
    validation_graphs = _pyg_dataset(
        validation[IDENTITY_COLUMN].astype(str).tolist(),
        np.zeros(len(validation), dtype=float),
        graphs,
        validation_desc,
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_graphs,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_graphs,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    model = _gine_model(config, 23 if use_descriptors else 0)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch)
            loss = torch.nn.functional.mse_loss(prediction, batch.y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
    model.eval()
    output: list[np.ndarray] = []
    with torch.no_grad():
        for batch in validation_loader:
            scaled = model(batch).detach().cpu().numpy()
            output.append(scaled * target_scale + target_mean)
    return np.concatenate(output)


def _chemprop_datapoints(
    frame: pd.DataFrame,
    descriptors: Mapping[str, np.ndarray] | None,
):
    from chemprop import data as chemprop_data

    points = []
    for row in frame.itertuples(index=False):
        values = row._asdict()
        identity = str(values[IDENTITY_COLUMN])
        x_d = None if descriptors is None else np.asarray(descriptors[identity], dtype=float)
        points.append(
            chemprop_data.MoleculeDatapoint.from_smi(
                str(values[SMILES_COLUMN]),
                y=np.asarray([float(values[TARGET_COLUMN])], dtype=float),
                x_d=x_d,
                name=identity,
            )
        )
    return points


def _make_chemprop_model(
    *,
    config: GNNTrainingConfig,
    target_scaler: Any,
    descriptor_scaler: Any | None,
    descriptor_dim: int,
):
    from chemprop import models, nn

    message_passing = nn.BondMessagePassing(
        d_h=config.hidden_dim,
        depth=config.depth,
        dropout=config.dropout,
    )
    aggregation = nn.MeanAggregation()
    predictor = nn.RegressionFFN(
        input_dim=config.hidden_dim + descriptor_dim,
        hidden_dim=config.ffn_hidden_dim,
        n_layers=config.ffn_layers,
        dropout=config.dropout,
        output_transform=nn.UnscaleTransform.from_standard_scaler(target_scaler),
    )
    descriptor_transform = (
        nn.ScaleTransform.from_standard_scaler(descriptor_scaler)
        if descriptor_scaler is not None
        else None
    )
    return models.MPNN(
        message_passing,
        aggregation,
        predictor,
        batch_norm=True,
        warmup_epochs=min(2, config.max_epochs),
        init_lr=config.learning_rate / 3,
        max_lr=config.learning_rate,
        final_lr=config.learning_rate / 3,
        X_d_transform=descriptor_transform,
    )


def _chemprop_datasets(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    descriptor_map: Mapping[str, np.ndarray],
    use_descriptors: bool,
):
    from chemprop import data, featurizers

    descriptor_source = descriptor_map if use_descriptors else None
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    train_dataset = data.MoleculeDataset(
        _chemprop_datapoints(train, descriptor_source), featurizer
    )
    validation_dataset = data.MoleculeDataset(
        _chemprop_datapoints(validation, descriptor_source), featurizer
    )
    target_scaler = train_dataset.normalize_targets()
    descriptor_scaler = train_dataset.normalize_inputs("X_d") if use_descriptors else None
    return train_dataset, validation_dataset, target_scaler, descriptor_scaler


def _chemprop_early_stop_epoch(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    descriptor_map: Mapping[str, np.ndarray],
    use_descriptors: bool,
    config: GNNTrainingConfig,
    seed: int,
) -> int:
    require_chemprop()
    import lightning.pytorch as pl
    import torch
    from chemprop import data
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    _configure_torch(seed, config.torch_threads)
    pl.seed_everything(seed, workers=True, verbose=False)
    train_dataset, validation_dataset, target_scaler, descriptor_scaler = (
        _chemprop_datasets(
            train,
            validation,
            descriptor_map=descriptor_map,
            use_descriptors=use_descriptors,
        )
    )
    train_loader = data.build_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=0,
        shuffle=True,
        seed=seed,
    )
    validation_loader = data.build_dataloader(
        validation_dataset,
        batch_size=config.batch_size,
        num_workers=0,
        shuffle=False,
    )
    model = _make_chemprop_model(
        config=config,
        target_scaler=target_scaler,
        descriptor_scaler=descriptor_scaler,
        descriptor_dim=23 if use_descriptors else 0,
    )
    with tempfile.TemporaryDirectory(prefix="chemprop-early-stop-") as directory:
        checkpoint = ModelCheckpoint(
            dirpath=directory,
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            filename="best-{epoch:03d}",
        )
        early_stop = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=config.patience,
            min_delta=config.min_delta,
        )
        trainer = pl.Trainer(
            accelerator="cpu",
            devices=1,
            max_epochs=config.max_epochs,
            deterministic=True,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[checkpoint, early_stop],
            gradient_clip_val=config.grad_clip,
        )
        trainer.fit(model, train_loader, validation_loader)
        if not checkpoint.best_model_path:
            raise RuntimeError("Chemprop early stopping produced no checkpoint")
        payload = torch.load(
            checkpoint.best_model_path,
            map_location="cpu",
            weights_only=False,
        )
        return int(payload["epoch"]) + 1


def _chemprop_refit_predict(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    descriptor_map: Mapping[str, np.ndarray],
    use_descriptors: bool,
    config: GNNTrainingConfig,
    seed: int,
    epochs: int,
) -> np.ndarray:
    require_chemprop()
    import lightning.pytorch as pl
    import torch
    from chemprop import data

    _configure_torch(seed, config.torch_threads)
    pl.seed_everything(seed, workers=True, verbose=False)
    train_dataset, validation_dataset, target_scaler, descriptor_scaler = (
        _chemprop_datasets(
            train,
            validation,
            descriptor_map=descriptor_map,
            use_descriptors=use_descriptors,
        )
    )
    train_loader = data.build_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=0,
        shuffle=True,
        seed=seed,
    )
    validation_loader = data.build_dataloader(
        validation_dataset,
        batch_size=config.batch_size,
        num_workers=0,
        shuffle=False,
    )
    model = _make_chemprop_model(
        config=config,
        target_scaler=target_scaler,
        descriptor_scaler=descriptor_scaler,
        descriptor_dim=23 if use_descriptors else 0,
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
        gradient_clip_val=config.grad_clip,
    )
    trainer.fit(model, train_loader)
    batches = trainer.predict(model, dataloaders=validation_loader)
    output = [batch.detach().cpu().numpy().reshape(-1) for batch in batches]
    prediction = np.concatenate(output)
    if not np.isfinite(prediction).all():
        raise RuntimeError("Chemprop produced non-finite predictions")
    return prediction


def _checkpoint_paths(output_dir: Path, model: str, repeat: int, fold: int) -> tuple[Path, Path]:
    stem = f"{model}__repeat-{repeat}__fold-{fold}"
    checkpoint_dir = output_dir / "checkpoints"
    return checkpoint_dir / f"{stem}.csv", checkpoint_dir / f"{stem}.json"


def _load_checkpoint(
    predictions_path: Path,
    manifest_path: Path,
    *,
    contract_sha256: str,
) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    if not predictions_path.exists() and not manifest_path.exists():
        return None
    if not predictions_path.exists() or not manifest_path.exists():
        raise RuntimeError(f"Incomplete GNN checkpoint: {manifest_path.stem}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("contract_sha256") != contract_sha256:
        raise RuntimeError(f"Stale GNN checkpoint contract: {manifest_path.name}")
    if manifest.get("predictions_sha256") != sha256_file(predictions_path):
        raise RuntimeError(f"Tampered GNN checkpoint predictions: {predictions_path.name}")
    return pd.read_csv(predictions_path), manifest


def run_gnn_nested(
    development: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    output_dir: Path,
    contract_sha256: str,
    backends: Sequence[str] = SUPPORTED_BACKENDS,
    variants: Sequence[str] = SUPPORTED_VARIANTS,
    config: GNNTrainingConfig = GNNTrainingConfig(),
    smoke: bool = False,
) -> GNNBenchmarkResult:
    """Run or resume GNN outer folds without loading the temporal-test table."""

    required = {IDENTITY_COLUMN, SMILES_COLUMN, SCAFFOLD_COLUMN, TARGET_COLUMN}
    missing = sorted(required - set(development.columns))
    if missing:
        raise ValueError(f"Development data is missing: {', '.join(missing)}")
    if len(contract_sha256) != 64:
        raise ValueError("contract_sha256 must be a full SHA-256 digest")
    data = development.copy()
    data[IDENTITY_COLUMN] = data[IDENTITY_COLUMN].astype(str)
    if data[IDENTITY_COLUMN].duplicated().any():
        raise ValueError("Development identities must be unique")
    data[TARGET_COLUMN] = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")
    if not np.isfinite(data[TARGET_COLUMN].to_numpy(dtype=float)).all():
        raise ValueError("Development targets must be finite")
    data = data.sort_values(IDENTITY_COLUMN, kind="mergesort").reset_index(drop=True)
    normalized_assignments = _normalize_assignments(
        assignments, identity_col=IDENTITY_COLUMN
    )
    for backend in backends:
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(f"Unsupported backend: {backend}")
    for variant in variants:
        if variant not in SUPPORTED_VARIANTS:
            raise ValueError(f"Unsupported variant: {variant}")

    features: MolecularFeatures = features_from_frame(data)
    descriptor_map = {
        identity: features.physicochemical[index].copy()
        for index, identity in enumerate(features.row_ids)
    }
    graphs = _precompute_graphs(data) if "edge_aware_gine" in backends else {}
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_keys = list(
        normalized_assignments[["repeat", "fold"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    if smoke:
        fold_keys = fold_keys[:1]

    all_predictions: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    row_lookup = data.set_index(IDENTITY_COLUMN, drop=False)
    for repeat, fold in fold_keys:
        fold_assignment = normalized_assignments.loc[
            normalized_assignments["repeat"].eq(repeat)
            & normalized_assignments["fold"].eq(fold)
        ]
        train_ids, validation_ids = _validate_outer_fold(
            data,
            fold_assignment,
            identity_col=IDENTITY_COLUMN,
            scaffold_col=SCAFFOLD_COLUMN,
        )
        outer_training = row_lookup.loc[train_ids].reset_index(drop=True)
        outer_validation = row_lookup.loc[validation_ids].reset_index(drop=True)
        shared_inner = make_shared_inner_splits(
            outer_training,
            seed=config.seed_base + int(repeat) * 100 + int(fold),
            n_splits=config.inner_folds,
        )
        for backend in backends:
            for variant in variants:
                name = model_name(backend, variant)
                predictions_path, manifest_path = _checkpoint_paths(
                    output_dir, name, int(repeat), int(fold)
                )
                loaded = _load_checkpoint(
                    predictions_path,
                    manifest_path,
                    contract_sha256=contract_sha256,
                )
                if loaded is not None:
                    prediction_frame, checkpoint_manifest = loaded
                    all_predictions.append(prediction_frame)
                    metric_rows.append(dict(checkpoint_manifest["metrics"]))
                    continue

                use_descriptors = variant == GRAPH_DESCRIPTOR_VARIANT
                start = time.perf_counter()
                selected_epochs: list[int] = []
                for inner_fold, (inner_train, inner_validation) in enumerate(shared_inner):
                    train_frame = outer_training.iloc[inner_train].reset_index(drop=True)
                    validation_frame = outer_training.iloc[inner_validation].reset_index(
                        drop=True
                    )
                    inner_seed = (
                        config.seed_base
                        + int(repeat) * 1000
                        + int(fold) * 10
                        + inner_fold
                    )
                    if backend == "edge_aware_gine":
                        epoch = _gine_early_stop_epoch(
                            train_frame,
                            validation_frame,
                            graphs=graphs,
                            descriptor_map=descriptor_map,
                            use_descriptors=use_descriptors,
                            config=config,
                            seed=inner_seed,
                        )
                    else:
                        epoch = _chemprop_early_stop_epoch(
                            train_frame,
                            validation_frame,
                            descriptor_map=descriptor_map,
                            use_descriptors=use_descriptors,
                            config=config,
                            seed=inner_seed,
                        )
                    selected_epochs.append(int(epoch))
                refit_epochs = max(1, int(np.median(selected_epochs)))
                refit_seed = config.seed_base + int(repeat) * 100 + int(fold)
                if backend == "edge_aware_gine":
                    prediction = _gine_refit_predict(
                        outer_training,
                        outer_validation,
                        graphs=graphs,
                        descriptor_map=descriptor_map,
                        use_descriptors=use_descriptors,
                        config=config,
                        seed=refit_seed,
                        epochs=refit_epochs,
                    )
                else:
                    prediction = _chemprop_refit_predict(
                        outer_training,
                        outer_validation,
                        descriptor_map=descriptor_map,
                        use_descriptors=use_descriptors,
                        config=config,
                        seed=refit_seed,
                        epochs=refit_epochs,
                    )
                elapsed = float(time.perf_counter() - start)
                scores = regression_metrics(
                    outer_validation[TARGET_COLUMN].to_numpy(dtype=float), prediction
                )
                metrics = {
                    "model": name,
                    "backend": backend,
                    "variant": variant,
                    "repeat": int(repeat),
                    "fold": int(fold),
                    "n_train": len(outer_training),
                    "n_validation": len(outer_validation),
                    "inner_best_epochs": selected_epochs,
                    "refit_epochs": refit_epochs,
                    **scores,
                    "runtime_seconds": elapsed,
                }
                prediction_frame = pd.DataFrame(
                    {
                        "model": name,
                        "backend": backend,
                        "variant": variant,
                        "repeat": int(repeat),
                        "fold": int(fold),
                        IDENTITY_COLUMN: outer_validation[IDENTITY_COLUMN].astype(str),
                        SCAFFOLD_COLUMN: outer_validation[SCAFFOLD_COLUMN].astype(str),
                        "y_true": outer_validation[TARGET_COLUMN].astype(float),
                        "y_pred": prediction.astype(float),
                    }
                ).sort_values(IDENTITY_COLUMN, kind="mergesort")
                atomic_write_text(
                    predictions_path,
                    prediction_frame.to_csv(
                        index=False, lineterminator="\n", float_format="%.12g"
                    ),
                )
                checkpoint_manifest = {
                    "schema_version": GNN_BENCHMARK_SCHEMA_VERSION,
                    "contract_sha256": contract_sha256,
                    "predictions_sha256": sha256_file(predictions_path),
                    "metrics": metrics,
                    "temporal_test_consulted": False,
                }
                atomic_write_json(manifest_path, checkpoint_manifest)
                all_predictions.append(prediction_frame)
                metric_rows.append(metrics)

    predictions = pd.concat(all_predictions, ignore_index=True).sort_values(
        ["model", "repeat", "fold", IDENTITY_COLUMN], kind="mergesort"
    ).reset_index(drop=True)
    fold_metrics = pd.DataFrame(metric_rows).sort_values(
        ["model", "repeat", "fold"], kind="mergesort"
    ).reset_index(drop=True)
    repeat_rows: list[dict[str, Any]] = []
    for (name, backend, variant, repeat), group in predictions.groupby(
        ["model", "backend", "variant", "repeat"], sort=True
    ):
        if not smoke and group[IDENTITY_COLUMN].duplicated().any():
            raise RuntimeError("Duplicate GNN OOF prediction within a repeat")
        repeat_rows.append(
            {
                "model": name,
                "backend": backend,
                "variant": variant,
                "repeat": int(repeat),
                "n_oof": len(group),
                **regression_metrics(group["y_true"], group["y_pred"]),
            }
        )
    repeat_metrics = pd.DataFrame(repeat_rows)
    summary_rows: list[dict[str, Any]] = []
    for (name, backend, variant), group in repeat_metrics.groupby(
        ["model", "backend", "variant"], sort=True
    ):
        summary_rows.append(
            {
                "model": name,
                "backend": backend,
                "variant": variant,
                "n_repeats": len(group),
                "rmse_mean": float(group["rmse"].mean()),
                "rmse_sd": float(group["rmse"].std(ddof=0)),
                "mae_mean": float(group["mae"].mean()),
                "mae_sd": float(group["mae"].std(ddof=0)),
                "r2_mean": float(group["r2"].mean()),
                "r2_sd": float(group["r2"].std(ddof=0)),
            }
        )
    model_summary = pd.DataFrame(summary_rows).sort_values(
        ["rmse_mean", "model"], kind="mergesort"
    ).reset_index(drop=True)
    manifest = {
        "schema_version": GNN_BENCHMARK_SCHEMA_VERSION,
        "contract_sha256": contract_sha256,
        "backends": list(backends),
        "variants": list(variants),
        "config": config.as_dict(),
        "smoke": bool(smoke),
        "temporal_test_consulted": False,
        "feature_manifest_sha256": sha256_json(features.manifest),
    }
    for filename, frame in {
        "oof_predictions.csv": predictions,
        "fold_metrics.csv": fold_metrics,
        "repeat_metrics.csv": repeat_metrics,
        "model_summary.csv": model_summary,
    }.items():
        atomic_write_text(
            output_dir / filename,
            frame.to_csv(index=False, lineterminator="\n", float_format="%.12g"),
        )
    atomic_write_json(output_dir / "run_manifest.json", manifest)
    atomic_write_text(output_dir / "RUN_COMPLETE", contract_sha256 + "\n")
    return GNNBenchmarkResult(
        predictions=predictions,
        fold_metrics=fold_metrics,
        repeat_metrics=repeat_metrics,
        model_summary=model_summary,
        run_manifest=manifest,
    )

