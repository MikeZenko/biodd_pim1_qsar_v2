"""Dependency-light contract for the Chemprop D-MPNN baseline.

This adapter validates immutable inputs and serializes training settings. It
does not fit a model; the later VM training runner must explicitly consume the
contract after the split and temporal-test locks have been verified.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Sequence

import numpy as np

from .gnn_features import canonicalize_smiles


EXPECTED_CHEMPROP_VERSION = "2.2.4"


def installed_chemprop_version() -> str | None:
    """Return the installed Chemprop version without importing the package."""

    try:
        return version("chemprop")
    except PackageNotFoundError:
        return None


def require_chemprop() -> str:
    """Require the audited Chemprop release and return its version."""

    installed = installed_chemprop_version()
    if installed is None:
        raise ImportError(
            "Chemprop is not installed in this environment. Run "
            "scripts/setup_gnn_vm.sh on the compute VM."
        )
    if installed != EXPECTED_CHEMPROP_VERSION:
        raise RuntimeError(
            f"Expected chemprop=={EXPECTED_CHEMPROP_VERSION}, found {installed}"
        )
    try:
        import_module("chemprop")
    except (ImportError, OSError) as exc:
        raise ImportError(
            "The audited Chemprop distribution is installed but could not be "
            "imported in this environment"
        ) from exc
    return installed


@dataclass(frozen=True)
class ChempropConfig:
    """Serializable, side-effect-free D-MPNN training configuration."""

    depth: int = 3
    message_hidden_dim: int = 256
    ffn_hidden_dim: int = 256
    ffn_num_layers: int = 2
    dropout: float = 0.1
    aggregation: str = "mean"
    batch_size: int = 64
    max_epochs: int = 150
    patience: int = 20
    min_delta: float = 0.001
    grad_clip: float = 5.0
    warmup_epochs: int = 2
    init_lr: float = 1e-4
    max_lr: float = 1e-3
    final_lr: float = 1e-4
    seed: int = 13

    def __post_init__(self) -> None:
        for name in (
            "depth",
            "message_hidden_dim",
            "ffn_hidden_dim",
            "ffn_num_layers",
            "batch_size",
            "max_epochs",
            "patience",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.warmup_epochs, bool)
            or not isinstance(self.warmup_epochs, int)
            or self.warmup_epochs < 0
        ):
            raise ValueError("warmup_epochs must be a non-negative integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.min_delta < 0:
            raise ValueError("min_delta must be non-negative")
        if self.grad_clip <= 0:
            raise ValueError("grad_clip must be positive")
        if self.aggregation not in {"mean", "sum", "norm"}:
            raise ValueError("aggregation must be 'mean', 'sum', or 'norm'")
        if min(self.init_lr, self.max_lr, self.final_lr) <= 0:
            raise ValueError("Learning rates must be positive")
        if self.init_lr > self.max_lr or self.final_lr > self.max_lr:
            raise ValueError("init_lr and final_lr cannot exceed max_lr")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def cli_arguments(self) -> tuple[str, ...]:
        """Return the equivalent Chemprop 2.2.4 CLI settings."""

        return (
            "--task-type",
            "regression",
            "--loss-function",
            "mse",
            "--metrics",
            "rmse",
            "mae",
            "r2",
            "--tracking-metric",
            "rmse",
            "--depth",
            str(self.depth),
            "--message-hidden-dim",
            str(self.message_hidden_dim),
            "--ffn-hidden-dim",
            str(self.ffn_hidden_dim),
            "--ffn-num-layers",
            str(self.ffn_num_layers),
            "--dropout",
            str(self.dropout),
            "--aggregation",
            self.aggregation,
            "--batch-size",
            str(self.batch_size),
            "--epochs",
            str(self.max_epochs),
            "--patience",
            str(self.patience),
            "--min-delta",
            str(self.min_delta),
            "--grad-clip",
            str(self.grad_clip),
            "--warmup-epochs",
            str(self.warmup_epochs),
            "--init-lr",
            str(self.init_lr),
            "--max-lr",
            str(self.max_lr),
            "--final-lr",
            str(self.final_lr),
            "--data-seed",
            str(self.seed),
            "--pytorch-seed",
            str(self.seed),
        )


@dataclass(frozen=True)
class ChempropInputBatch:
    """Aligned, canonicalized single-task inputs for a future D-MPNN run."""

    row_ids: tuple[str, ...]
    canonical_smiles: tuple[str, ...]
    targets: np.ndarray
    descriptors: np.ndarray | None = None

    @classmethod
    def from_sequences(
        cls,
        *,
        row_ids: Sequence[str],
        smiles: Sequence[str],
        targets: Sequence[float] | np.ndarray,
        descriptors: Sequence[Sequence[float]] | np.ndarray | None = None,
    ) -> "ChempropInputBatch":
        """Validate alignment and freeze arrays without importing Chemprop."""

        frozen_row_ids = tuple(row_ids)
        frozen_smiles = tuple(canonicalize_smiles(value) for value in smiles)
        if not frozen_row_ids:
            raise ValueError("A Chemprop input batch cannot be empty")
        if len(set(frozen_row_ids)) != len(frozen_row_ids):
            raise ValueError("row_ids must be unique")
        if any(not isinstance(value, str) or not value for value in frozen_row_ids):
            raise ValueError("Every row_id must be a non-empty string")
        if len(frozen_row_ids) != len(frozen_smiles):
            raise ValueError("row_ids and smiles have different lengths")

        target_array = np.asarray(targets, dtype=np.float32)
        if target_array.ndim == 1:
            target_array = target_array.reshape(-1, 1)
        if target_array.shape != (len(frozen_row_ids), 1):
            raise ValueError("targets must contain one scalar value per row")
        if not np.isfinite(target_array).all():
            raise ValueError("targets contains a non-finite value")
        target_array = target_array.copy()
        target_array.setflags(write=False)

        descriptor_array: np.ndarray | None = None
        if descriptors is not None:
            descriptor_array = np.asarray(descriptors, dtype=np.float32)
            if descriptor_array.ndim != 2:
                raise ValueError("descriptors must be a two-dimensional array")
            if descriptor_array.shape[0] != len(frozen_row_ids):
                raise ValueError("descriptors and row_ids have different lengths")
            if descriptor_array.shape[1] == 0:
                raise ValueError("descriptors cannot have zero columns")
            if not np.isfinite(descriptor_array).all():
                raise ValueError("descriptors contains a non-finite value")
            descriptor_array = descriptor_array.copy()
            descriptor_array.setflags(write=False)

        return cls(
            row_ids=frozen_row_ids,
            canonical_smiles=frozen_smiles,
            targets=target_array,
            descriptors=descriptor_array,
        )

    @property
    def descriptor_dim(self) -> int:
        return 0 if self.descriptors is None else int(self.descriptors.shape[1])


class ChempropAdapter:
    """Validated contract boundary for a later, explicit VM training runner."""

    backend = "chemprop_dmpnn"

    def __init__(self, config: ChempropConfig | None = None) -> None:
        self.config = config or ChempropConfig()

    def assert_environment(self) -> str:
        """Assert that the exact audited Chemprop distribution is installed."""

        return require_chemprop()

    def prepare_inputs(
        self,
        *,
        row_ids: Sequence[str],
        smiles: Sequence[str],
        targets: Sequence[float] | np.ndarray,
        descriptors: Sequence[Sequence[float]] | np.ndarray | None = None,
    ) -> ChempropInputBatch:
        """Build an immutable aligned batch without fitting any transformer."""

        return ChempropInputBatch.from_sequences(
            row_ids=row_ids,
            smiles=smiles,
            targets=targets,
            descriptors=descriptors,
        )

    def training_contract(self, batch: ChempropInputBatch) -> dict[str, Any]:
        """Describe the required future run; this method never starts training."""

        return {
            "backend": self.backend,
            "expected_chemprop_version": EXPECTED_CHEMPROP_VERSION,
            "task_type": "regression",
            "loss": "mse",
            "tracking_metric": "rmse",
            "target_scaling": "fit_on_training_partition_only",
            "descriptor_scaling": (
                "fit_on_training_partition_only"
                if batch.descriptors is not None
                else "not_applicable"
            ),
            "n_rows": len(batch.row_ids),
            "descriptor_dim": batch.descriptor_dim,
            "config": self.config.as_dict(),
        }
