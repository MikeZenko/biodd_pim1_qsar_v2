#!/usr/bin/env python3
"""Capture the audited runtime used for a v2 curation/modeling stage."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pim1_qsar.artifacts import atomic_write_json, sha256_file, sha256_json  # noqa: E402


AUDITED_DISTRIBUTIONS = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "joblib",
    "rdkit",
    "chembl-structure-pipeline",
    "PyYAML",
    "torch",
    "torch-geometric",
    "chemprop",
    "lightning",
    "matplotlib",
    "seaborn",
    "pytest",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requirements = args.requirements.resolve()
    if not requirements.is_file():
        raise FileNotFoundError(requirements)
    packages = {}
    for distribution in AUDITED_DISTRIBUTIONS:
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    accelerator = "not_installed"
    try:
        import torch

        accelerator = "cuda" if torch.cuda.is_available() else "cpu"
    except (ImportError, OSError):
        pass
    core = {
        "label": args.label,
        "python": sys.version,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "torch_accelerator": accelerator,
        "packages": packages,
        "requirements_filename": requirements.name,
        "requirements_sha256": sha256_file(requirements),
    }
    payload = dict(core)
    payload["environment_sha256"] = sha256_json(core)
    atomic_write_json(args.output, payload)
    print(f"{args.label}: {payload['environment_sha256']}")


if __name__ == "__main__":
    main()

