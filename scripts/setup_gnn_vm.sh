#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GNN_VENV="${GNN_VENV:-$PROJECT_ROOT/.venv-gnn}"

"$PYTHON_BIN" - <<'PY'
import sys

if not ((3, 11) <= sys.version_info[:2] < (3, 15)):
    raise SystemExit(
        "The GNN environment requires Python 3.11-3.14; "
        f"found {sys.version.split()[0]}"
    )
PY

if [[ ! -d "$GNN_VENV" ]]; then
  "$PYTHON_BIN" -m venv "$GNN_VENV"
fi

VENV_PYTHON="$GNN_VENV/bin/python"

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "GNN virtual environment is incomplete: $GNN_VENV" >&2
  exit 1
fi

# Install Torch from its CPU index first so the CPU-only VM does not pull a
# CUDA-enabled build. The requirements file repeats the exact pin so that the
# environment contract remains visible in one place.
"$VENV_PYTHON" -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  'torch==2.12.1'
"$VENV_PYTHON" -m pip install -r "$PROJECT_ROOT/requirements-gnn.txt"

"$VENV_PYTHON" - <<'PY'
from importlib.metadata import version
import chemprop
from rdkit import Chem
import torch
import torch_geometric

expected = {
    "torch": "2.12.1",
    "torch-geometric": "2.8.0",
    "chemprop": "2.2.4",
    "rdkit": "2026.3.3",
}

for distribution, expected_version in expected.items():
    installed = version(distribution)
    # Official CPU-only Torch wheels may append a local ``+cpu`` build tag.
    comparable = installed.split("+", 1)[0] if distribution == "torch" else installed
    if comparable != expected_version:
        raise SystemExit(
            f"Version mismatch for {distribution}: "
            f"expected {expected_version}, found {installed}"
        )
    print(f"{distribution}={installed}")

print(f"torch_accelerator={'cuda' if torch.cuda.is_available() else 'cpu'}")
if Chem.MolFromSmiles("CCO") is None:
    raise SystemExit("RDKit smoke import succeeded but molecule parsing failed")
print("GNN environment is ready. No model training was started.")
PY
