#!/usr/bin/env bash
# Reproduce the frozen-data model evaluation on a compute VM.
# This intentionally skips the live ChEMBL download and descriptor regeneration.

set -euo pipefail

cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-$PWD/.venv/bin/python}"
TARGET_ID="${TARGET_ID:-CHEMBL2147}"
SEED="${SEED:-42}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python runtime not found: $PYTHON_BIN" >&2
  echo "Create the VM environment first: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

export MPLBACKEND=Agg
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

"$PYTHON_BIN" data_audit.py
"$PYTHON_BIN" 5.Comparing_Regressors.py --target-id "$TARGET_ID" --seed "$SEED" --force
"$PYTHON_BIN" reconcile_selected_model.py --target-id "$TARGET_ID" --seed "$SEED"
"$PYTHON_BIN" generate_evaluation_artifacts.py --target-id "$TARGET_ID" --seed "$SEED"
"$PYTHON_BIN" 6.Applicability_Domain.py --target-id "$TARGET_ID" --seed "$SEED"
"$PYTHON_BIN" 7.Model_Interpretability.py --target-id "$TARGET_ID" --seed "$SEED"
"$PYTHON_BIN" generate_manuscript_artifacts.py --target-id "$TARGET_ID"
"$PYTHON_BIN" make_paper_figures.py --target-id "$TARGET_ID" --seed "$SEED"
