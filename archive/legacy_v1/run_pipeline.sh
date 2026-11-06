#!/bin/bash
# BIODD Pipeline Runner
# Machine Learning-Driven Bioactivity Prediction for Pancreatic Cancer Drug Discovery
#
# This script runs all steps in sequence, training a model for PIM1 kinase.
#
# Target:
#   - CHEMBL2147: PIM1 (Serine/threonine-protein kinase Pim-1)

set -e

if [[ "${ALLOW_LIVE_REFRESH:-}" != "1" ]]; then
    echo "run_pipeline.sh is a legacy live-data workflow and is disabled by default."
    echo "For the frozen manuscript evaluation on a VM, use: bash run_vm_evaluation.sh"
    echo "Set ALLOW_LIVE_REFRESH=1 only for an intentional development refresh."
    exit 2
fi

# Change to script directory
cd "$(dirname "$0")"

# Activate virtual environment if not already active
if [[ "$VIRTUAL_ENV" == "" ]]; then
    if [[ -d "biodd_env" ]]; then
        source biodd_env/bin/activate
    else
        echo "WARNING: Virtual environment 'biodd_env' not found."
        echo "Please create it with: python -m venv biodd_env && pip install -r requirements.txt"
    fi
fi

echo "============================================"
echo "BIODD Pipeline - Starting"
echo "============================================"
echo "Machine Learning for Pancreatic Cancer Drug Discovery"
echo ""
echo "Target: CHEMBL2147 - PIM1 (Proto-oncogene serine/threonine-protein kinase Pim-1)"
echo ""
echo "Start time: $(date)"
echo ""

# Step 1: Download data
echo "[Step 1/7] Downloading bioactivity data from ChEMBL..."
echo "  Downloading IC50 data for PIM1 kinase..."
python 1.Download_Bioactivity_Data.py
echo "  Step 1 complete!"
echo ""

# Step 2: EDA
echo "[Step 2/7] Exploratory Data Analysis..."
python 2.Exploratory_Data_Analysis.py
echo "  Step 2 complete!"
echo ""

# Step 3: Descriptors
echo "[Step 3/7] Calculating molecular descriptors..."
echo "  Computing Morgan fingerprints (ECFP6) and physicochemical descriptors..."
python 3.Descriptor_Calculation_and_Dataset_Preparation.py
echo "  Step 3 complete!"
echo ""

# Steps 4-7: PIM1-specific modeling
TARGET_ID="CHEMBL2147"
TARGET_NAME="PIM1"

echo "============================================"
echo "Training model for ${TARGET_NAME} (${TARGET_ID})"
echo "============================================"
echo ""

echo "[Step 4/7] Training Random Forest models for ${TARGET_NAME}..."
python 4.Regression_Models_with_Random_Forest.py --target-id "${TARGET_ID}"
echo "  Step 4 complete!"
echo ""

echo "[Step 5/7] Comparing all regression models for ${TARGET_NAME}..."
python 5.Comparing_Regressors.py --target-id "${TARGET_ID}"
echo "  Step 5 complete!"
echo ""

echo "[Step 6/7] Applicability Domain Analysis for ${TARGET_NAME}..."
python 6.Applicability_Domain.py --target-id "${TARGET_ID}"
echo "  Step 6 complete!"
echo ""

echo "[Step 7/7] Model Interpretability Analysis for ${TARGET_NAME}..."
python 7.Model_Interpretability.py --target-id "${TARGET_ID}"
echo "  Step 7 complete!"
echo ""

echo "============================================"
echo "BIODD Pipeline - Complete!"
echo "============================================"
echo "End time: $(date)"
echo ""
echo "Results saved in:"
echo "  - data/              : Processed datasets"
echo "  - models/CHEMBL2147/ : PIM1 trained models"
echo "  - plots/             : Visualizations"
echo "  - results/           : Performance metrics"
