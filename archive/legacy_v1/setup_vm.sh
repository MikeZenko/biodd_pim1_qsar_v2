#!/bin/bash
# BIODD VM Setup Script
# Run this after uploading to Google Cloud VM

set -e

echo "============================================"
echo "BIODD Project VM Setup"
echo "============================================"

# Update system
echo "[1/6] Updating system packages..."
sudo apt-get update -y
sudo apt-get upgrade -y

# Install Python and dependencies
echo "[2/6] Installing Python and build tools..."
sudo apt-get install -y python3.12 python3.12-venv python3-pip build-essential libxrender1 libxext6

# Create virtual environment
echo "[3/6] Creating virtual environment..."
python3.12 -m venv biodd_env
source biodd_env/bin/activate

# Upgrade pip
echo "[4/6] Upgrading pip..."
pip install --upgrade pip wheel setuptools

# Install dependencies
echo "[5/6] Installing Python dependencies..."
pip install -r requirements.txt

# Create output directories
echo "[6/6] Creating output directories..."
mkdir -p data models plots results

echo ""
echo "============================================"
echo "Setup complete!"
echo "============================================"
echo ""
echo "To activate environment: source biodd_env/bin/activate"
echo "To run project: ./run_pipeline.sh"
echo ""
