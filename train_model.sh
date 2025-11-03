#!/bin/bash
# Train Model with Real Data
# Usage: bash train_model.sh

echo "========================================"
echo "Training Heart Disease Risk Model"
echo "Using Real Data"
echo "========================================"
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

echo "Starting training..."
echo ""

python train_real_data.py

echo ""
read -p "Press enter to continue..."


