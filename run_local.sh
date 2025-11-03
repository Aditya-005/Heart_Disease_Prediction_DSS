#!/bin/bash
# Local CLI Test Runner
# Usage: ./run_local.sh

echo "Running Local Heart Disease Risk Assessment..."
echo "=============================================="

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Check if model exists
if [ ! -f "models/baseline_model.pkl" ]; then
    echo "Error: Model not found. Please train the model first:"
    echo "  jupyter notebook notebooks/train_demo.ipynb"
    exit 1
fi

# Check if sample ECG exists
if [ ! -f "sample_data/sample_ecg.csv" ]; then
    echo "Warning: Sample ECG not found. Generating..."
    python generate_sample_ecg.py
fi

# Run CLI test
echo ""
echo "Running prediction with sample data..."
echo "--------------------------------------"
python main.py \
    --ecg sample_data/sample_ecg.csv \
    --age 55 \
    --bp_systolic 140 \
    --bp_diastolic 90 \
    --cholesterol 220 \
    --gender M \
    --save

echo ""
echo "Done! Check predictions.csv for saved results."

