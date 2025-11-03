#!/bin/bash
# Streamlit Application Launcher
# Usage: ./start_streamlit.sh

echo "Starting Heart Disease Risk Assessment DSS..."
echo "============================================"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "Error: Streamlit is not installed."
    echo "Please run: pip install -r requirements.txt"
    exit 1
fi

# Check if model exists
if [ ! -f "models/baseline_model.pkl" ]; then
    echo "Warning: Model not found. Please train the model first:"
    echo "  jupyter notebook notebooks/train_demo.ipynb"
    echo ""
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Start Streamlit
echo "Launching Streamlit application..."
streamlit run app.py

