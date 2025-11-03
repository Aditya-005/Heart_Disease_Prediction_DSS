@echo off
REM Streamlit Application Launcher for Windows
REM Usage: start_streamlit.bat

echo Starting Heart Disease Risk Assessment DSS...
echo ============================================

REM Activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Check if model exists
if not exist models\baseline_model.pkl (
    echo Warning: Model not found. Please train the model first:
    echo   jupyter notebook notebooks\train_demo.ipynb
    echo.
    pause
)

REM Start Streamlit
echo Launching Streamlit application...
python -m streamlit run app.py

pause

