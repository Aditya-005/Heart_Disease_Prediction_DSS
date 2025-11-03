@echo off
REM Local CLI Test Runner for Windows
REM Usage: run_local.bat

echo Running Local Heart Disease Risk Assessment...
echo ==============================================

REM Activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Check if model exists
if not exist models\baseline_model.pkl (
    echo Error: Model not found. Please train the model first:
    echo   python train_real_data.py
    echo   or
    echo   .\train_model.bat
    pause
    exit /b 1
)

REM Check if sample ECG exists
if not exist sample_data\sample_ecg.csv (
    echo Warning: Sample ECG not found. Generating...
    python generate_sample_ecg.py
)

REM Run CLI test
echo.
echo Running prediction with sample data...
echo --------------------------------------
python main.py --ecg sample_data\sample_ecg.csv --age 55 --bp_systolic 140 --bp_diastolic 90 --cholesterol 220 --gender M --save

echo.
echo Done! Check predictions.csv for saved results.
pause

