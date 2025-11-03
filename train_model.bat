@echo off
REM Train Model with Real Data
REM Usage: train_model.bat

echo ========================================
echo Training Heart Disease Risk Model
echo Using Real Data
echo ========================================
echo.

REM Activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

echo Starting training...
echo.

python train_real_data.py

echo.
pause


