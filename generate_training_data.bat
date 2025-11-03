@echo off
REM Generate Synthetic Training Dataset
REM Usage: generate_training_data.bat

echo ========================================
echo Generating Synthetic Training Dataset
echo ========================================
echo.

REM Activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

echo.
set /p num_samples="Enter number of samples (default 500): "
if "%num_samples%"=="" set num_samples=500

echo.
echo Generating %num_samples% synthetic samples...
python generate_training_dataset.py --num-samples %num_samples%

echo.
pause

