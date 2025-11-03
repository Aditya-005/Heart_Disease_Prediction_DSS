@echo off
REM Download ECG Data from PhysioNet
REM Usage: download_data.bat

echo ========================================
echo Downloading ECG Data from PhysioNet
echo ========================================
echo.

REM Activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

echo.
echo Options:
echo   1. Download sample records (10 records for testing)
echo   2. Download PTB-DB dataset (requires PhysioNet account)
echo.
set /p choice="Enter choice (1 or 2): "

if "%choice%"=="1" (
    echo Downloading sample records...
    python download_physionet_data.py --dataset sample --num-samples 10
) else if "%choice%"=="2" (
    echo Downloading PTB-DB dataset...
    echo.
    echo Note: You need:
    echo   - PhysioNet account: https://physionet.org/
    echo   - Signed data use agreement for PTB-DB
    echo.
    pause
    python download_physionet_data.py --dataset ptbdb
) else (
    echo Invalid choice. Using sample download...
    python download_physionet_data.py --dataset sample --num-samples 10
)

echo.
pause

