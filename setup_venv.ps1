# PowerShell script to create and setup virtual environment
Write-Host "Creating virtual environment..." -ForegroundColor Green
python -m venv venv

Write-Host "`nActivating virtual environment..." -ForegroundColor Green
& .\venv\Scripts\Activate.ps1

Write-Host "`nUpgrading pip..." -ForegroundColor Green
python -m pip install --upgrade pip

Write-Host "`nInstalling dependencies from requirements.txt..." -ForegroundColor Green
pip install -r requirements.txt

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Virtual environment setup complete!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "`nTo activate the virtual environment in the future, run:" -ForegroundColor Yellow
Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "`nOr if you have execution policy restrictions:" -ForegroundColor Yellow
Write-Host "  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser" -ForegroundColor White
Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "`nTo deactivate, simply type:" -ForegroundColor Yellow
Write-Host "  deactivate" -ForegroundColor White



