# SENTIVA Launcher (PowerShell)
Write-Host ""
Write-Host "  SENTIVA -- Multilingual Sentiment AI" -ForegroundColor Cyan
Write-Host "  ======================================" -ForegroundColor DarkGray
Write-Host ""
Write-Host "  [1/3] Installing dependencies..." -ForegroundColor Yellow
Set-Location "$PSScriptRoot\backend"
pip install -r requirements.txt -q
Write-Host "  [2/3] Starting server (model training on first run ~30-60s)..." -ForegroundColor Yellow
Write-Host "  [3/3] Open http://localhost:8000 in your browser" -ForegroundColor Green
Write-Host ""
Start-Process "http://localhost:8000"
python -m uvicorn main:app --host 0.0.0.0 --port 8000
