@echo off
title SENTIVA
color 0A
echo.
echo  ============================================
echo    SENTIVA  --  Multilingual Sentiment AI
echo  ============================================
echo.
echo  [1/2] Training model and starting server...
cd /d "%~dp0backend"
echo         This may take 30-60 seconds on first run.
echo.
echo  [2/2] Open http://localhost:8000 in your browser
echo         Press Ctrl+C to stop the server.
echo.
start "" http://localhost:8000
python -m uvicorn main:app --host 0.0.0.0 --port 8000
pause
