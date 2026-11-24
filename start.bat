@echo off
title bskdetect
cd /d "%~dp0"

echo ========================================
echo   bskdetect Testbed Startup
echo ========================================
echo.

echo [1/3] Checking Python...
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: .venv\Scripts\python.exe not found
    echo Please run: python -m venv .venv ^&^& .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

echo [2/3] Starting server (http://127.0.0.1:8765)...
start "bskdetect-server" /min cmd /c ".venv\Scripts\python.exe -m uvicorn bsk.server:app --host 127.0.0.1 --port 8765 --reload > server.log 2>&1"

echo Waiting for server...
timeout /t 4 /nobreak >nul

echo [3/3] Opening browser...
timeout /t 2 /nobreak >nul
start http://127.0.0.1:8765

echo.
echo ========================================
echo   Server running at http://127.0.0.1:8765
echo   Log: server.log
echo ========================================
echo.
echo Press any key to stop server...
pause >nul

taskkill /fi "WINDOWTITLE eq bskdetect-server*" /t /f >nul 2>&1
echo Server stopped.
