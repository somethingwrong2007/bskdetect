@echo off
setlocal
title bskdetect realtime
cd /d "%~dp0"

echo ========================================
echo   bskdetect 实时模式启动
echo ========================================
echo.

echo [1/3] Checking Python...
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: .venv\Scripts\python.exe not found
    echo Please run: python -m venv .venv ^&^& .venv\Scripts\pip install -e .
    pause
    exit /b 1
)

set "PYTHONUNBUFFERED=1"
set "BSK_PROFILE=auto"
set "BSK_TRT=0"

echo [2/3] Starting realtime server (http://127.0.0.1:8000)...
start "bskdetect-server" /min cmd /c ".venv\Scripts\python.exe -u -m uvicorn bsk.server:app --host 127.0.0.1 --port 8000 > server.log 2>&1"

echo Waiting for model warmup...
timeout /t 8 /nobreak >nul

echo [3/3] Opening browser...
start http://127.0.0.1:8000/

echo.
echo ========================================
echo   Realtime server: http://127.0.0.1:8000
echo   Mode: on-demand realtime (no reload)
echo   Log: server.log
echo ========================================
echo.
echo Press any key to stop server...
pause >nul

taskkill /fi "WINDOWTITLE eq bskdetect-server*" /t /f >nul 2>&1
echo Server stopped.
endlocal