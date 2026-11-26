@echo off
title MITRA - AI Companion ^& Care Monitor
color 0B
cd /d "%~dp0"
echo =======================================================
echo          STARTING MITRA AI COMPANION
echo =======================================================
echo.

rem Hub interpreter: the project .venv if it has the hub packages, else system python
set "PY=python"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -c "import flask, groq" >nul 2>&1 && set "PY=.venv\Scripts\python.exe"
)
"%PY%" -c "import flask, flask_cors, requests" >nul 2>&1
if errorlevel 1 (
    echo Missing hub packages. Run:  "%PY%" -m pip install -r requirements.txt
    pause
    exit /b 1
)

rem Reuse a hub that is already running; a second one would only fail with "port in use"
curl -s -o nul --max-time 1 http://127.0.0.1:5000/ping && (
    echo [1/3] Mitra hub already running on port 5000 - reusing it.
    goto ready
)

echo [1/3] Starting Mitra hub on port 5000 (%PY%)...
start "Mitra Server" cmd /k ""%PY%" mitra_server.py"

echo        Waiting for the hub to answer /ping ...
set /a TRIES=0
:waitloop
curl -s -o nul --max-time 1 http://127.0.0.1:5000/ping && goto ready
set /a TRIES+=1
if %TRIES% GEQ 40 (
    echo The hub did not start - check the "Mitra Server" window for the error.
    pause
    exit /b 1
)
rem ~1 s pause that works even without an interactive console (timeout /t does not)
ping -n 2 127.0.0.1 >nul
goto waitloop
:ready

echo [2/3] Opening dashboard: http://localhost:5000
start "" "http://localhost:5000"

echo [3/3] Launching Mitra AI Voice Companion...
echo.
echo Tip: Say 'Mitra' to wake up!
echo      Say 'Mitra stop' to interrupt!
echo =======================================================
echo.

rem Companion needs whisper/sounddevice; use whichever interpreter has them
set "CPY=%PY%"
"%CPY%" -c "import whisper, sounddevice" >nul 2>&1 || set "CPY=python"
"%CPY%" mitra_ultimate.py

pause
