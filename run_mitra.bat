@echo off
title MITRA - AI Companion & Care Monitor
color 0B
echo =======================================================
echo          STARTING MITRA AI COMPANION
echo =======================================================
echo.

echo [1/3] Starting Parent Dashboard Server (port 5000)...
start "Mitra Server" /min python mitra_server.py

timeout /t 2 >nul

echo [2/3] Opening Parent Dashboard in browser...
start "" "parent_dashboard.html"

echo [3/3] Launching Mitra AI Voice Companion...
echo.
echo Tip: Say 'Mitra' to wake up!
echo      Say 'Mitra stop' to interrupt!
echo =======================================================
echo.

python mitra_ultimate.py

pause
