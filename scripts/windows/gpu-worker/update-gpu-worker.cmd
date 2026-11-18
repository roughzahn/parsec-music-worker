@echo off
title Parsec GPU Worker Updater
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0update-gpu-worker.ps1"
set "UPDATE_EXIT=%ERRORLEVEL%"
echo.
if not "%UPDATE_EXIT%"=="0" (
  echo The update did not complete successfully.
) else (
  echo Update script finished.
)
echo.
pause
exit /b %UPDATE_EXIT%
