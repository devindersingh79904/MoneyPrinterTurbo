@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\automation\stop_services.ps1"
pause
