@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0make_video.ps1" %*
pause
