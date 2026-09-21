@echo off
"%~dp0.venv\Scripts\python.exe" "%~dp0launcher\api.py" menu
if errorlevel 1 pause
