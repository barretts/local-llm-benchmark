@echo off
"%~dp0.venv\Scripts\python.exe" "%~dp0launcher\api.py" aider --repo "%cd%"
if errorlevel 1 pause
