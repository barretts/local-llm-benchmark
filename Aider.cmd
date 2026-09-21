@echo off
if "%~1"=="" goto menu
"%~dp0.venv\Scripts\python.exe" "%~dp0launcher\api.py" aider --repo "%~1"
goto done
:menu
"%~dp0.venv\Scripts\python.exe" "%~dp0launcher\api.py" menu
:done
if errorlevel 1 pause
