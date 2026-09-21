@echo off
if "%~1"=="" goto prompt
"%~dp0.venv\Scripts\python.exe" "%~dp0launcher\api.py" aider --repo "%~1"
goto done
:prompt
"%~dp0.venv\Scripts\python.exe" "%~dp0launcher\api.py" aider --prompt-repo
:done
if errorlevel 1 pause
