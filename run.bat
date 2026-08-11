@echo off
rem Windows launcher: sets up the virtual environment on first run, then starts
rem the app. Double-click it, or run it from cmd / PowerShell.
setlocal
cd /d "%~dp0"

set PY=.venv\Scripts\python.exe

if not exist "%PY%" call :make_venv
if not exist "%PY%" goto :no_python

echo [1/3] Checking dependencies...
"%PY%" -m pip install --quiet --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto :pip_failed

if not exist ".env" goto :make_env

echo [2/3] Configuration found.
echo [3/3] Starting the server.
echo.
echo    Open http://127.0.0.1:5000 in Chrome or Edge.
echo    Use 127.0.0.1 - the microphone will not work from any other address.
echo    Press Ctrl+C in this window to stop.
echo.
"%PY%" app.py
goto :end

:make_venv
echo [0/3] Creating the virtual environment (first run only)...
py -3 -m venv .venv 2>nul
if not exist "%PY%" python -m venv .venv 2>nul
goto :eof

:make_env
copy /y ".env.example" ".env" >nul
echo.
echo Created .env for you. Opening it in Notepad now.
echo Paste in your DEEPGRAM_API_KEY and OPENROUTER_API_KEY, save, close
echo Notepad, then run this file again.
echo.
notepad .env
pause
goto :end

:no_python
echo.
echo Could not find Python. Install Python 3.10 or newer from
echo    https://www.python.org/downloads/
echo and tick "Add python.exe to PATH" during setup, then run this again.
echo.
pause
goto :end

:pip_failed
echo.
echo Installing dependencies failed. If you are behind a corporate proxy or
echo firewall, that is the usual cause. The error above has the detail.
echo.
pause

:end
endlocal
