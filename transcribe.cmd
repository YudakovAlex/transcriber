@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "VENV_DIR=%SCRIPT_DIR%.venv"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"
set "STAMP_FILE=%VENV_DIR%\.deps_installed"
set "PYTHON_CMD="
set "PYTHON_ARGS="

where ffmpeg >nul 2>&1
if errorlevel 1 (
  echo ffmpeg not found on PATH. Install it and retry.
  echo Expected example: C:\ffmpeg\bin\ffmpeg.exe
  exit /b 1
)

if not exist "%PYTHON_EXE%" (
  call :find_python3
  if not defined PYTHON_CMD (
    echo Failed to find Python 3.8+. Ensure one of py, python, or python3 is on PATH.
    exit /b 1
  )
  echo Creating venv...
  %PYTHON_CMD% %PYTHON_ARGS% -m venv "%VENV_DIR%"
  if errorlevel 1 (
    echo Failed to create venv. Ensure Python 3.8+ is installed.
    exit /b 1
  )
)

if not exist "%STAMP_FILE%" (
  if exist "%SCRIPT_DIR%requirements.txt" (
    "%PYTHON_EXE%" -m pip install -r "%SCRIPT_DIR%requirements.txt"
  ) else (
    "%PYTHON_EXE%" -m pip install openai-whisper torch
  )
  if errorlevel 1 (
    echo Dependency install failed.
    exit /b 1
  )
  echo installed> "%STAMP_FILE%"
)

"%PYTHON_EXE%" "%SCRIPT_DIR%transcribe.py" %*
endlocal
goto :eof

:find_python3
where py >nul 2>&1
if not errorlevel 1 (
  py -3.12 -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3,12) else 1)" >nul 2>&1
  if not errorlevel 1 (
    set "PYTHON_CMD=py"
    set "PYTHON_ARGS=-3.12"
    goto :eof
  )
)

set "PY312=%LOCALAPPDATA%\Python\pythoncore-3.12-64\python.exe"
if exist "%PY312%" (
  "%PY312%" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3,12) else 1)" >nul 2>&1
  if not errorlevel 1 (
    set "PYTHON_CMD=%PY312%"
    set "PYTHON_ARGS="
    goto :eof
  )
)

where py >nul 2>&1
if not errorlevel 1 (
  py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3,8) else 1)" >nul 2>&1
  if not errorlevel 1 (
    set "PYTHON_CMD=py"
    set "PYTHON_ARGS=-3"
    goto :eof
  )
)

where python >nul 2>&1
if not errorlevel 1 (
  python -c "import sys; raise SystemExit(0 if sys.version_info >= (3,8) else 1)" >nul 2>&1
  if not errorlevel 1 (
    set "PYTHON_CMD=python"
    goto :eof
  )
)

where python3 >nul 2>&1
if not errorlevel 1 (
  python3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3,8) else 1)" >nul 2>&1
  if not errorlevel 1 (
    set "PYTHON_CMD=python3"
    goto :eof
  )
)

set "PYTHON_CMD="
set "PYTHON_ARGS="
goto :eof
