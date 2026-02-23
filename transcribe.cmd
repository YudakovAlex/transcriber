@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "VENV_DIR=%SCRIPT_DIR%venv"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"
set "STAMP_FILE=%VENV_DIR%\.deps_installed"

where ffmpeg >nul 2>&1
if errorlevel 1 (
  echo ffmpeg not found on PATH. Install it and retry.
  echo Expected example: C:\ffmpeg\bin\ffmpeg.exe
  exit /b 1
)

if not exist "%PYTHON_EXE%" (
  echo Creating venv...
  py -3 -m venv "%VENV_DIR%"
  if errorlevel 1 (
    echo Failed to create venv. Ensure Python is installed.
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
