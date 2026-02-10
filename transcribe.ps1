$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $ScriptDir ".venv"
$PythonExe = Join-Path $VenvDir "Scripts\python.exe"
$StampFile = Join-Path $VenvDir ".deps_installed"

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Error "ffmpeg not found on PATH. Install it and retry. Example: C:\ffmpeg\bin\ffmpeg.exe"
    exit 1
}

if (-not (Test-Path $PythonExe)) {
    Write-Host "Creating venv..."
    py -3 -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create venv. Ensure Python is installed."
        exit 1
    }
}

$ReqFile = Join-Path $ScriptDir "requirements.txt"
if (-not (Test-Path $StampFile)) {
    if (Test-Path $ReqFile) {
        & $PythonExe -m pip install -r $ReqFile
    } else {
        & $PythonExe -m pip install openai-whisper torch
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Dependency install failed."
        exit 1
    }
    "installed" | Set-Content -Path $StampFile -Encoding Ascii
}

& $PythonExe (Join-Path $ScriptDir "transcribe.py") @args
