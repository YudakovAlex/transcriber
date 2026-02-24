$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $ScriptDir ".venv"
$PythonExe = Join-Path $VenvDir "Scripts\python.exe"
$StampFile = Join-Path $VenvDir ".deps_installed"

function Get-Python3Command {
    $python312Path = Join-Path $env:LocalAppData "Python\pythoncore-3.12-64\python.exe"
    $preferredCandidates = @(
        @{ cmd = "py"; args = @("-3.12") },
        @{ cmd = "python3.12"; args = @() },
        @{ cmd = $python312Path; args = @() }
    )

    foreach ($candidate in $preferredCandidates) {
        try {
            if (($candidate.cmd -eq $python312Path) -and (-not (Test-Path $candidate.cmd))) { continue }
            if ($candidate.cmd -ne $python312Path) { $null = Get-Command $candidate.cmd -ErrorAction Stop }
            & $candidate.cmd @($candidate.args + @("-c", "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)")) | Out-Null
            if ($LASTEXITCODE -eq 0) { return $candidate }
        } catch {}
    }

    $fallbackCandidates = @(
        @{ cmd = "py"; args = @("-3") },
        @{ cmd = "python"; args = @() },
        @{ cmd = "python3"; args = @() }
    )

    foreach ($candidate in $fallbackCandidates) {
        try {
            $null = Get-Command $candidate.cmd -ErrorAction Stop
            & $candidate.cmd @($candidate.args + @("-c", "import sys; raise SystemExit(0 if sys.version_info >= (3, 8) else 1)")) | Out-Null
            if ($LASTEXITCODE -eq 0) { return $candidate }
        } catch {}
    }

    return $null
}

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Error "ffmpeg not found on PATH. Install it and retry. Example: C:\ffmpeg\bin\ffmpeg.exe"
    exit 1
}

if (-not (Test-Path $PythonExe)) {
    Write-Host "Creating venv..."
    $PythonCmd = Get-Python3Command
    if (-not $PythonCmd) {
        Write-Error "Failed to find Python 3.8+ (3.12 preferred). Install Python 3.12 and ensure one of 'py', 'python', or 'python3' is on PATH."
        exit 1
    }
    & $PythonCmd.cmd @($PythonCmd.args + @("-m", "venv", $VenvDir))
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create venv. Ensure Python 3.8+ is installed."
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
