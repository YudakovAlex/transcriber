# Transcriber setup for Windows: create venv, install deps, install ffmpeg and add to PATH.
# Run: powershell -ExecutionPolicy Bypass -File setup.ps1
# Or: .\setup.ps1 (if execution policy allows)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$VenvDir = Join-Path $ScriptDir ".venv"
$PythonExe = Join-Path $VenvDir "Scripts\python.exe"
$PipExe = Join-Path $VenvDir "Scripts\pip.exe"
$FfmpegDir = Join-Path $ScriptDir "ffmpeg"
$FfmpegBin = Join-Path $FfmpegDir "bin"

function Test-PathEntry {
    param(
        [string]$PathValue,
        [string]$Entry
    )

    if ([string]::IsNullOrWhiteSpace($PathValue)) { return $false }
    if ([string]::IsNullOrWhiteSpace($Entry)) { return $false }

    $NormalizedEntry = [IO.Path]::GetFullPath($Entry).TrimEnd('\')
    foreach ($item in $PathValue.Split(';')) {
        $candidate = $item.Trim().Trim('"')
        if ([string]::IsNullOrWhiteSpace($candidate)) { continue }
        try {
            $NormalizedCandidate = [IO.Path]::GetFullPath($candidate).TrimEnd('\')
        } catch {
            continue
        }
        if ($NormalizedCandidate.Equals($NormalizedEntry, [StringComparison]::OrdinalIgnoreCase)) {
            return $true
        }
    }

    return $false
}

function Add-UserPathEntry {
    param([string]$Entry)

    $UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if (Test-PathEntry -PathValue $UserPath -Entry $Entry) {
        return $true
    }

    try {
        $NewUserPath = if ([string]::IsNullOrWhiteSpace($UserPath)) { $Entry } else { "$UserPath;$Entry" }
        [Environment]::SetEnvironmentVariable("Path", $NewUserPath, "User")
    } catch {
        Write-Error "Failed to update User PATH with '$Entry': $($_.Exception.Message)"
        return $false
    }

    $UpdatedUserPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if (-not (Test-PathEntry -PathValue $UpdatedUserPath -Entry $Entry)) {
        Write-Error "PATH update did not persist for '$Entry'. Update User PATH manually."
        return $false
    }

    return $true
}

# Resolve a Python 3 launcher across common Windows setups.
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

function Get-PythonMajorMinor {
    param($PythonCmd)
    try {
        return (& $PythonCmd.cmd @($PythonCmd.args + @("-c", "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"))).Trim()
    } catch {}
    return $null
}

# ---- Venv ----
Write-Host "=== Venv ===" -ForegroundColor Cyan
if (-not (Test-Path $PythonExe)) {
    Write-Host "Creating venv..."
    $PythonCmd = Get-Python3Command
    if (-not $PythonCmd) {
        Write-Error "Failed to find Python 3.8+ (3.12 preferred). Install Python 3.12 and ensure one of 'py', 'python', or 'python3' is on PATH."
    }
    $PythonVersion = Get-PythonMajorMinor -PythonCmd $PythonCmd
    if ($PythonVersion -and ($PythonVersion -ne "3.12")) {
        Write-Warning "Using Python $PythonVersion. Python 3.12 is recommended for CUDA-enabled Torch wheels on Windows."
    }
    & $PythonCmd.cmd @($PythonCmd.args + @("-m", "venv", $VenvDir))
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create venv. Install Python 3.12 (or 3.8+) and ensure one of 'py', 'python', or 'python3' is on PATH."
    }
    Write-Host "Venv created." -ForegroundColor Green
} else {
    Write-Host "Venv already exists." -ForegroundColor Green
}

$ReqFile = Join-Path $ScriptDir "requirements.txt"
if (Test-Path $ReqFile) {
    Write-Host "Installing dependencies..."
    & $PythonExe -m pip install --upgrade pip -q
    & $PythonExe -m pip install -r $ReqFile
    if ($LASTEXITCODE -ne 0) { Write-Error "pip install failed." }
    Write-Host "Dependencies installed." -ForegroundColor Green
} else {
    & $PythonExe -m pip install openai-whisper torch tqdm
}

# ---- FFmpeg ----
Write-Host "`n=== FFmpeg ===" -ForegroundColor Cyan
$FfmpegInPath = $false
try {
    $null = Get-Command ffmpeg -ErrorAction Stop
    $FfmpegInPath = $true
} catch {}

if ($FfmpegInPath) {
    Write-Host "ffmpeg is already on PATH." -ForegroundColor Green
} else {
    # Try winget first
    $WingetOk = $false
    try {
        Write-Host "Trying winget install (Gyan.FFmpeg)..."
        winget install --id Gyan.FFmpeg --accept-package-agreements --accept-source-agreements
        if ($LASTEXITCODE -eq 0) {
            $WingetOk = $true
            Write-Host "FFmpeg installed via winget. Restart the terminal or add the install path to PATH if needed." -ForegroundColor Green
        }
    } catch {}

    if (-not $WingetOk) {
        # Fallback: download BtbN build and extract to project\ffmpeg
        Write-Host "Downloading FFmpeg (BtbN build)..."
        $ZipPath = Join-Path $env:TEMP "ffmpeg-win64-gpl.zip"
        $LatestZipUrl = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
        try {
            Invoke-WebRequest -Uri $LatestZipUrl -OutFile $ZipPath -UseBasicParsing
        } catch {
            Write-Warning "Download failed: $_"
            Write-Host "Install FFmpeg manually: winget install Gyan.FFmpeg or download from https://ffmpeg.org/download.html"
            exit 1
        }
        if (-not (Test-Path $ZipPath)) { Write-Error "Download failed." }

        if (Test-Path $FfmpegDir) { Remove-Item $FfmpegDir -Recurse -Force }
        Expand-Archive -Path $ZipPath -DestinationPath $FfmpegDir -Force
        Remove-Item $ZipPath -Force -ErrorAction SilentlyContinue
        # BtbN zip contains a single folder like ffmpeg-master-latest-win64-gpl; move bin up to project\ffmpeg\bin
        $Inner = Get-ChildItem $FfmpegDir -Directory | Select-Object -First 1
        if ($Inner) {
            $InnerBin = Join-Path $Inner.FullName "bin"
            if (Test-Path $InnerBin) {
                New-Item -ItemType Directory -Path $FfmpegBin -Force | Out-Null
                Copy-Item (Join-Path $InnerBin "*") $FfmpegBin -Recurse -Force
                Remove-Item $Inner.FullName -Recurse -Force
            } else {
                Move-Item (Join-Path $Inner.FullName "*") $FfmpegDir -Force
            }
        }
        Write-Host "FFmpeg extracted to $FfmpegDir" -ForegroundColor Green
    }
}

# ---- Register launchers/tools in PATH and verify ----
Write-Host "`n=== PATH ===" -ForegroundColor Cyan
$PathEntriesToEnsure = @($ScriptDir)
if ((Test-Path $FfmpegBin) -and (Test-Path (Join-Path $FfmpegBin "ffmpeg.exe"))) {
    $PathEntriesToEnsure += $FfmpegBin
}

$PathUpdateFailed = $false
foreach ($entry in $PathEntriesToEnsure) {
    if (Add-UserPathEntry -Entry $entry) {
        Write-Host "PATH check ok: $entry" -ForegroundColor Green
    } else {
        $PathUpdateFailed = $true
    }
}

# Update current session PATH so commands can run immediately.
$CurrentPath = $env:Path
foreach ($entry in $PathEntriesToEnsure) {
    if (-not (Test-PathEntry -PathValue $CurrentPath -Entry $entry)) {
        $env:Path = if ([string]::IsNullOrWhiteSpace($env:Path)) { $entry } else { "$env:Path;$entry" }
        $CurrentPath = $env:Path
    }
}

if ($PathUpdateFailed) {
    Write-Error "One or more PATH updates failed. Verify your User PATH and add missing entries manually."
}

# ---- Stamp for transcribe.cmd so it skips install on next run ----
$StampFile = Join-Path $VenvDir ".deps_installed"
"installed" | Set-Content -Path $StampFile -Encoding Ascii -Force

Write-Host "`nSetup complete. Run in a new terminal: transcribe file.mp3" -ForegroundColor Cyan
