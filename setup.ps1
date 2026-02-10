# Transcriber setup for Windows: create venv, install deps, install ffmpeg and add to PATH.
# Run: powershell -ExecutionPolicy Bypass -File setup.ps1
# Or: .\setup.ps1 (if execution policy allows)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$VenvDir = Join-Path $ScriptDir "venv"
$PythonExe = Join-Path $VenvDir "Scripts\python.exe"
$PipExe = Join-Path $VenvDir "Scripts\pip.exe"
$FfmpegDir = Join-Path $ScriptDir "ffmpeg"
$FfmpegBin = Join-Path $FfmpegDir "bin"

# ---- Venv ----
Write-Host "=== Venv ===" -ForegroundColor Cyan
if (-not (Test-Path $PythonExe)) {
    Write-Host "Creating venv..."
    py -3 -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create venv. Install Python 3 and ensure 'py' is on PATH."
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

# ---- Register FFmpeg in PATH (if we have a local ffmpeg\bin) ----
if ((Test-Path $FfmpegBin) -and (Test-Path (Join-Path $FfmpegBin "ffmpeg.exe"))) {
    $UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if ($UserPath -notlike "*$FfmpegBin*") {
        [Environment]::SetEnvironmentVariable("Path", "$UserPath;$FfmpegBin", "User")
        Write-Host "Added $FfmpegBin to User PATH. Restart the terminal for it to take effect." -ForegroundColor Green
    } else {
        Write-Host "FFmpeg bin already in User PATH." -ForegroundColor Green
    }
}

# ---- Stamp for transcribe.cmd so it skips install on next run ----
$StampFile = Join-Path $VenvDir ".deps_installed"
"installed" | Set-Content -Path $StampFile -Encoding Ascii -Force

Write-Host "`nSetup complete. Add this folder to PATH to run: transcribe file.mp3" -ForegroundColor Cyan
