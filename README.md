# Transcriber

Transcribe audio (MP3, WAV, OGG) to text using [OpenAI Whisper](https://github.com/openai/whisper). Handles long files by splitting into overlapping chunks and merging transcripts.

## Quick Start

```bash
# 1) Setup (once)
setup.cmd          # Windows (or run setup.ps1 in PowerShell)
./setup.sh         # Linux / macOS (chmod +x first if needed)

# 2) Open a new terminal, then run:
transcribe recording.mp3
```

Transcripts are saved next to each source file (for example `recording.mp3` -> `recording.txt`).

## Requirements

- Python 3.8+
- ffmpeg on PATH

## Recent Fixes

- Windows setup now detects Python from `py`, `python`, or `python3` instead of assuming only one launcher exists.
- Windows launchers were aligned to use `.venv` consistently.
- Setup now validates PATH entries at the end and reports clear errors if PATH updates fail.
- `transcribe.py` now prints selected compute device at startup.
- `fp16` is now enabled automatically when CUDA is available and disabled on CPU.

## Installation

### Windows

1. Clone or download this repository.
2. Run setup:
   - `setup.cmd`, or
   - `powershell -ExecutionPolicy Bypass -File setup.ps1`
3. Open a new terminal.
4. Run:

```cmd
transcribe recording.mp3
```

If `transcribe` is not resolved as a command, run directly from the repo:

```powershell
.\transcribe.ps1 recording.mp3
```

### Linux / macOS

1. Clone or download this repository.
2. Run:

```bash
chmod +x setup.sh transcribe
./setup.sh
```

3. Add the project folder to PATH, then open a new terminal.

## Setup Side Effects

Running `setup.ps1` / `setup.cmd` (Windows) or `setup.sh` (Linux/macOS) changes your environment in these ways:

- Creates a virtual environment at `.venv/` in the repo.
- Installs Python dependencies into `.venv/` (including `openai-whisper`, `torch`, and transitive deps).
- Creates `.venv/.deps_installed` as a marker file.
- Verifies `ffmpeg` availability.
- If `ffmpeg` is missing:
  - Windows tries `winget install Gyan.FFmpeg`, then falls back to downloading and extracting FFmpeg into `./ffmpeg/`.
  - Linux/macOS tries package manager install and may fall back to a downloaded static build (from `setup.sh`).
- Updates PATH:
  - Adds the project folder so `transcribe` can be called directly.
  - Adds local `ffmpeg\bin` when a local FFmpeg install exists.
  - Updates current shell PATH for immediate use and updates User PATH for future shells.
- Performs network downloads during dependency install and (if needed) FFmpeg install.

Whisper model files are not downloaded by setup. They are downloaded on first transcription run and cached under your user profile cache directory.

## Estimated Disk Space

Approximate space required (varies by OS, wheel/build selection, and model choice):

- Repo + scripts: `< 20 MB`
- `.venv` with CPU-only Torch: `~1.2 GB to 2.0 GB`
- `.venv` with CUDA-enabled Torch: `~3.5 GB to 6.0 GB`
- Local FFmpeg folder (`./ffmpeg`) if downloaded by setup: `~120 MB to 250 MB`
- Whisper model cache (first use, outside repo):
  - `tiny`: `~75 MB`
  - `base`: `~150 MB`
  - `small`: `~500 MB`
  - `medium`: `~1.5 GB`
  - `large`: `~3.0 GB`

Rule-of-thumb totals:

- CPU setup + `small` model: `~2 GB to 3 GB`
- CUDA setup + `small` model: `~4 GB to 7 GB`
- CUDA setup + `large` model: `~7 GB to 10+ GB`

During installation, temporary download/cache usage can add additional short-lived disk usage.

## GPU / CUDA Notes

At startup, the app prints the selected device:

- `Using device: cuda (...), fp16=True` means GPU acceleration is active.
- `Using device: cpu, fp16=False` means it is running on CPU.

If you have an NVIDIA GPU but still see CPU:

1. Check Torch build:

```powershell
.\.venv\Scripts\python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

2. If version ends with `+cpu`, reinstall a CUDA-enabled Torch build.
3. On this project, Python 3.12 is recommended for best CUDA wheel compatibility on Windows.

Example reinstall flow:

```powershell
Remove-Item -Recurse -Force .\.venv
& "C:\Users\<you>\AppData\Local\Python\pythoncore-3.12-64\python.exe" -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch
.\.venv\Scripts\python -m pip install -r .\requirements.txt
```

## Usage

```text
transcribe <file_or_dir> [<file_or_dir> ...] [options]
```

Options:

- `--model`: `tiny`, `base`, `small`, `medium`, `large` (default: `small`)
- `--language`: language code such as `en` (default: auto-detect)
- `--chunk-duration`: max seconds per chunk (`0` disables chunking, default: `300`)
- `--overlap`: overlap between chunks in seconds (default: `30`)

Examples:

```bash
transcribe recording.mp3
transcribe file1.mp3 file2.wav dir/
transcribe recording.mp3 --model base --language en
transcribe long.mp3 --chunk-duration 180 --overlap 20
transcribe short.mp3 --chunk-duration 0
```

## Tests

```bash
python -m unittest test_transcribe -v
```
