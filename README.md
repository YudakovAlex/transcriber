# Transcriber

Transcribe audio (MP3, WAV, OGG) and MP4 audio tracks to text using [OpenAI Whisper](https://github.com/openai/whisper). Handles long files by splitting into overlapping chunks and merging transcripts.

## Quick Start

```bash
# 1) Setup (once)
setup.cmd          # Windows (or run setup.ps1 in PowerShell)
./setup.sh         # Linux / macOS (chmod +x first if needed)

# 2) Open a new terminal, then run:
transcribe recording.mp3
```

Transcripts are saved next to each source file (for example `recording.mp3` -> `recording.txt`).

MP4 files must contain an audio track. FFmpeg extracts the audio automatically; no manual conversion is needed. MP4 files are also included when transcribing a directory, and `recording.mp4` produces `recording.txt`.

## Requirements

- Python 3.10+ (Python 3.12 recommended for the pinned benchmark/Modal image)
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
transcribe recording.mp4
transcribe file1.mp3 file2.wav dir/
transcribe recording.mp3 --model base --language en
transcribe long.mp3 --chunk-duration 180 --overlap 20
transcribe short.mp3 --chunk-duration 0
```

## Tests

```bash
python -m unittest test_transcribe -v
```

Run the complete offline suite, including recovery and real FFmpeg preparation tests:

```bash
python -m unittest discover -v
```

## Experimental Modal backend

Local transcription remains the default. The optional backend runs the same OpenAI
Whisper engine on independent GPU containers. Its speed, accuracy, and cost gates
have **not yet been validated with real recordings**; no deployed benchmark results
are included in this repository. See [benchmark and rollout guide](docs/modal-backend.md).

Install the optional client in the environment used by your launcher:

```bash
python -m pip install -r requirements-modal.txt
python -m modal setup
```

Deployment and weight provisioning are explicit operations that can incur charges:

```bash
modal deploy modal_app.py
modal run modal_app.py --model small
```

Then opt in for a recording:

```bash
transcribe recording.mp4 --backend modal --modal-workers 2 --model small --language en
```

The CLI extracts MP4 audio locally and uploads only mono 16 kHz PCM. One upload is
shared by all chunks. The worker loads the selected model once per container from
a separately provisioned, checksum-verified weights cache. `tiny`, `base`, `small`,
`medium`, and `large` are supported; provision each model before selecting it.
`large` is pinned to large-v3. The initial cost target applies to `small`; larger
models use a 16 GiB host-memory profile and need separate qualification.

`--modal-workers` accepts 1–4 and defaults to 2. `--chunk-duration 0` sends one
request and provides no chunk parallelism. Modal-only options are rejected when
`--backend local` is selected. Model and language defaults otherwise stay the same.
Each file produces its usual adjacent `.txt`, written atomically only on success.

Modal jobs print a checkpoint directory under `.transcriber-jobs/` next to the
recording. Ctrl+C saves completed chunks and attempts to cancel outstanding calls.
Resume with the original input and transcription options:

```bash
transcribe recording.mp4 --backend modal --model small --language en --resume ".transcriber-jobs/JOB_DIRECTORY"
```

Source content, normalized audio, model checksum, decoding settings, chunk boundaries,
and implementation version are checked before results are reused. A changed worker
count is allowed. There is no automatic fallback to local inference. A killed client
can leave up to the configured number of calls running; resume reconciles their IDs.
Remote results expire after seven days. Completed local checkpoints remain reusable.

Successful jobs delete their uploaded audio. Interrupted uploads expire after seven
days and are removed by a daily CPU cleanup function (there can be up to one additional
day before deletion). Local PCM and transcript checkpoints are retained until you
delete the job directory. The checkpoint directory contains recording/transcript data;
it is ignored by Git. The model cache persists separately.

Authentication uses the active Modal profile, or `MODAL_TOKEN_ID` and
`MODAL_TOKEN_SECRET`. `MODAL_ENVIRONMENT` selects an environment and
`TRANSCRIBER_MODAL_APP` can select a separately named deployment. Do not put tokens in
source files. Normal CLI use never deploys an app or downloads weights on GPU startup.
