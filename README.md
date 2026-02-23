# Transcriber

Transcribe audio (MP3, WAV, OGG) to text using [OpenAI Whisper](https://github.com/openai/whisper). Handles long files by splitting into overlapping chunks and merging transcripts. One command from the shell, no GUI.

---

## Quick start

```bash
# 1. Setup (once): creates venv, installs deps, installs ffmpeg if needed
setup.cmd          # Windows (or run setup.ps1 in PowerShell)
./setup.sh         # Linux / macOS (chmod +x first if needed)

# 2. Add this folder to your PATH, then in a new terminal:
transcribe recording.mp3
```

Transcripts are saved next to each file (e.g. `recording.mp3` → `recording.txt`).

---

## Requirements

- **Python 3** (3.8+)
- **ffmpeg** on PATH (Whisper uses it to decode audio). The setup scripts can install it for you.

---

## Installation

### Windows

1. Clone or download this repository.
2. Run **setup** (creates a virtualenv, installs Python deps, and optionally installs ffmpeg):
   - Double‑click `setup.cmd`, or
   - In Command Prompt: `setup.cmd`
   - In PowerShell: `powershell -ExecutionPolicy Bypass -File setup.ps1`
3. Add the **Transcriber folder** to your [system or user PATH](https://docs.microsoft.com/en-us/windows/win32/procthread/environment-variables).
4. Open a **new** terminal (so PATH is updated) and run:
   ```cmd
   transcribe recording.mp3
   ```

### Linux / macOS

1. Clone or download this repository.
2. Run **setup**:
   ```bash
   chmod +x setup.sh transcribe
   ./setup.sh
   ```
   This creates a venv, installs dependencies, and tries to install ffmpeg via your package manager (apt, dnf, yum, pacman, zypper) or downloads a static build.
3. Add the project folder to PATH (e.g. in `~/.bashrc` or `~/.profile`):
   ```bash
   export PATH="/path/to/Transcriber:$PATH"
   ```
4. In a new terminal:
   ```bash
   transcribe recording.mp3
   ```

---

## Usage

```text
transcribe <file_or_dir> [<file_or_dir> ...] [options]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Whisper model: `tiny`, `base`, `small`, `medium`, `large` (larger = more accurate, slower) | small |
| `--language` | Two-letter code (e.g. `en`); omit to auto-detect | — |
| `--chunk-duration` | Max seconds per chunk; `0` = no chunking | 300 |
| `--overlap` | Overlap between chunks (seconds) | 30 |

**Examples:**

```bash
transcribe recording.mp3
transcribe file1.mp3 file2.wav dir/
transcribe recording.mp3 --model base --language en
transcribe long.mp3 --chunk-duration 180 --overlap 20
transcribe short.mp3 --chunk-duration 0
```

- **Input:** One or more files or directories. Directories are scanned for `.mp3`, `.wav`, `.ogg`.
- **Output:** For each audio file, a `.txt` with the same base name in the same folder.

---

## Features

- **Formats:** MP3, WAV, OGG
- **Long files:** Optional chunking with overlap to reduce memory use; progress bar (tqdm) when processing multiple chunks
- **Batch:** Process multiple files or a whole directory in one run
- **Portable:** Use the launchers (`transcribe`, `transcribe.cmd`, `transcribe.ps1`) so the script runs with the project venv and deps without activating it yourself

---

## Running without the launcher

With the project venv activated (or its `python` on PATH):

```bash
python transcribe.py file.mp3
```

---

## Tests

From the project root with the venv activated:

```bash
python -m unittest test_transcribe -v
```

---

## Project layout

```text
Transcriber/
├── transcribe.py       # Main script
├── transcribe.cmd      # Windows launcher (CMD)
├── transcribe.ps1      # Windows launcher (PowerShell)
├── transcribe          # Linux/macOS launcher
├── setup.cmd           # Windows setup (invokes setup.ps1)
├── setup.ps1           # Windows setup
├── setup.sh            # Linux/macOS setup
├── requirements.txt
├── test_transcribe.py
├── .gitignore
└── README.md
```

After setup you’ll also have a `venv/` directory and, if the script installed it, an `ffmpeg/` directory (both are in `.gitignore`).

---

## Troubleshooting

- **“ffmpeg not found”**  
  Install ffmpeg and add it to PATH, or run the setup script again (it can install ffmpeg). On Windows, a common location is `C:\ffmpeg\bin`.

- **“No module named 'whisper'”**  
  Run the setup script for your OS so the virtualenv and dependencies are created. Then use the `transcribe` launcher (which uses that venv), or activate the venv and run `python transcribe.py`.

- **First run is slow**  
  Whisper downloads the model (e.g. “small”) on first use. Later runs reuse the cached model.

- **PATH not updated**  
  After adding the Transcriber folder to PATH, open a **new** terminal so the change is picked up.
