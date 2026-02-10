#!/usr/bin/env bash
# Transcriber setup for Linux: create venv, install deps, install ffmpeg and add to PATH.
# Run: ./setup.sh   or   bash setup.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/venv"
PYTHON="${VENV_DIR}/bin/python"
PIP="${VENV_DIR}/bin/pip"
FFMPEG_DIR="$SCRIPT_DIR/ffmpeg"
FFMPEG_BIN="$FFMPEG_DIR/bin"

# ---- Venv ----
echo "=== Venv ==="
if [[ ! -x "$PYTHON" ]]; then
    echo "Creating venv..."
    python3 -m venv "$VENV_DIR"
    echo "Venv created."
else
    echo "Venv already exists."
fi

REQ_FILE="$SCRIPT_DIR/requirements.txt"
if [[ -f "$REQ_FILE" ]]; then
    echo "Installing dependencies..."
    "$PIP" install -q --upgrade pip
    "$PIP" install -r "$REQ_FILE"
    echo "Dependencies installed."
else
    "$PIP" install openai-whisper torch tqdm
fi

# ---- FFmpeg ----
echo ""
echo "=== FFmpeg ==="
if command -v ffmpeg &>/dev/null; then
    echo "ffmpeg is already on PATH."
else
    # Try package manager
    INSTALLED=
    if command -v apt-get &>/dev/null; then
        echo "Installing ffmpeg via apt..."
        sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg && INSTALLED=1
    elif command -v dnf &>/dev/null; then
        echo "Installing ffmpeg via dnf..."
        sudo dnf install -y ffmpeg && INSTALLED=1
    elif command -v yum &>/dev/null; then
        echo "Installing ffmpeg via yum..."
        sudo yum install -y ffmpeg && INSTALLED=1
    elif command -v pacman &>/dev/null; then
        echo "Installing ffmpeg via pacman..."
        sudo pacman -Sy --noconfirm ffmpeg && INSTALLED=1
    elif command -v zypper &>/dev/null; then
        echo "Installing ffmpeg via zypper..."
        sudo zypper install -y ffmpeg && INSTALLED=1
    fi

    if [[ -n "$INSTALLED" ]]; then
        echo "FFmpeg installed. Restart the terminal or run 'hash -r' if needed."
    else
        # Fallback: download static build (BtbN Linux 64-bit GPL)
        echo "Downloading FFmpeg static build..."
        ARCH=$(uname -m)
        case "$ARCH" in
            x86_64)
                FFMPEG_URL="https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz"
                ;;
            aarch64|arm64)
                FFMPEG_URL="https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linuxarm64-gpl.tar.xz"
                ;;
            *)
                echo "Unsupported arch: $ARCH. Install ffmpeg manually (e.g. from your package manager or https://ffmpeg.org/download.html)."
                exit 1
                ;;
        esac
        TARBALL="${TMPDIR:-/tmp}/ffmpeg-linux.tar.xz"
        if command -v wget &>/dev/null; then
            wget -q -O "$TARBALL" "$FFMPEG_URL"
        elif command -v curl &>/dev/null; then
            curl -sL -o "$TARBALL" "$FFMPEG_URL"
        else
            echo "Need wget or curl to download ffmpeg."
            exit 1
        fi
        rm -rf "$FFMPEG_DIR"
        mkdir -p "$FFMPEG_DIR"
        tar -xJf "$TARBALL" -C "$FFMPEG_DIR"
        rm -f "$TARBALL"
        # Tarball contains single dir like ffmpeg-master-latest-linux64-gpl; move bin up
        INNER=$(find "$FFMPEG_DIR" -maxdepth 1 -type d ! -path "$FFMPEG_DIR" | head -1)
        if [[ -d "$INNER/bin" ]]; then
            mv "$INNER/bin" "$FFMPEG_DIR/"
            rm -rf "$INNER"
        fi
        echo "FFmpeg extracted to $FFMPEG_DIR"
    fi
fi

# ---- Register FFmpeg in PATH (if we have a local ffmpeg/bin) ----
if [[ -x "$FFMPEG_BIN/ffmpeg" ]]; then
    if [[ ":$PATH:" != *":$FFMPEG_BIN:"* ]]; then
        export PATH="$FFMPEG_BIN:$PATH"
        SHELL_RC=""
        [[ -f "$HOME/.bashrc" ]] && SHELL_RC="$HOME/.bashrc"
        [[ -f "$HOME/.profile" ]] && SHELL_RC="${SHELL_RC:-$HOME/.profile}"
        if [[ -n "$SHELL_RC" ]]; then
            if ! grep -q "Transcriber ffmpeg" "$SHELL_RC" 2>/dev/null; then
                echo "" >> "$SHELL_RC"
                echo "# Transcriber ffmpeg" >> "$SHELL_RC"
                echo "export PATH=\"$FFMPEG_BIN:\$PATH\"" >> "$SHELL_RC"
                echo "Added $FFMPEG_BIN to PATH in $SHELL_RC"
            fi
        fi
        echo "Added $FFMPEG_BIN to PATH for this session."
    fi
fi

# ---- Stamp for transcribe launcher ----
STAMP_FILE="$VENV_DIR/.deps_installed"
echo "installed" > "$STAMP_FILE"

# Make launcher executable
[[ -f "$SCRIPT_DIR/transcribe" ]] && chmod +x "$SCRIPT_DIR/transcribe"

echo ""
echo "Setup complete. Add this folder to PATH to run: transcribe file.mp3"
echo "  export PATH=\"$SCRIPT_DIR:\$PATH\""
echo "  (or add the above to your .bashrc / .profile)"
