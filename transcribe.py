import argparse
import shutil
import sys
import threading
import time
from pathlib import Path

import whisper


def ensure_ffmpeg_available() -> None:
    if shutil.which("ffmpeg") is None:
        raise SystemExit(
            "ffmpeg not found. Install it and ensure it's on PATH, "
            "then retry (Whisper needs ffmpeg to read audio files)."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe one or more audio files using Whisper."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Path(s) to audio files or a directory containing audio files.",
    )
    parser.add_argument(
        "--model",
        default="small",
        help="Whisper model name (tiny, base, small, medium, large).",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Optional language code (e.g. en). If omitted, auto-detect.",
    )
    return parser.parse_args()


SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".ogg"}


class Spinner:
    def __init__(self, message: str) -> None:
        self._message = message
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)

    def __enter__(self) -> "Spinner":
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop_event.set()
        self._thread.join(timeout=1)
        sys.stdout.write("\r" + " " * 80 + "\r")
        sys.stdout.flush()

    def _spin(self) -> None:
        frames = "|/-\\"
        idx = 0
        while not self._stop_event.is_set():
            frame = frames[idx % len(frames)]
            sys.stdout.write(f"\r{frame} {self._message}")
            sys.stdout.flush()
            idx += 1
            time.sleep(0.2)


def collect_audio_paths(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            for ext in sorted(SUPPORTED_EXTENSIONS):
                paths.extend(sorted(p.glob(f"*{ext}")))
        else:
            paths.append(p)
    return paths


def main() -> int:
    args = parse_args()
    ensure_ffmpeg_available()

    audio_paths = collect_audio_paths(args.inputs)
    if not audio_paths:
        raise SystemExit("No audio files found in the provided inputs.")

    model = whisper.load_model(args.model)

    for audio_path in audio_paths:
        if not audio_path.is_file():
            print(f"Skipping missing file: {audio_path}")
            continue
        if audio_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            print(f"Skipping unsupported file: {audio_path}")
            continue

        print(f"Transcribing: {audio_path}")
        with Spinner("Working..."):
            result = model.transcribe(
                str(audio_path),
                language=args.language,
                fp16=False,
            )
        out_path = audio_path.with_suffix(".txt")
        out_path.write_text(result["text"].strip() + "\n", encoding="utf-8")
        print(f"Transcript: {out_path}")

    return 0


# python transcribe.py "C:\Users\Alexander_Yudakov\Videos\2026-02-09 13-06-02.mp3"

if __name__ == "__main__":
    raise SystemExit(main())
