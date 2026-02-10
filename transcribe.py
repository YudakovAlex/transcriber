"""
Transcribe audio files (MP3, WAV, OGG) using Whisper. Requires ffmpeg on PATH.

CLI usage:
  transcribe <file_or_dir> [<file_or_dir> ...] [options]
  transcribe recording.mp3
  transcribe file1.mp3 file2.wav folder/
  transcribe recording.mp3 --model base --language en
  transcribe long.mp3 --chunk-duration 180 --overlap 20
  transcribe short.mp3 --chunk-duration 0

Options:
  --model           Whisper model (tiny, base, small, medium, large). Default: small
  --language        Language code, e.g. en; omit to auto-detect
  --chunk-duration  Max seconds per chunk; 0 = no chunking. Default: 300
  --overlap         Overlap between chunks (seconds). Default: 30
"""

import argparse
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np
import whisper
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Defaults (change here or override via CLI)
# -----------------------------------------------------------------------------
SAMPLE_RATE = 16000  # Whisper's fixed input sample rate
DEFAULT_MODEL = "small"  # Whisper model: tiny, base, small, medium, large
DEFAULT_LANGUAGE = None  # e.g. "en"; None = auto-detect
DEFAULT_CHUNK_DURATION_SEC = 300.0  # Max seconds per chunk; 0 = no chunking
DEFAULT_OVERLAP_SEC = 30.0  # Overlap between consecutive chunks (seconds)
SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".ogg"}


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
        default=DEFAULT_MODEL,
        help=f"Whisper model name (tiny, base, small, medium, large). Default: {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        help="Optional language code (e.g. en). If omitted, auto-detect.",
    )
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=DEFAULT_CHUNK_DURATION_SEC,
        metavar="SECONDS",
        help=f"Max duration per chunk in seconds. Use 0 to disable. Default: {DEFAULT_CHUNK_DURATION_SEC}.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=DEFAULT_OVERLAP_SEC,
        metavar="SECONDS",
        help=f"Overlap between consecutive chunks in seconds. Default: {DEFAULT_OVERLAP_SEC}.",
    )
    return parser.parse_args()


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


def chunk_audio(
    audio: np.ndarray,
    chunk_duration_sec: float=300.0,
    overlap_sec: float=30.0,
) -> list[np.ndarray]:
    """Split audio into overlapping chunks. Each chunk is chunk_duration_sec long; consecutive chunks overlap by overlap_sec."""
    total_samples = len(audio)
    chunk_samples = int(chunk_duration_sec * SAMPLE_RATE)
    step_samples = int((chunk_duration_sec - overlap_sec) * SAMPLE_RATE)
    if step_samples <= 0 or chunk_samples <= 0:
        return [audio]
    chunks = []
    start = 0
    while start < total_samples:
        end = min(start + chunk_samples, total_samples)
        chunks.append(audio[start:end])
        if end >= total_samples:
            break
        start += step_samples
    return chunks


def merge_chunk_results(
    chunk_results: list[dict],
    overlap_sec: float,
) -> str:
    """Merge transcription results from overlapping chunks, trimming overlap to avoid repetition."""
    if not chunk_results:
        return ""
    if len(chunk_results) == 1:
        return (chunk_results[0].get("text") or "").strip()

    parts = []
    for i, result in enumerate(chunk_results):
        segments = result.get("segments") or []
        text = result.get("text") or ""
        if i == 0:
            parts.append(text.strip())
            continue
        # From second chunk onward: drop segments that fall entirely inside the overlap
        segment_texts = []
        for s in segments:
            if s.get("end", 0) <= overlap_sec:
                continue
            segment_texts.append((s.get("text") or "").strip())
        chunk_tail = " ".join(segment_texts).strip()
        if chunk_tail:
            parts.append(chunk_tail)
    return " ".join(parts).strip()


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
        out_path = audio_path.with_suffix(".txt")

        chunk_sec = args.chunk_duration
        overlap_sec = max(0.0, min(args.overlap, chunk_sec - 1.0) if chunk_sec > 0 else 0.0)

        if chunk_sec > 0:
            audio = whisper.load_audio(str(audio_path))
            duration_sec = len(audio) / SAMPLE_RATE
            chunks = chunk_audio(audio, chunk_sec, overlap_sec) if duration_sec > chunk_sec else [audio]
            num_chunks = len(chunks)
            if num_chunks > 1:
                print(f"  Splitting into {num_chunks} chunks (~{chunk_sec}s each, {overlap_sec}s overlap)")
            chunk_results = []
            if num_chunks > 1:
                for chunk in tqdm(chunks, desc="Chunks", unit="chunk"):
                    chunk_results.append(
                        model.transcribe(
                            chunk,
                            language=args.language,
                            fp16=False,
                        )
                    )
            else:
                with Spinner("Working..."):
                    chunk_results.append(
                        model.transcribe(
                            chunks[0],
                            language=args.language,
                            fp16=False,
                        )
                    )
            text = merge_chunk_results(chunk_results, overlap_sec)
        else:
            with Spinner("Working..."):
                result = model.transcribe(
                    str(audio_path),
                    language=args.language,
                    fp16=False,
                )
            text = (result.get("text") or "").strip()

        out_path.write_text(text + "\n", encoding="utf-8")
        print(f"Transcript: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
