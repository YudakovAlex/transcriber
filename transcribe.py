"""Transcribe local audio or opt in to the experimental Modal GPU backend."""

import argparse
import importlib
import math
import shutil
import sys
from pathlib import Path

from transcription.audio import chunk_audio
from transcription.contracts import SAMPLE_RATE
from transcription.merge import merge_chunk_results
from transcription.progress import Spinner

DEFAULT_MODEL = "small"
DEFAULT_LANGUAGE = None
DEFAULT_CHUNK_DURATION_SEC = 300.0
DEFAULT_OVERLAP_SEC = 30.0
SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".ogg", ".mp4"}


def __getattr__(name):
    # Backwards-compatible access for callers/tests of the original script.
    # Normal Modal and --help invocations do not import the local inference stack.
    if name in {"torch", "whisper"}:
        return importlib.import_module(name)
    if name == "tqdm":
        return importlib.import_module("tqdm").tqdm
    raise AttributeError(name)


def ensure_ffmpeg_available():
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found. Install it and ensure it is on PATH, then retry.")


def parse_args():
    parser = argparse.ArgumentParser(description="Transcribe MP3, WAV, OGG, or MP4 audio using Whisper.")
    parser.add_argument("inputs", nargs="+", help="Audio/video files or directories.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--language", default=DEFAULT_LANGUAGE, help="Language code; omit for per-chunk detection.")
    parser.add_argument("--chunk-duration", type=float, default=DEFAULT_CHUNK_DURATION_SEC,
                        metavar="SECONDS", help="Max chunk seconds; 0 disables chunking. Default: 300.")
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP_SEC,
                        metavar="SECONDS", help="Overlap seconds. Default: 30.")
    parser.add_argument("--backend", choices=("local", "modal"), default="local")
    parser.add_argument("--modal-workers", type=int, choices=(1, 2, 3, 4), default=None,
                        help="Outstanding Modal calls (default: 2). Modal backend only.")
    parser.add_argument("--resume", type=Path, metavar="JOB_DIRECTORY", help="Resume one Modal recording with the same options.")
    args = parser.parse_args()
    if args.backend == "local" and (args.modal_workers is not None or args.resume is not None):
        parser.error("--modal-workers and --resume require --backend modal")
    if not math.isfinite(args.chunk_duration) or args.chunk_duration < 0 or not math.isfinite(args.overlap):
        parser.error("chunk duration must be finite and nonnegative; overlap must be finite")
    args.modal_workers = args.modal_workers or 2
    return args


def collect_audio_paths(inputs):
    paths = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            for extension in sorted(SUPPORTED_EXTENSIONS):
                paths.extend(sorted(path.glob(f"*{extension}")))
        else:
            paths.append(path)
    return paths


def main():
    args = parse_args()
    ensure_ffmpeg_available()
    paths = collect_audio_paths(args.inputs)
    if not paths:
        raise SystemExit("No audio files found in the provided inputs.")
    if args.backend == "modal" and args.resume and len(paths) != 1:
        raise SystemExit("--resume requires exactly one recording.")
    backend = None
    for source in paths:
        if not source.is_file():
            print(f"Skipping missing file: {source}")
            continue
        if source.suffix.lower() not in SUPPORTED_EXTENSIONS:
            print(f"Skipping unsupported file: {source}")
            continue
        print(f"Transcribing: {source}")
        try:
            if args.backend == "modal":
                from transcription.pipeline import transcribe_remote
                transcribe_remote(source, model=args.model, language=args.language,
                                  chunk_duration=args.chunk_duration, overlap=args.overlap,
                                  workers=args.modal_workers, resume=args.resume)
            else:
                if backend is None:
                    from transcription.backends.local import LocalBackend
                    backend = LocalBackend(args.model)
                backend.transcribe_file(source, args.language, args.chunk_duration, args.overlap)
        except KeyboardInterrupt:
            print("Interrupted. Completed Modal chunks are saved in the printed job directory.", file=sys.stderr)
            return 130
        except (ValueError, RuntimeError, OSError) as error:
            print(f"Transcription failed: {error}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
