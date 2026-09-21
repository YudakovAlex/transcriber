"""Deploy explicitly with `modal deploy modal_app.py`; never deployed by the CLI."""

import json
import os
import re
import shutil
import time
import urllib.request
import uuid
from pathlib import Path

import modal

from transcription.audio import file_sha256, read_pcm
from transcription.backends.modal import APP_NAME, AUDIO_VOLUME, WEIGHTS_VOLUME
from transcription.contracts import (IMPLEMENTATION_VERSION, MODEL_HASHES, TORCH_VERSION,
                                     WHISPER_VERSION)
from transcription.worker import execute_chunk

app = modal.App(os.environ.get("TRANSCRIBER_MODAL_APP", APP_NAME))
weights = modal.Volume.from_name(WEIGHTS_VOLUME, create_if_missing=True)
audio_volume = modal.Volume.from_name(AUDIO_VOLUME, create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("ffmpeg")
         .pip_install(f"torch=={TORCH_VERSION}", index_url="https://download.pytorch.org/whl/cu128")
         .pip_install(f"openai-whisper=={WHISPER_VERSION}", "numpy==2.2.6", "tqdm==4.67.1")
         .add_local_python_source("transcription"))
WEIGHT_ROOT, AUDIO_ROOT = Path("/weights"), Path("/audio")


def model_path(model):
    if model not in MODEL_HASHES:
        raise ValueError(f"Unsupported model: {model}")
    return WEIGHT_ROOT / f"{MODEL_HASHES[model]}.pt"


@app.function(image=image, volumes={"/weights": weights}, cpu=1, memory=1024,
              max_containers=1, timeout=1800)
def provision(model: str = "small"):
    """Serial CPU-only download; atomic publication prevents concurrent partial reads."""
    weights.reload()
    path = model_path(model)
    if path.exists() and file_sha256(path) == MODEL_HASHES[model]:
        return {"model": model, "cached": True}
    filename = "large-v3" if model == "large" else model
    url = f"https://openaipublic.azureedge.net/main/whisper/models/{MODEL_HASHES[model]}/{filename}.pt"
    temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        if file_sha256(temporary) != MODEL_HASHES[model]:
            raise ValueError("Downloaded model checksum mismatch.")
        temporary.replace(path)
        weights.commit()
    finally:
        temporary.unlink(missing_ok=True)
    return {"model": model, "cached": False}


def job_directory(job_id):
    if not isinstance(job_id, str) or not re.fullmatch("[0-9a-f]{32}", job_id):
        raise ValueError("Invalid job ID.")
    return AUDIO_ROOT / "jobs" / job_id


@app.function(image=image, volumes={"/weights": weights, "/audio": audio_volume},
              cpu=1, memory=1024, timeout=300)
def manage_audio(action: str, metadata: dict):
    try:
        directory = job_directory(metadata["job_id"])
        audio_volume.reload()
        if action == "delete":
            if directory.exists():
                shutil.rmtree(directory)
                audio_volume.commit()
            return {"ready": False}
        if action not in ("inspect", "verify"):
            raise ValueError("Unknown audio operation.")
        weights.reload()
        if not model_path(metadata["model"]).is_file():
            raise ValueError("Model cache is missing. Run: modal run modal_app.py --model " + metadata["model"])
        from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE
        language = metadata.get("language")
        if language is not None and language.lower() not in LANGUAGES and language.lower() not in TO_LANGUAGE_CODE:
            raise ValueError(f"Unsupported language: {language}")
        pcm = directory / "audio.pcm"
        ready = (pcm.is_file() and pcm.stat().st_size == metadata["sample_count"] * 2
                 and file_sha256(pcm) == metadata["recording_id"])
        if ready:
            (directory / "lease.json").write_text(json.dumps({"expires_at": time.time() + 7 * 86400}))
            audio_volume.commit()
        return {"ready": ready, "implementation_version": IMPLEMENTATION_VERSION}
    except (ValueError, KeyError) as error:
        return {"ready": False, "error": str(error)}


@app.cls(image=image, gpu="L4", cpu=(2, 2), memory=(8192, 8192),
         volumes={"/weights": weights, "/audio": audio_volume},
         max_containers=4, min_containers=0, buffer_containers=0, scaledown_window=2,
         startup_timeout=180, timeout=600, retries=0)
class WhisperWorker:
    model: str = modal.parameter(default="small")
    pool: str = modal.parameter(default="default")  # Benchmark-only pool isolation.

    @modal.enter()
    def load(self):
        import torch
        import whisper
        started = time.perf_counter()
        self.worker_id = os.environ.get("MODAL_TASK_ID", uuid.uuid4().hex)
        self.startup_error = None
        try:
            weights.reload()
            path = model_path(self.model)
            if not path.exists() or file_sha256(path) != MODEL_HASHES[self.model]:
                raise ValueError("Model cache missing/corrupt; run the CPU provisioning command.")
            self.network = whisper.load_model(str(path), device="cuda")
            torch.cuda.synchronize()
        except Exception as error:
            # Return a terminal error rather than entering an indefinite startup crash loop.
            self.startup_error = f"{type(error).__name__}: {error}"
        self.load_seconds = time.perf_counter() - started
        self.calls_served = 0

    @modal.method()
    def warmup(self, hold_seconds: int = 30):
        """Explicit benchmark-only warmup; normal CLI calls never invoke it."""
        if self.startup_error:
            raise ValueError(self.startup_error)
        time.sleep(min(60, max(0, hold_seconds)))
        return {"worker_id": self.worker_id, "model_load_seconds": self.load_seconds}

    @modal.method()
    def transcribe(self, request: dict, payload: bytes | None = None):
        import torch
        from transcription.contracts import IDENTITY_FIELDS
        if self.startup_error or request.get("model") != self.model:
            return {**{key: request.get(key) for key in IDENTITY_FIELDS}, "status": "error",
                    "error": {"type": "startup", "message": self.startup_error or "Model mismatch"}}

        def read_audio(chunk):
            expected = f"jobs/{chunk['job_id']}/audio.pcm"
            if chunk.get("audio_ref") != expected:
                raise ValueError("Invalid audio reference.")
            audio_volume.reload()
            return read_pcm(job_directory(chunk["job_id"]) / "audio.pcm",
                            chunk["start_sample"], chunk["sample_count"])

        result = execute_chunk(request, payload, model=self.network, read_audio=read_audio,
                               synchronize=torch.cuda.synchronize, worker_id=self.worker_id,
                               load_seconds=self.load_seconds)
        result["execution"]["container_call_index"] = self.calls_served
        result["execution"]["gpu"] = torch.cuda.get_device_name(0)
        self.calls_served += 1
        return result


@app.function(image=image, volumes={"/audio": audio_volume}, cpu=0.25, memory=256,
              schedule=modal.Period(days=1), timeout=300, max_containers=1)
def cleanup_expired():
    audio_volume.reload()
    root = AUDIO_ROOT / "jobs"
    removed = 0
    for directory in root.iterdir() if root.exists() else []:
        if not re.fullmatch("[0-9a-f]{32}", directory.name) or not directory.is_dir():
            continue
        lease = directory / "lease.json"
        try:
            expires = json.loads(lease.read_text())["expires_at"]
        except (OSError, ValueError, KeyError):
            expires = directory.stat().st_mtime + 7 * 86400
        if expires < time.time():
            shutil.rmtree(directory)
            removed += 1
    audio_volume.commit()
    return removed


@app.local_entrypoint()
def main(model: str = "small"):
    print(provision.remote(model))
