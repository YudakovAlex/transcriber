"""Optional SDK adapter. Importing this module does not authenticate or deploy."""

import importlib
import io
import json
import os
import time

from ..audio import read_pcm
from ..pipeline import CallExpired

APP_NAME = "transcriber-whisper-v1"
AUDIO_VOLUME = "transcriber-audio-v1"
WEIGHTS_VOLUME = "transcriber-weights-v1"


class ModalBackend:
    def __init__(self, model="small", workers=2, gpu="L4", transfer="volume", pool="default",
                 scaledown_window=2):
        if workers not in (1, 2, 3, 4) or gpu not in ("L4", "T4") or transfer not in ("volume", "direct"):
            raise ValueError("Unsupported Modal worker, GPU, or transfer configuration.")
        try:
            self.sdk = importlib.import_module("modal")
        except ImportError as error:
            raise RuntimeError("Install the optional backend: python -m pip install -r requirements-modal.txt") from error
        self.model, self.workers, self.gpu, self.transfer = model, workers, gpu, transfer
        self.app_name = os.environ.get("TRANSCRIBER_MODAL_APP", APP_NAME)
        self.environment = os.environ.get("MODAL_ENVIRONMENT")
        lookup = {"environment_name": self.environment}
        try:
            self.control = self.sdk.Function.from_name(self.app_name, "manage_audio", **lookup)
            self.control.hydrate()
            self.volume = self.sdk.Volume.from_name(AUDIO_VOLUME, **lookup)
            worker = self.sdk.Cls.from_name(self.app_name, "WhisperWorker", **lookup)
            worker.hydrate()
            memory = 16384 if model in ("medium", "large") else 8192
            self.worker = worker.with_options(gpu=gpu, memory=(memory, memory),
                                              max_containers=workers, scaledown_window=scaledown_window)(
                                                  model=model, pool=pool)
        except Exception as error:
            raise RuntimeError("Cannot connect to Modal. Run 'python -m modal setup', select the correct "
                               "workspace/environment, and deploy modal_app.py. " + str(error)) from error

    def prepare(self, pcm_path, manifest):
        self.pcm_path = pcm_path
        self.manifest = manifest
        metadata = {key: manifest[key] for key in ("job_id", "recording_id", "sample_count")}
        metadata["model"] = self.model
        metadata["language"] = manifest["config"]["language"]
        self.metadata = metadata
        # CPU preflight checks weights, versions, language and existing upload before any GPU call.
        status = self.control.remote("inspect", metadata)
        if status.get("error"):
            raise RuntimeError(status["error"])
        from ..contracts import IMPLEMENTATION_VERSION
        if status.get("implementation_version") != IMPLEMENTATION_VERSION:
            raise RuntimeError("Worker version differs from client; redeploy modal_app.py.")
        if self.transfer == "direct":
            return {"strategy": "direct", "bytes": 0, "reused": False,
                    "note": "Direct payload transfer is included in submission round-trip times."}
        uploaded = 0
        if not status["ready"]:
            lease = json.dumps({"expires_at": time.time() + 7 * 86400}).encode()
            with self.volume.batch_upload(force=True) as upload:
                upload.put_file(pcm_path, f"/jobs/{manifest['job_id']}/audio.pcm")
                upload.put_file(io.BytesIO(lease), f"/jobs/{manifest['job_id']}/lease.json")
            uploaded = manifest["bytes"]
        checked = self.control.remote("verify", metadata)
        if not checked.get("ready"):
            raise RuntimeError(checked.get("error", "Uploaded PCM checksum did not match."))
        return {"strategy": "volume", "bytes": uploaded, "reused": uploaded == 0}

    def submit(self, request):
        payload = None
        if self.transfer == "direct":
            payload = read_pcm(self.pcm_path, request["start_sample"], request["sample_count"])
        call = self.worker.transcribe.spawn(request, payload)
        return call.object_id

    def poll(self, call_id):
        try:
            return self.sdk.FunctionCall.from_id(call_id).get(timeout=0)
        except self.sdk.exception.OutputExpiredError as error:
            raise CallExpired() from error
        except self.sdk.exception.FunctionTimeoutError as error:
            return {"status": "error", "error": {"type": "timeout", "message": str(error)}}
        except TimeoutError:
            return None
        except Exception as error:
            # Keep the call ID on transport failures; don't turn unknown status into a resubmit.
            raise RuntimeError(f"Unable to retrieve Modal call {call_id}; resume to reconcile it: {error}") from error

    def cancel(self, call_id):
        self.sdk.FunctionCall.from_id(call_id).cancel()

    def cleanup(self, manifest):
        if self.transfer == "volume":
            result = self.control.remote("delete", self.metadata)
            if result.get("error"):
                raise RuntimeError(result["error"])
