"""Testable worker logic; dependencies are injected by the deployed class."""

import hashlib
import math
import time

from .contracts import IDENTITY_FIELDS, SAMPLE_RATE, validate_request, validate_result


def execute_chunk(request, payload, *, model, read_audio, synchronize=lambda: None,
                  worker_id="unknown", load_seconds=0, sleep=time.sleep):
    started_at, started = time.time(), time.perf_counter()
    identity = {key: request.get(key) for key in IDENTITY_FIELDS}
    execution = {"worker_id": worker_id, "model_load_seconds": load_seconds,
                 "started_at": started_at, "attempts": 0}
    try:
        validate_request(request)
        if time.time() > request.get("deadline_at", float("inf")):
            raise ValueError("Job deadline expired before inference started.")
        read_started = time.perf_counter()
        # Only transient I/O errors retry here. Modal function retries are disabled,
        # avoiding multiplied retries; platform preemption can still replay an input.
        for attempt in range(3):
            execution["attempts"] = attempt + 1
            try:
                data = payload if payload is not None else read_audio(request)
                break
            except OSError:
                if attempt == 2:
                    raise
                sleep(2 ** attempt)
        if not isinstance(data, bytes) or len(data) != request["sample_count"] * 2:
            raise ValueError("Invalid/truncated PCM payload.")
        if hashlib.sha256(data).hexdigest() != request["pcm_sha256"]:
            raise ValueError("Chunk PCM checksum mismatch.")
        import numpy as np
        audio = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
        execution["read_seconds"] = time.perf_counter() - read_started
        synchronize()
        inference_started = time.perf_counter()
        raw = model.transcribe(audio, language=request["requested_language"], fp16=True)
        synchronize()
        execution["inference_seconds"] = time.perf_counter() - inference_started
        duration = request["sample_count"] / SAMPLE_RATE
        for segment in raw.get("segments", []):
            if not all(math.isfinite(float(segment[key])) for key in ("start", "end")):
                raise ValueError("Nonfinite segment timestamp from inference.")
        segments = [{"start": min(duration, max(0.0, float(s["start"]))),
                     "end": min(duration, max(0.0, float(s["end"]))), "text": s["text"]}
                    for s in raw.get("segments", [])]
        result = {**identity, "status": "ok", "text": raw.get("text", ""),
                  "language": raw.get("language", request["requested_language"] or "unknown"),
                  "segments": segments, "execution": execution}
        validate_result(result, request)
    except Exception as error:
        result = {**identity, "status": "error", "error": {
            "type": type(error).__name__, "message": str(error), "retryable": False},
            "execution": execution}
    execution["finished_at"] = time.time()
    execution["execution_seconds"] = time.perf_counter() - started
    return result
