"""Versioned wire contracts. No inference or Modal dependencies."""

import hashlib
import json
import math
import re

SCHEMA_VERSION = 1
IMPLEMENTATION_VERSION = "whisper-chunks-v1"
SAMPLE_RATE = 16000
WHISPER_VERSION = "20250625"
TORCH_VERSION = "2.10.0"
MODEL_HASHES = {
    "tiny": "65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9",
    "base": "ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e",
    "small": "9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794",
    "medium": "345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1",
    "large": "e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb",
}
IDENTITY_FIELDS = (
    "schema_version", "implementation_version", "job_id", "recording_id",
    "config_fingerprint", "chunk_index", "chunk_count", "start_sample",
    "sample_count", "start_offset", "sample_rate", "model", "checkpoint_sha256",
    "requested_language", "pcm_sha256",
)


def fingerprint(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def configuration(model, language, chunk_duration, overlap):
    if model not in MODEL_HASHES:
        raise ValueError(f"Modal supports {', '.join(MODEL_HASHES)}; got {model!r}.")
    return {
        "schema_version": SCHEMA_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "model": model, "checkpoint_sha256": MODEL_HASHES[model],
        "language": language, "chunk_duration": float(chunk_duration), "overlap": float(overlap),
        "sample_rate": SAMPLE_RATE, "whisper_version": WHISPER_VERSION,
        "torch_version": TORCH_VERSION, "fp16": True,
        "language_policy": "per-chunk", "merge": "legacy-segment-end",
        "decoding": {"condition_on_previous_text": True, "word_timestamps": False,
                     "temperature": [0, 0.2, 0.4, 0.6, 0.8, 1.0]},
    }


def validate_request(request):
    if not isinstance(request, dict):
        raise ValueError("Chunk request must be an object.")
    for key in IDENTITY_FIELDS:
        if key not in request:
            raise ValueError(f"Missing chunk metadata: {key}")
    if request["schema_version"] != SCHEMA_VERSION or request["implementation_version"] != IMPLEMENTATION_VERSION:
        raise ValueError("Client/worker version mismatch; redeploy or start a new job.")
    for key, length in (("job_id", 32), ("recording_id", 64), ("config_fingerprint", 64), ("pcm_sha256", 64)):
        if not isinstance(request[key], str) or not re.fullmatch(f"[0-9a-f]{{{length}}}", request[key]):
            raise ValueError(f"Invalid {key}")
    for key in ("chunk_index", "start_sample", "sample_count", "chunk_count"):
        if type(request[key]) is not int or request[key] < 0:
            raise ValueError(f"Invalid {key}")
    if not request["sample_count"] or request["chunk_index"] >= request["chunk_count"]:
        raise ValueError("Empty chunk or invalid chunk index.")
    if request["sample_rate"] != SAMPLE_RATE or request["start_offset"] != request["start_sample"] / SAMPLE_RATE:
        raise ValueError("Invalid sample rate or start offset.")
    if MODEL_HASHES.get(request["model"]) != request["checkpoint_sha256"]:
        raise ValueError("Unknown model/checkpoint.")
    if request["requested_language"] is not None and not isinstance(request["requested_language"], str):
        raise ValueError("Language must be a code or null.")
    return request


def validate_result(result, request):
    validate_request(request)
    if not isinstance(result, dict):
        raise ValueError("Chunk result must be an object.")
    for key in IDENTITY_FIELDS:
        if result.get(key) != request[key]:
            raise ValueError(f"Result identity mismatch: {key}")
    if result.get("status") != "ok":
        raise ValueError(f"Chunk {request['chunk_index']} failed: {result.get('error', 'unknown error')}")
    if not isinstance(result.get("text"), str) or not isinstance(result.get("segments"), list):
        raise ValueError("Invalid text or segment list.")
    if not isinstance(result.get("language"), str):
        raise ValueError("Missing detected language.")
    if request["chunk_count"] > 1 and result["text"].strip() and not result["segments"]:
        raise ValueError("Nonempty multi-chunk text has no segments; refusing to lose text.")
    previous_start = -1.0
    duration = request["sample_count"] / SAMPLE_RATE
    for segment in result["segments"]:
        if not isinstance(segment, dict) or not isinstance(segment.get("text"), str):
            raise ValueError("Invalid segment.")
        start, end = segment.get("start"), segment.get("end")
        if not all(isinstance(t, (float, int)) and math.isfinite(t) for t in (start, end)):
            raise ValueError("Invalid segment timestamps.")
        if not 0 <= start <= end <= duration + 0.02 or start < previous_start:
            raise ValueError("Segment timestamps outside chunk or out of order.")
        previous_start = start
    return result


def global_segments(result):
    """Return a copy; never mutate the stored chunk-relative timestamps."""
    offset = result["start_offset"]
    return [{**s, "start": s["start"] + offset, "end": s["end"] + offset}
            for s in result["segments"]]
