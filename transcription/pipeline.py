"""Checkpointed, bounded remote execution independent of the Modal SDK."""

import hashlib
import time
import uuid
from pathlib import Path

from .audio import chunk_ranges, decode_pcm, effective_overlap, file_sha256, read_pcm
from .contracts import (IDENTITY_FIELDS, IMPLEMENTATION_VERSION, SAMPLE_RATE,
                        SCHEMA_VERSION, configuration, fingerprint, validate_result)
from .merge import merge_chunk_results, ordered_results
from .storage import atomic_write, job_lock, read_json, write_json


class CallExpired(Exception):
    """The platform no longer has the result; the chunk can be resubmitted."""


class ChunkFailed(RuntimeError):
    pass


def prepare_job(source, directory, config, resume=False):
    source, directory = Path(source).resolve(), Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    source_identity = {"path": str(source), "sha256": file_sha256(source)}
    manifest_path, pcm_path = directory / "job.json", directory / "audio.pcm"
    saved = read_json(manifest_path) if resume else None
    if saved and (saved.get("source") != source_identity or saved.get("config") != config
                  or saved.get("config_fingerprint") != fingerprint(config)):
        raise ValueError("Resume source/configuration changed; start a new job with the original options.")
    if saved and saved.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported checkpoint version.")
    if saved and pcm_path.exists() and file_sha256(pcm_path) == saved["recording_id"]:
        audio = {key: saved[key] for key in ("recording_id", "sample_count", "duration", "bytes")}
        if pcm_path.stat().st_size != audio["bytes"] or audio["bytes"] != 2 * audio["sample_count"]:
            raise ValueError("Checkpoint PCM metadata is inconsistent.")
    else:
        temporary = directory / "audio.preparing.pcm"
        audio = decode_pcm(source, temporary)
        if file_sha256(source) != source_identity["sha256"]:
            temporary.unlink(missing_ok=True)
            raise ValueError("Source changed during decoding; retry with a stable file.")
        if saved and any(saved[key] != audio[key] for key in audio):
            temporary.unlink(missing_ok=True)
            raise ValueError("Decoded audio changed; start a new job.")
        temporary.replace(pcm_path)
    job_id = saved["job_id"] if saved else uuid.uuid4().hex
    ranges = chunk_ranges(audio["sample_count"], config["chunk_duration"], config["overlap"])
    if not ranges or not audio["sample_count"]:
        raise ValueError("Recording contains no audio.")
    requests = []
    for index, (start, count) in enumerate(ranges):
        requests.append({
            "schema_version": SCHEMA_VERSION, "implementation_version": IMPLEMENTATION_VERSION,
            "job_id": job_id, "recording_id": audio["recording_id"],
            "config_fingerprint": fingerprint(config), "chunk_index": index,
            "chunk_count": len(ranges), "start_sample": start, "sample_count": count,
            "start_offset": start / SAMPLE_RATE, "sample_rate": SAMPLE_RATE,
            "model": config["model"], "checkpoint_sha256": config["checkpoint_sha256"],
            "requested_language": config["language"],
            "pcm_sha256": hashlib.sha256(read_pcm(pcm_path, start, count)).hexdigest(),
            "audio_ref": f"jobs/{job_id}/audio.pcm",
        })
    if saved and requests != saved["requests"]:
        raise ValueError("Checkpoint chunk boundaries/identities changed.")
    manifest = saved or {"schema_version": SCHEMA_VERSION, "job_id": job_id,
                         "source": source_identity, "config": config,
                         "config_fingerprint": fingerprint(config), "requests": requests,
                         "created_at": time.time(), **audio}
    write_json(manifest_path, manifest)
    return manifest


class JobRunner:
    """Transport owns inference retries. This scheduler never retries an ambiguous submit."""

    def __init__(self, transport, directory, workers=2, deadline_seconds=3600,
                 progress=print, clock=time.monotonic, sleep=time.sleep):
        if workers not in (1, 2, 3, 4):
            raise ValueError("Worker count must be between 1 and 4.")
        self.transport, self.directory, self.workers = transport, Path(directory), workers
        self.deadline_seconds, self.progress = deadline_seconds, progress
        self.clock, self.sleep = clock, sleep
        self.state_path = self.directory / "state.json"

    def _save(self):
        write_json(self.state_path, self.state)

    def _cancel(self):
        for entry in self.state["calls"].values():
            if entry["status"] == "pending":
                try:
                    self.transport.cancel(entry["call_id"])
                    entry["status"] = "cancelled"
                except Exception as error:
                    # Retain the call ID: resume must reconcile before submitting again.
                    self.progress(f"Cancellation unconfirmed for {entry['call_id']}: {error}")
        self._save()

    def run(self, manifest):
        started = self.clock()
        requests = manifest["requests"]
        self.state = read_json(self.state_path) if self.state_path.exists() else {
            "job_id": manifest["job_id"], "calls": {}, "history": [], "status": "running"}
        if self.state.get("job_id") != manifest["job_id"]:
            raise ValueError("Checkpoint state belongs to another job.")
        completed = {}
        for request in requests:
            index = request["chunk_index"]
            result_path = self.directory / "results" / f"{index:06d}.json"
            if result_path.exists():
                completed[index] = validate_result(read_json(result_path), request)
        self.progress(f"Chunks: {len(completed)}/{len(requests)} complete; resume: {self.directory}")
        try:
            # Reconcile pending calls even when their result was checkpointed just before a crash.
            for key, entry in self.state["calls"].items():
                if not key.isdigit() or int(key) >= len(requests):
                    raise ValueError("Checkpoint has an unexpected chunk index.")
                if entry["status"] == "submitting":
                    raise ChunkFailed(
                        "A submit acknowledgement was lost. Check the Modal dashboard, cancel any "
                        "orphan call, then mark this state's call entry 'cancelled' before resuming. "
                        "Automatic resubmission could incur duplicate charges.")
                if entry["status"] == "failed":
                    # A new invocation is an explicit resume, not an automatic retry loop.
                    self.state["history"].append(dict(entry, chunk_index=int(key)))
                    entry["status"] = "retry_ready"
                if int(key) in completed and entry["status"] == "pending":
                    self.transport.cancel(entry["call_id"])
                    entry["status"] = "complete"
            self.state["status"] = "running"
            self._save()
            while len(completed) < len(requests):
                if self.clock() - started >= self.deadline_seconds:
                    raise TimeoutError("Job deadline exceeded; completed chunks are saved.")
                changed = False
                for key, entry in list(self.state["calls"].items()):
                    if entry["status"] != "pending":
                        continue
                    index = int(key)
                    try:
                        result = self.transport.poll(entry["call_id"])
                    except CallExpired:
                        entry["status"] = "expired"
                        self.state["history"].append(dict(entry, chunk_index=index))
                        self._save()
                        changed = True
                        continue
                    if result is None:
                        continue
                    if not isinstance(result, dict):
                        raise ValueError(f"Chunk {index} returned a malformed result.")
                    if result.get("status") != "ok":
                        entry["status"] = "failed"
                        entry["error"] = result.get("error")
                        self._save()
                        raise ChunkFailed(f"Chunk {index}: {result.get('error', 'worker failed')}")
                    validate_result(result, requests[index])
                    result.setdefault("execution", {})["call_id"] = entry["call_id"]
                    write_json(self.directory / "results" / f"{index:06d}.json", result)
                    completed[index] = result
                    entry["status"] = "complete"
                    entry["received_at"] = time.time()
                    self._save()
                    execution = result.get("execution", {})
                    retries = max(0, execution.get("attempts", 1) - 1)
                    self.progress(f"Chunks: {len(completed)}/{len(requests)} complete "
                                  f"({self.clock() - started:.1f}s, chunk {index}, retries {retries})")
                    changed = True
                pending = sum(e["status"] == "pending" for e in self.state["calls"].values())
                for request in requests:
                    index, key = request["chunk_index"], str(request["chunk_index"])
                    entry = self.state["calls"].get(key)
                    if index in completed or (entry and entry["status"] == "pending"):
                        continue
                    if entry and entry["status"] == "failed":
                        raise ChunkFailed(f"Chunk {index} previously failed: {entry.get('error')}. "
                                          "Resolve the cause and start a new job.")
                    if pending >= self.workers:
                        break
                    entry = {"status": "submitting", "submitted_at": time.time()}
                    self.state["calls"][key] = entry
                    self._save()  # Persist intent before a potentially ambiguous network operation.
                    call_id = self.transport.submit(dict(request, deadline_at=time.time() +
                                                         max(1, self.deadline_seconds - (self.clock() - started))))
                    entry.update(call_id=call_id, status="pending", acknowledged_at=time.time())
                    self._save()
                    pending += 1
                    changed = True
                if not changed:
                    self.sleep(0.25)
            self.state["status"] = "complete"
            self._save()
            return ordered_results(requests, completed.values())
        except BaseException:
            self.state["status"] = "interrupted"
            self._cancel()
            raise


def transcribe_remote(source, model="small", language=None, chunk_duration=300.0,
                      overlap=30.0, workers=2, resume=None, directory=None,
                      transport_factory=None, output_path=None, progress=print):
    """Prepare locally, then invoke the remote backend; no model is loaded locally."""
    started = time.perf_counter()
    source = Path(source).resolve()
    config = configuration(model, language, chunk_duration, effective_overlap(chunk_duration, overlap))
    directory = Path(resume or directory or source.parent / ".transcriber-jobs" / uuid.uuid4().hex)
    with job_lock(directory):
        progress(f"Preparing audio locally; job: {directory}")
        manifest = prepare_job(source, directory, config, resume=bool(resume))
        preparation = time.perf_counter() - started
        if transport_factory is None:
            from .backends.modal import ModalBackend
            transport_factory = ModalBackend
        transport = transport_factory(model=model, workers=workers)
        upload_started = time.perf_counter()
        progress("Checking model cache and uploading audio...")
        upload = transport.prepare(directory / "audio.pcm", manifest)
        upload_seconds = time.perf_counter() - upload_started
        results = JobRunner(transport, directory, workers, progress=progress).run(manifest)
        merge_started = time.perf_counter()
        text = merge_chunk_results(results, config["overlap"])
        languages = sorted({r["language"] for r in results})
        if len(languages) > 1:
            progress(f"Detected different chunk languages: {', '.join(languages)}")
        output = Path(output_path) if output_path else source.with_suffix(".txt")
        atomic_write(output, text + "\n")
        metrics = {"backend": "modal", "workers": workers, "preparation_seconds": preparation,
                   "upload_seconds": upload_seconds, "upload": upload,
                   "merge_output_seconds": time.perf_counter() - merge_started,
                   "end_to_end_seconds": time.perf_counter() - started,
                   "original_audio_seconds": manifest["duration"],
                   "processed_audio_seconds": sum(r["sample_count"] for r in results) / SAMPLE_RATE,
                   "results": results, "languages": languages, "config": config,
                   "job_directory": str(directory.resolve())}
        metrics["calls"] = read_json(directory / "state.json")["calls"]
        write_json(directory / "metrics.json", metrics)
        try:
            transport.cleanup(manifest)
        except Exception as error:
            progress(f"Transcript saved; remote cleanup will be retried by retention cleanup: {error}")
        progress(f"Transcript: {output}")
        return metrics
