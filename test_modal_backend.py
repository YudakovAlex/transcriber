"""Offline contract, failure recovery, and real FFmpeg preparation tests."""

import copy
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from transcription.audio import chunk_ranges, decode_pcm, file_sha256
from transcription.contracts import (IDENTITY_FIELDS, SAMPLE_RATE, configuration,
                                     global_segments, validate_result)
from transcription.merge import merge_chunk_results, ordered_results
from transcription.pipeline import CallExpired, ChunkFailed, JobRunner, prepare_job, transcribe_remote
from transcription.storage import atomic_write, job_lock, read_json, write_json
from transcription.worker import execute_chunk

ROOT = Path(__file__).resolve().parent
TMP = ROOT / ".tmp_test"


def result_for(request, text=None):
    text = text if text is not None else f"chunk {request['chunk_index']}"
    return {**{key: request[key] for key in IDENTITY_FIELDS}, "status": "ok",
            "text": text, "language": "en", "segments": [
                {"start": 0.0, "end": min(2.0, request["sample_count"] / SAMPLE_RATE), "text": text}],
            "execution": {"attempts": 1}}


class FakeTransport:
    def __init__(self, **kwargs):
        self.calls, self.cancelled, self.submitted, self.delivered = {}, [], [], []
        self.next_id, self.max_pending = 0, 0
        self.reverse = False

    def prepare(self, pcm, manifest):
        return {"bytes": manifest["bytes"]}

    def cleanup(self, manifest):
        self.cleaned = True

    def submit(self, request):
        call_id = f"call-{self.next_id}"
        self.next_id += 1
        self.calls[call_id] = request
        self.submitted.append(request["chunk_index"])
        self.max_pending = max(self.max_pending, len(self.calls))
        return call_id

    def poll(self, call_id):
        if self.reverse and len(self.calls) > 1 and call_id == next(iter(self.calls)):
            return None
        request = self.calls.pop(call_id)
        self.delivered.append(request["chunk_index"])
        return result_for(request)

    def cancel(self, call_id):
        self.cancelled.append(call_id)
        self.calls.pop(call_id, None)


class JobTests(unittest.TestCase):
    def setUp(self):
        TMP.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=TMP)
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.source = self.root / "input.wav"
        self.source.write_bytes(b"fixture source")
        self.job = self.root / "job"
        self.config = configuration("small", "en", 5.0, 1.0)
        self.pcm = np.zeros(8 * SAMPLE_RATE, dtype="<i2").tobytes()

    def manifest(self):
        def decode(source, destination):
            destination.write_bytes(self.pcm)
            return {"recording_id": hashlib.sha256(self.pcm).hexdigest(),
                    "sample_count": len(self.pcm) // 2, "duration": 8.0, "bytes": len(self.pcm)}
        with patch("transcription.pipeline.decode_pcm", side_effect=decode):
            return prepare_job(self.source, self.job, self.config)

    def runner(self, transport, **kwargs):
        return JobRunner(transport, self.job, progress=lambda message: None, sleep=lambda seconds: None, **kwargs)

    def test_out_of_order_results_and_bound(self):
        manifest = self.manifest()
        transport = FakeTransport()
        transport.reverse = True
        results = self.runner(transport, workers=2).run(manifest)
        self.assertEqual(transport.delivered, [1, 0])
        self.assertEqual([r["chunk_index"] for r in results], [0, 1])
        self.assertLessEqual(transport.max_pending, 2)
        self.assertEqual(read_json(self.job / "state.json")["status"], "complete")

    def test_resume_completed_job_submits_nothing(self):
        manifest = self.manifest()
        self.runner(FakeTransport()).run(manifest)
        transport = FakeTransport()
        self.runner(transport).run(manifest)
        self.assertEqual(transport.submitted, [])

    def test_interrupt_saves_first_chunk_and_cancels_rest(self):
        manifest = self.manifest()
        transport = FakeTransport()
        original = transport.poll
        transport.poll = lambda call: original(call) if call == "call-0" else (_ for _ in ()).throw(KeyboardInterrupt())
        with self.assertRaises(KeyboardInterrupt):
            self.runner(transport).run(manifest)
        self.assertTrue((self.job / "results" / "000000.json").exists())
        self.assertIn("call-1", transport.cancelled)
        resumed = FakeTransport()
        self.runner(resumed).run(manifest)
        self.assertEqual(resumed.submitted, [1])

    def test_abrupt_exit_reconciles_pending_call(self):
        manifest = self.manifest()
        transport = FakeTransport()
        call_id = transport.submit(manifest["requests"][0])
        write_json(self.job / "state.json", {"job_id": manifest["job_id"], "history": [],
                   "calls": {"0": {"call_id": call_id, "status": "pending"}}})
        self.runner(transport).run(manifest)
        self.assertEqual(transport.submitted, [0, 1])

    def test_expired_call_is_resubmitted(self):
        manifest = self.manifest()
        transport = FakeTransport()
        write_json(self.job / "state.json", {"job_id": manifest["job_id"], "history": [],
                   "calls": {"0": {"call_id": "expired", "status": "pending"}}})
        original = transport.poll
        def poll(call):
            if call == "expired":
                raise CallExpired()
            return original(call)
        transport.poll = poll
        self.runner(transport).run(manifest)
        self.assertEqual(transport.submitted, [0, 1])

    def test_ambiguous_submit_is_not_automatically_retried(self):
        manifest = self.manifest()
        transport = FakeTransport()
        transport.submit = Mock(side_effect=OSError("lost acknowledgement"))
        with self.assertRaises(OSError):
            self.runner(transport).run(manifest)
        with self.assertRaisesRegex(ChunkFailed, "acknowledgement"):
            self.runner(FakeTransport()).run(manifest)
        transport.submit.assert_called_once()

    def test_failed_chunk_cancels_siblings_and_never_merges(self):
        manifest = self.manifest()
        transport = FakeTransport()
        transport.poll = lambda call: {"status": "error", "error": "CUDA out of memory"}
        with self.assertRaisesRegex(ChunkFailed, "CUDA"):
            self.runner(transport).run(manifest)
        self.assertIn("call-1", transport.cancelled)
        resumed = FakeTransport()
        self.assertEqual(len(self.runner(resumed).run(manifest)), 2)

    def test_wrong_result_identity_cancels_known_calls(self):
        manifest = self.manifest()
        transport = FakeTransport()
        transport.poll = lambda call: dict(result_for(manifest["requests"][0]), recording_id="wrong")
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            self.runner(transport).run(manifest)
        self.assertEqual(len(transport.cancelled), 2)

    def test_successful_output_cleanup_and_language_warning(self):
        self.manifest()
        transport = FakeTransport()
        original = transport.poll
        def poll(call):
            result = original(call)
            if result["chunk_index"] == 1:
                result["language"] = "fr"
            return result
        transport.poll = poll
        messages = []
        metrics = transcribe_remote(self.source, model="small", language="en", chunk_duration=5,
                      overlap=1, resume=self.job, transport_factory=lambda **kw: transport,
                      progress=messages.append)
        self.assertTrue(transport.cleaned)
        self.assertEqual(self.source.with_suffix(".txt").read_text().strip(), "chunk 0 chunk 1")
        self.assertEqual(metrics["languages"], ["en", "fr"])
        self.assertTrue(any("different chunk languages" in message for message in messages))

    def test_pcm_regeneration_after_local_cache_removal(self):
        manifest = self.manifest()
        (self.job / "audio.pcm").unlink()
        def decode(source, destination):
            destination.write_bytes(self.pcm)
            return {key: manifest[key] for key in ("recording_id", "sample_count", "duration", "bytes")}
        with patch("transcription.pipeline.decode_pcm", side_effect=decode):
            resumed = prepare_job(self.source, self.job, self.config, resume=True)
        self.assertEqual(resumed["job_id"], manifest["job_id"])

    def test_deadline_cancels_pending(self):
        manifest = self.manifest()
        transport = FakeTransport()
        transport.poll = lambda call: None
        ticks = iter([0, 0, 0, 0, 10])
        with self.assertRaises(TimeoutError):
            self.runner(transport, clock=lambda: next(ticks), deadline_seconds=5).run(manifest)
        self.assertEqual(len(transport.cancelled), 2)

    def test_cancellation_failure_keeps_call_for_resume(self):
        manifest = self.manifest()
        transport = FakeTransport()
        transport.poll = Mock(side_effect=KeyboardInterrupt())
        transport.cancel = Mock(side_effect=OSError("offline"))
        with self.assertRaises(KeyboardInterrupt):
            self.runner(transport).run(manifest)
        self.assertEqual(read_json(self.job / "state.json")["calls"]["0"]["status"], "pending")

    def test_changed_source_or_config_rejected(self):
        self.manifest()
        changed = dict(self.config, language="fr")
        with self.assertRaisesRegex(ValueError, "changed"):
            prepare_job(self.source, self.job, changed, resume=True)
        self.source.write_bytes(b"different content")
        with self.assertRaisesRegex(ValueError, "changed"):
            prepare_job(self.source, self.job, self.config, resume=True)

    def test_incomplete_temporary_checkpoint_is_ignored(self):
        manifest = self.manifest()
        (self.job / ".state.json.interrupted.tmp").write_text("{")
        self.assertEqual(len(self.runner(FakeTransport()).run(manifest)), 2)

    def test_corrupt_committed_checkpoint_fails_closed(self):
        manifest = self.manifest()
        (self.job / "state.json").write_text("{")
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            self.runner(FakeTransport()).run(manifest)

    def test_result_validation_and_global_offsets(self):
        request = self.manifest()["requests"][1]
        result = result_for(request)
        shifted = global_segments(result)
        self.assertEqual(shifted[0]["start"], 4.0)
        self.assertEqual(result["segments"][0]["start"], 0.0)
        for field, value in (("recording_id", "wrong"), ("model", "base"), ("start_offset", 0)):
            wrong = dict(result, **{field: value})
            with self.assertRaises(ValueError):
                validate_result(wrong, request)
        for segments in ([], [{"start": -1, "end": 2, "text": "bad"}],
                         [{"start": 0, "end": float("nan"), "text": "bad"}]):
            with self.assertRaises(ValueError):
                validate_result(dict(result, segments=segments), request)
        validate_result(dict(result, text="", segments=[]), request)

    def test_duplicates_and_missing_results(self):
        requests = self.manifest()["requests"]
        results = [result_for(r) for r in requests]
        self.assertEqual(len(ordered_results(requests, results + [copy.deepcopy(results[0])])), 2)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            ordered_results(requests, results[:1])
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            ordered_results(requests, results + [dict(results[0], text="different")])

    def test_existing_transcript_survives_remote_failure(self):
        self.manifest()
        output = self.source.with_suffix(".txt")
        output.write_text("old transcript")
        transport = FakeTransport()
        transport.poll = lambda call: {"status": "error", "error": "bad"}
        with self.assertRaises(ChunkFailed):
            transcribe_remote(self.source, model="small", language="en", chunk_duration=5,
                              overlap=1, resume=self.job, transport_factory=lambda **kw: transport,
                              progress=lambda message: None)
        self.assertEqual(output.read_text(), "old transcript")

    def test_atomic_write_failure_leaves_old_file(self):
        output = self.root / "output.txt"
        output.write_text("old")
        with patch("transcription.storage.os.replace", side_effect=OSError("locked")):
            with self.assertRaises(OSError):
                atomic_write(output, "new")
        self.assertEqual(output.read_text(), "old")

    def test_exclusive_job_lock(self):
        with job_lock(self.job):
            with self.assertRaisesRegex(ValueError, "another process"):
                with job_lock(self.job):
                    pass

    def test_worker_retries_transient_reads_and_reuses_model(self):
        request = self.manifest()["requests"][0]
        payload = self.pcm[:request["sample_count"] * 2]
        model = Mock()
        model.transcribe.return_value = {"text": "hello", "language": "en",
                                         "segments": [{"start": 0, "end": 1, "text": "hello"}]}
        read = Mock(side_effect=[OSError("temporary"), OSError("temporary"), payload])
        sleep = Mock()
        result = execute_chunk(request, None, model=model, read_audio=read, sleep=sleep)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["execution"]["attempts"], 3)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [1, 2])
        execute_chunk(request, payload, model=model, read_audio=read)
        self.assertEqual(model.transcribe.call_count, 2)

    def test_worker_permanent_error_and_retry_exhaustion(self):
        request = self.manifest()["requests"][0]
        model = Mock()
        read = Mock(side_effect=OSError("offline"))
        result = execute_chunk(request, None, model=model, read_audio=read, sleep=lambda _: None)
        self.assertEqual(result["status"], "error")
        self.assertEqual(read.call_count, 3)
        model.transcribe.assert_not_called()
        result = execute_chunk(request, b"corrupt", model=model, read_audio=read)
        self.assertEqual(result["error"]["type"], "ValueError")


class AudioAndCLITests(unittest.TestCase):
    def test_coverage_final_chunk_and_no_chunking(self):
        ranges = chunk_ranges(3600 * SAMPLE_RATE, 300, 30)
        self.assertEqual(len(ranges), 14)
        self.assertEqual(sum(n for _, n in ranges) / SAMPLE_RATE, 3990)
        self.assertEqual(ranges[-1], (3510 * SAMPLE_RATE, 90 * SAMPLE_RATE))
        self.assertEqual(chunk_ranges(100, 0, 0), [(0, 100)])

    def test_pcm_round_trip_is_exact(self):
        values = np.arange(-32768, 32768, dtype=np.int32).astype("<i2")
        samples = values.astype(np.float32) / 32768.0
        np.testing.assert_array_equal((samples * 32768).astype("<i2"), values)

    def test_legacy_boundary_rule_and_missing_metadata(self):
        results = [{"text": "Ship today.", "segments": [{"start": 1, "end": 2, "text": "Ship today."}]},
                   {"text": "today. Next.", "segments": [{"start": 25, "end": 35, "text": "today. Next."}]}]
        self.assertEqual(merge_chunk_results(results, 30), "Ship today. today. Next.")
        with self.assertRaisesRegex(ValueError, "no segments"):
            merge_chunk_results([results[0], {"text": "missing"}], 30)

    def test_local_help_does_not_import_optional_or_inference_packages(self):
        command = "import sys, transcribe; assert 'modal' not in sys.modules; assert 'torch' not in sys.modules; assert 'whisper' not in sys.modules"
        subprocess.run([sys.executable, "-c", command], cwd=ROOT, check=True)

    def test_cli_backend_defaults_and_validation(self):
        import transcribe
        for argv in (["x.wav", "--modal-workers", "2"], ["x.wav", "--resume", "job"],
                     ["x.wav", "--chunk-duration", "nan"]):
            with patch.object(sys, "argv", ["transcribe", *argv]), self.assertRaises(SystemExit):
                transcribe.parse_args()
        with patch.object(sys, "argv", ["transcribe", "x.wav", "--backend", "modal"]):
            self.assertEqual(transcribe.parse_args().modal_workers, 2)

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg not installed")
    def test_real_stereo_wav_mp4_and_invalid_audio(self):
        TMP.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TMP) as directory:
            root = Path(directory)
            wav = root / "stereo.wav"
            samples = np.zeros((8000, 2), dtype="<i2")
            samples[:, 0] = (np.sin(np.arange(8000) * 0.1) * 5000).astype("<i2")
            with wave.open(str(wav), "wb") as output:
                output.setnchannels(2)
                output.setsampwidth(2)
                output.setframerate(8000)
                output.writeframes(samples.tobytes())
            meta = decode_pcm(wav, root / "wav.pcm")
            self.assertEqual(meta["sample_count"], SAMPLE_RATE)
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                            "color=c=black:s=16x16:d=1", "-i", str(wav),
                            "-c:v", "mpeg4", "-c:a", "aac", "-shortest",
                            str(root / "audio.mp4")], check=True, capture_output=True)
            self.assertGreater(decode_pcm(root / "audio.mp4", root / "mp4.pcm")["sample_count"], 0)
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                            "color=c=black:s=16x16:d=0.1", "-c:v", "mpeg4",
                            str(root / "no-audio.mp4")], check=True, capture_output=True)
            with self.assertRaisesRegex(ValueError, "Cannot decode"):
                decode_pcm(root / "no-audio.mp4", root / "no-audio.pcm")
            (root / "bad.mp4").write_bytes(b"invalid")
            with self.assertRaisesRegex(ValueError, "Cannot decode"):
                decode_pcm(root / "bad.mp4", root / "bad.pcm")
            self.assertFalse((root / "bad.pcm").exists())
            empty = root / "empty.wav"
            with wave.open(str(empty), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(SAMPLE_RATE)
            with self.assertRaises(ValueError):
                decode_pcm(empty, root / "empty.pcm")


if __name__ == "__main__":
    unittest.main()
