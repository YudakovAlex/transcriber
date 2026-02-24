"""
Unit tests for transcribe.py.

Run from project root (with venv activated or use venv's python):
  python -m unittest test_transcribe -v
  python -m pytest test_transcribe.py -v
"""
import sys
import shutil
import unittest
import uuid
import runpy
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

# Import module under test (after setting mocks if needed)
import transcribe as M

TEST_TMP_ROOT = Path(__file__).resolve().parent / ".tmp_test"


@contextmanager
def _make_temp_dir():
    TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    path = TEST_TMP_ROOT / f"tmp_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield str(path)
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _make_temp_file(suffix: str) -> Path:
    TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    path = TEST_TMP_ROOT / f"tmp_{uuid.uuid4().hex}{suffix}"
    path.write_bytes(b"")
    return path


class TestEnsureFfmpegAvailable(unittest.TestCase):
    def test_raises_when_ffmpeg_missing(self) -> None:
        with patch("transcribe.shutil.which", return_value=None):
            with self.assertRaises(SystemExit):
                M.ensure_ffmpeg_available()

    def test_does_not_raise_when_ffmpeg_found(self) -> None:
        with patch("transcribe.shutil.which", return_value="/usr/bin/ffmpeg"):
            M.ensure_ffmpeg_available()


class TestParseArgs(unittest.TestCase):
    def test_defaults(self) -> None:
        with patch.object(sys, "argv", ["transcribe", "file.mp3"]):
            args = M.parse_args()
        self.assertEqual(args.inputs, ["file.mp3"])
        self.assertEqual(args.model, M.DEFAULT_MODEL)
        self.assertEqual(args.language, M.DEFAULT_LANGUAGE)
        self.assertEqual(args.chunk_duration, M.DEFAULT_CHUNK_DURATION_SEC)
        self.assertEqual(args.overlap, M.DEFAULT_OVERLAP_SEC)

    def test_multiple_inputs(self) -> None:
        with patch.object(sys, "argv", ["transcribe", "a.mp3", "b.wav", "c.ogg"]):
            args = M.parse_args()
        self.assertEqual(args.inputs, ["a.mp3", "b.wav", "c.ogg"])

    def test_overrides(self) -> None:
        with patch.object(sys, "argv", [
            "transcribe", "x.mp3",
            "--model", "base",
            "--language", "en",
            "--chunk-duration", "180",
            "--overlap", "20",
        ]):
            args = M.parse_args()
        self.assertEqual(args.model, "base")
        self.assertEqual(args.language, "en")
        self.assertEqual(args.chunk_duration, 180.0)
        self.assertEqual(args.overlap, 20.0)


class TestSpinner(unittest.TestCase):
    def test_enter_exit(self) -> None:
        spinner = M.Spinner("test")
        with spinner:
            self.assertTrue(spinner._thread.is_alive())
        self._wait_thread_stop(spinner._thread)

    def test_exit_clears_line(self) -> None:
        with patch("transcribe.sys.stdout") as mock_stdout:
            spinner = M.Spinner("x")
            with spinner:
                pass
        mock_stdout.write.assert_any_call("\r" + " " * 80 + "\r")
        mock_stdout.flush.assert_called()

    @staticmethod
    def _wait_thread_stop(thread, timeout: float = 2.0) -> None:
        import time
        deadline = time.monotonic() + timeout
        while thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.05)


class TestCollectAudioPaths(unittest.TestCase):
    def test_single_file(self) -> None:
        path = _make_temp_file(".mp3")
        try:
            result = M.collect_audio_paths([str(path)])
            self.assertEqual(len(result), 1)
            self.assertEqual(Path(result[0]).name, path.name)
        finally:
            path.unlink(missing_ok=True)

    def test_directory_glob(self) -> None:
        with _make_temp_dir() as d:
            root = Path(d)
            (root / "a.mp3").write_bytes(b"x")
            (root / "b.wav").write_bytes(b"y")
            (root / "c.ogg").write_bytes(b"z")
            (root / "d.txt").write_bytes(b"w")
            result = M.collect_audio_paths([d])
            names = {p.name for p in result}
            self.assertEqual(names, {"a.mp3", "b.wav", "c.ogg"})

    def test_mixed_files_and_dir(self) -> None:
        with _make_temp_dir() as d:
            root = Path(d)
            (root / "in_dir.mp3").write_bytes(b"x")
            single = _make_temp_file(".mp3")
            try:
                result = M.collect_audio_paths([str(single), d])
                self.assertGreaterEqual(len(result), 2)
                stems = {Path(p).stem for p in result}
                self.assertIn(single.stem, stems)
                self.assertIn("in_dir", stems)
            finally:
                single.unlink(missing_ok=True)


class TestChunkAudio(unittest.TestCase):
    def test_returns_single_chunk_when_audio_shorter_than_chunk(self) -> None:
        audio = np.zeros(1000, dtype=np.float32)
        chunks = M.chunk_audio(audio, chunk_duration_sec=10.0, overlap_sec=1.0)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(chunks[0]), 1000)

    def test_returns_single_chunk_when_step_zero_or_negative(self) -> None:
        audio = np.zeros(10000, dtype=np.float32)
        chunks = M.chunk_audio(audio, chunk_duration_sec=5.0, overlap_sec=5.0)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(chunks[0]), 10000)

    def test_returns_single_chunk_when_chunk_duration_zero(self) -> None:
        audio = np.zeros(1000, dtype=np.float32)
        chunks = M.chunk_audio(audio, chunk_duration_sec=0.0, overlap_sec=0.0)
        self.assertEqual(len(chunks), 1)

    def test_splits_into_overlapping_chunks(self) -> None:
        # 10 sec at 16kHz = 160000 samples; chunk 5 sec, overlap 1 sec -> step 4 sec = 64000
        duration_sec = 10.0
        samples = int(duration_sec * M.SAMPLE_RATE)
        audio = np.arange(samples, dtype=np.float32)
        chunk_sec = 5.0
        overlap_sec = 1.0
        chunks = M.chunk_audio(audio, chunk_duration_sec=chunk_sec, overlap_sec=overlap_sec)
        self.assertGreater(len(chunks), 1)
        chunk_len = int(chunk_sec * M.SAMPLE_RATE)
        for i, c in enumerate(chunks):
            self.assertLessEqual(len(c), chunk_len)
        # First chunk full length
        self.assertEqual(len(chunks[0]), chunk_len)
        # Overlap: second chunk's first overlap_len samples match first chunk's last overlap_len
        overlap_len = int(overlap_sec * M.SAMPLE_RATE)
        np.testing.assert_array_equal(chunks[1][:overlap_len], chunks[0][-overlap_len:])


class TestMergeChunkResults(unittest.TestCase):
    def test_empty_returns_empty_string(self) -> None:
        self.assertEqual(M.merge_chunk_results([], 30.0), "")

    def test_single_result_returns_stripped_text(self) -> None:
        self.assertEqual(
            M.merge_chunk_results([{"text": "  hello world  "}], 30.0),
            "hello world",
        )

    def test_single_result_none_text_returns_empty(self) -> None:
        self.assertEqual(M.merge_chunk_results([{}], 30.0), "")

    def test_multiple_results_first_full_rest_trimmed_by_overlap(self) -> None:
        results = [
            {"text": "First part.", "segments": [{"start": 0, "end": 2, "text": "First part."}]},
            {
                "text": "Overlap here second part.",
                "segments": [
                    {"start": 0, "end": 25, "text": "Overlap here "},   # within 30s overlap
                    {"start": 25, "end": 40, "text": "second part."},   # after overlap
                ],
            },
        ]
        merged = M.merge_chunk_results(results, overlap_sec=30.0)
        self.assertIn("First part.", merged)
        self.assertIn("second part.", merged)
        self.assertNotIn("Overlap here", merged)  # segment end 25 <= 30, dropped

    def test_multiple_results_segment_after_overlap_included(self) -> None:
        results = [
            {"text": "One", "segments": [{"start": 0, "end": 1, "text": "One"}]},
            {
                "text": "Two",
                "segments": [
                    {"start": 0, "end": 30, "text": "overlap"},
                    {"start": 31, "end": 35, "text": "Two"},
                ],
            },
        ]
        merged = M.merge_chunk_results(results, overlap_sec=30.0)
        self.assertIn("One", merged)
        self.assertIn("Two", merged)


class TestMain(unittest.TestCase):
    def test_exits_when_no_audio_files_found(self) -> None:
        with patch.object(M, "parse_args") as mock_parse:
            mock_parse.return_value = MagicMock(
                inputs=["/nonexistent"],
                model=M.DEFAULT_MODEL,
                language=M.DEFAULT_LANGUAGE,
                chunk_duration=M.DEFAULT_CHUNK_DURATION_SEC,
                overlap=M.DEFAULT_OVERLAP_SEC,
            )
            with patch.object(M, "ensure_ffmpeg_available"):
                with patch.object(M, "collect_audio_paths", return_value=[]):
                    with self.assertRaises(SystemExit):
                        M.main()

    def test_skips_missing_file(self) -> None:
        real_path = _make_temp_file(".mp3")
        real_path.unlink(missing_ok=True)
        with patch.object(sys, "argv", ["transcribe", str(real_path)]):
            with patch.object(M, "ensure_ffmpeg_available"):
                with patch.object(M, "collect_audio_paths", return_value=[real_path]):
                    with patch("transcribe.whisper.load_model"):
                        out = M.main()
        self.assertEqual(out, 0)

    def test_skips_unsupported_extension(self) -> None:
        path = _make_temp_file(".flac")
        try:
            with patch.object(sys, "argv", ["transcribe", str(path)]):
                with patch.object(M, "ensure_ffmpeg_available"):
                    with patch.object(M, "collect_audio_paths", return_value=[path]):
                        with patch("transcribe.whisper.load_model"):
                            out = M.main()
            self.assertEqual(out, 0)
        finally:
            path.unlink(missing_ok=True)

    def test_transcribes_single_file_no_chunking(self) -> None:
        with _make_temp_dir() as d:
            root = Path(d)
            audio_file = root / "rec.mp3"
            audio_file.write_bytes(b"fake")
            out_file = root / "rec.txt"
            mock_model = MagicMock()
            mock_model.transcribe.return_value = {"text": "Hello world"}
            with patch.object(sys, "argv", ["transcribe", str(audio_file), "--chunk-duration", "0"]):
                with patch.object(M, "ensure_ffmpeg_available"):
                    with patch.object(M, "collect_audio_paths", return_value=[audio_file]):
                        with patch("transcribe.whisper.load_model", return_value=mock_model):
                            with patch("transcribe.torch.cuda.is_available", return_value=False):
                                out = M.main()
            self.assertEqual(out, 0)
            mock_model.transcribe.assert_called_once()
            self.assertTrue(out_file.exists())
            self.assertIn("Hello world", out_file.read_text(encoding="utf-8"))
            self.assertFalse(mock_model.transcribe.call_args.kwargs["fp16"])

    def test_transcribes_with_chunking(self) -> None:
        with _make_temp_dir() as d:
            root = Path(d)
            audio_file = root / "long.mp3"
            audio_file.write_bytes(b"fake")
            out_file = root / "long.txt"
            mock_model = MagicMock()
            mock_model.transcribe.side_effect = [
                {"text": "Part one.", "segments": [{"start": 0, "end": 2, "text": "Part one."}]},
                {"text": "Part two.", "segments": [{"start": 31, "end": 35, "text": "Part two."}]},
            ]
            # Audio long enough to create exactly 2 chunks: 8 sec at 16kHz with 5s chunks and 1s overlap.
            fake_audio = np.zeros(int(8 * M.SAMPLE_RATE), dtype=np.float32)
            with patch.object(sys, "argv", ["transcribe", str(audio_file), "--chunk-duration", "5", "--overlap", "1"]):
                with patch.object(M, "ensure_ffmpeg_available"):
                    with patch.object(M, "collect_audio_paths", return_value=[audio_file]):
                        with patch("transcribe.whisper.load_model", return_value=mock_model):
                            with patch("transcribe.whisper.load_audio", return_value=fake_audio):
                                with patch("transcribe.tqdm", side_effect=lambda it, **kw: it):
                                    with patch("transcribe.torch.cuda.is_available", return_value=True):
                                        with patch("transcribe.torch.cuda.get_device_name", return_value="Mock GPU"):
                                            out = M.main()
            self.assertEqual(out, 0)
            self.assertEqual(mock_model.transcribe.call_count, 2)
            self.assertTrue(out_file.exists())
            text = out_file.read_text(encoding="utf-8")
            self.assertIn("Part one.", text)
            self.assertIn("Part two.", text)
            self.assertTrue(mock_model.transcribe.call_args.kwargs["fp16"])

    def test_chunking_single_chunk_uses_spinner_path(self) -> None:
        with _make_temp_dir() as d:
            root = Path(d)
            audio_file = root / "short.mp3"
            audio_file.write_bytes(b"fake")
            out_file = root / "short.txt"
            mock_model = MagicMock()
            mock_model.transcribe.return_value = {"text": "Only chunk."}
            fake_audio = np.zeros(int(4 * M.SAMPLE_RATE), dtype=np.float32)  # shorter than chunk-duration
            with patch.object(sys, "argv", ["transcribe", str(audio_file), "--chunk-duration", "5", "--overlap", "1"]):
                with patch.object(M, "ensure_ffmpeg_available"):
                    with patch.object(M, "collect_audio_paths", return_value=[audio_file]):
                        with patch("transcribe.whisper.load_model", return_value=mock_model):
                            with patch("transcribe.whisper.load_audio", return_value=fake_audio):
                                with patch("transcribe.torch.cuda.is_available", return_value=False):
                                    out = M.main()
            self.assertEqual(out, 0)
            mock_model.transcribe.assert_called_once()
            self.assertTrue(out_file.exists())
            self.assertIn("Only chunk.", out_file.read_text(encoding="utf-8"))
            self.assertFalse(mock_model.transcribe.call_args.kwargs["fp16"])


class TestEntrypoint(unittest.TestCase):
    def test_module_entrypoint_help_exits_zero(self) -> None:
        script_path = Path(M.__file__).resolve()
        with patch.object(sys, "argv", [str(script_path), "--help"]):
            with self.assertRaises(SystemExit) as ctx:
                runpy.run_path(str(script_path), run_name="__main__")
        self.assertEqual(ctx.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
