"""Original sequential Whisper execution with model reuse across files."""

import time

import torch
import whisper
from tqdm import tqdm

from ..audio import chunk_audio, effective_overlap
from ..contracts import SAMPLE_RATE
from ..merge import merge_chunk_results
from ..progress import Spinner
from ..storage import atomic_write


class LocalBackend:
    def __init__(self, model="small"):
        started = time.perf_counter()
        self.fp16 = torch.cuda.is_available()
        self.device = "cuda" if self.fp16 else "cpu"
        self.model_name = model
        self.model = whisper.load_model(model, device=self.device)
        self.load_seconds = time.perf_counter() - started
        gpu = f" ({torch.cuda.get_device_name(0)})" if self.fp16 else ""
        print(f"Using device: {self.device}{gpu}, fp16={self.fp16}")

    def transcribe_file(self, source, language=None, chunk_duration=300.0, overlap=30.0,
                        output_path=None):
        started = time.perf_counter()
        overlap = effective_overlap(chunk_duration, overlap)
        results, timings = [], []
        duration = None
        preparation = 0.0
        if chunk_duration > 0:
            audio = whisper.load_audio(str(source))
            duration = len(audio) / SAMPLE_RATE
            chunks = chunk_audio(audio, chunk_duration, overlap) if duration > chunk_duration else [audio]
            preparation = time.perf_counter() - started
            if len(chunks) > 1:
                print(f"  Splitting into {len(chunks)} chunks (~{chunk_duration}s each, {overlap}s overlap)")
                for chunk in tqdm(chunks, desc="Chunks", unit="chunk"):
                    tick = time.perf_counter()
                    results.append(self.model.transcribe(chunk, language=language, fp16=self.fp16))
                    timings.append(time.perf_counter() - tick)
            else:
                with Spinner("Working..."):
                    tick = time.perf_counter()
                    results.append(self.model.transcribe(chunks[0], language=language, fp16=self.fp16))
                    timings.append(time.perf_counter() - tick)
        else:
            with Spinner("Working..."):
                tick = time.perf_counter()
                results.append(self.model.transcribe(str(source), language=language, fp16=self.fp16))
                timings.append(time.perf_counter() - tick)
        merge_started = time.perf_counter()
        text = merge_chunk_results(results, overlap)
        output = output_path or source.with_suffix(".txt")
        atomic_write(output, text + "\n")
        print(f"Transcript: {output}")
        return {"backend": "local", "device": self.device, "fp16": self.fp16,
                "model": self.model_name, "load_seconds": self.load_seconds,
                "preparation_seconds": preparation, "chunk_inference_seconds": timings,
                "merge_output_seconds": time.perf_counter() - merge_started,
                "end_to_end_seconds": time.perf_counter() - started,
                "original_audio_seconds": duration, "results": results}
