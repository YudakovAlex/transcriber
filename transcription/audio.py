"""Local FFmpeg decoding and sample-exact chunk boundaries."""

import hashlib
import math
import subprocess
from pathlib import Path

from .contracts import SAMPLE_RATE


def effective_overlap(chunk_duration, overlap):
    if not math.isfinite(chunk_duration) or not math.isfinite(overlap):
        raise ValueError("Chunk duration and overlap must be finite.")
    if chunk_duration < 0:
        raise ValueError("Chunk duration must be nonnegative.")
    return max(0.0, min(overlap, chunk_duration - 1.0) if chunk_duration > 0 else 0.0)


def chunk_ranges(total_samples, chunk_duration=300.0, overlap=30.0):
    chunk_samples = int(chunk_duration * SAMPLE_RATE)
    step = int((chunk_duration - overlap) * SAMPLE_RATE)
    if step <= 0 or chunk_samples <= 0:
        return [(0, total_samples)]
    ranges = []
    start = 0
    while start < total_samples:
        end = min(start + chunk_samples, total_samples)
        ranges.append((start, end - start))
        if end >= total_samples:
            break
        start += step
    return ranges


def chunk_audio(audio, chunk_duration_sec=300.0, overlap_sec=30.0):
    return [audio[start:start + count] for start, count in
            chunk_ranges(len(audio), chunk_duration_sec, overlap_sec)]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_pcm(source, destination):
    """Match whisper.load_audio's FFmpeg conversion, streaming to disk (no video upload)."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-nostdin", "-threads", "0", "-i", str(source),
               "-vn", "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le",
               "-ar", str(SAMPLE_RATE), "-"]
    try:
        with destination.open("wb") as output:
            result = subprocess.run(command, stdout=output, stderr=subprocess.PIPE, check=False)
        if result.returncode:
            raise ValueError(f"Cannot decode audio: {result.stderr.decode(errors='replace')[-2000:]}")
        size = destination.stat().st_size
        if not size or size % 2:
            raise ValueError("Recording contains no decodable audio samples.")
        return {"recording_id": file_sha256(destination), "sample_count": size // 2,
                "duration": size / (2 * SAMPLE_RATE), "bytes": size}
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def read_pcm(path, start, count):
    with Path(path).open("rb") as stream:
        stream.seek(start * 2)
        data = stream.read(count * 2)
    if len(data) != count * 2:
        raise ValueError("PCM file is truncated.")
    return data
