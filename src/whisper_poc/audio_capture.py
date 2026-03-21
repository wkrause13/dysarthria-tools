from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from queue import Queue
from typing import Generator
import wave

import numpy as np
import sounddevice as sd


def frame_byte_count(sample_rate: int, frame_ms: int, channels: int) -> int:
    samples_per_frame = sample_rate * frame_ms // 1000
    return samples_per_frame * channels * 2


def pcm16_rms(pcm_bytes: bytes) -> float:
    if not pcm_bytes:
        return 0.0
    samples = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32)
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples))))


def iter_wav_frames(
    wav_path: Path,
    *,
    frame_bytes: int,
    sample_rate: int,
    channels: int,
    trailing_silence_frames: int = 0,
) -> Generator[bytes, None, None]:
    with wave.open(str(wav_path), "rb") as handle:
        if handle.getframerate() != sample_rate:
            raise ValueError(
                f"expected sample_rate={sample_rate}, got {handle.getframerate()}"
            )
        if handle.getnchannels() != channels:
            raise ValueError(f"expected channels={channels}, got {handle.getnchannels()}")
        if handle.getsampwidth() != 2:
            raise ValueError(f"expected 16-bit PCM, got sample width={handle.getsampwidth()}")

        samples_per_frame = frame_bytes // (channels * 2)
        while True:
            frame = handle.readframes(samples_per_frame)
            if len(frame) != frame_bytes:
                break
            yield frame

    for _ in range(max(0, trailing_silence_frames)):
        yield b"\x00" * frame_bytes


@contextmanager
def microphone_queue(
    sample_rate: int,
    channels: int,
    blocksize: int,
    device: int | None = None,
) -> Generator[Queue[bytes], None, None]:
    queue: Queue[bytes] = Queue()

    def callback(indata, frames, time_info, status) -> None:  # pragma: no cover - exercised manually
        if status:
            print(f"[audio] status={status}")
        queue.put(bytes(indata))

    stream = sd.RawInputStream(
        samplerate=sample_rate,
        channels=channels,
        dtype="int16",
        blocksize=blocksize,
        callback=callback,
        device=device,
    )
    with stream:
        yield queue
