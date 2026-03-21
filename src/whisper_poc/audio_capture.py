from __future__ import annotations

from contextlib import contextmanager
from queue import Queue
from typing import Generator

import sounddevice as sd


def frame_byte_count(sample_rate: int, frame_ms: int, channels: int) -> int:
    samples_per_frame = sample_rate * frame_ms // 1000
    return samples_per_frame * channels * 2


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
