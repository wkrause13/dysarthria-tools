from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

import httpx
import sounddevice as sd

_API_BASE = "https://api.elevenlabs.io/v1/text-to-speech"


class ElevenLabsSpeaker:
    def __init__(
        self,
        api_key: str,
        voice_id: str,
        *,
        model_id: str = "eleven_flash_v2_5",
        output_sample_rate: int = 24000,
        enabled: bool = True,
        client: httpx.Client | None = None,
        stream_factory: Any = None,
    ) -> None:
        self.api_key = api_key
        self.voice_id = voice_id
        self.model_id = model_id
        self.output_sample_rate = output_sample_rate
        self.enabled = enabled
        self._client = client or httpx.Client(timeout=30.0)
        self._stream_factory = stream_factory or self._default_stream_factory
        self._pending_text: deque[str] = deque()
        self._is_playing = False
        self._stop_event = threading.Event()

    @staticmethod
    def _default_stream_factory(samplerate: int) -> sd.RawOutputStream:
        return sd.RawOutputStream(samplerate=samplerate, channels=1, dtype="int16")

    def speak(self, text: str) -> float:
        if not self.enabled or not text:
            return 0.0
        if self._is_playing:
            self._pending_text.append(text)
            return 0.0
        return self._start_speaking(text)

    def _start_speaking(self, text: str) -> float:
        self._is_playing = True
        self._stop_event.clear()
        start = time.perf_counter()
        thread = threading.Thread(
            target=self._stream_audio, args=(text,), daemon=True
        )
        thread.start()
        return time.perf_counter() - start

    def _stream_audio(self, text: str) -> None:
        url = f"{_API_BASE}/{self.voice_id}/stream"
        stream: Any = None
        try:
            with self._client.stream(
                "POST",
                url,
                headers={
                    "xi-api-key": self.api_key,
                    "Content-Type": "application/json",
                },
                params={
                    "output_format": f"pcm_{self.output_sample_rate}",
                    "optimize_streaming_latency": "3",
                },
                json={
                    "text": text,
                    "model_id": self.model_id,
                },
            ) as response:
                response.raise_for_status()
                stream = self._stream_factory(self.output_sample_rate)
                stream.start()
                for chunk in response.iter_bytes(chunk_size=4096):
                    if self._stop_event.is_set():
                        break
                    stream.write(chunk)
        except Exception:
            pass
        finally:
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass
            self._is_playing = False

    def process_queue(self) -> float:
        if self._is_playing or not self._pending_text:
            return 0.0
        return self._start_speaking(self._pending_text.popleft())

    def has_pending(self) -> bool:
        return bool(self._pending_text)

    def stop(self) -> None:
        if not self.enabled:
            return
        self._pending_text.clear()
        self._stop_event.set()

    def is_speaking(self) -> bool:
        if not self.enabled:
            return False
        return self._is_playing
