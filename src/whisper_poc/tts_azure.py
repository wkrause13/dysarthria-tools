from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import httpx
import sounddevice as sd

from whisper_poc.prosody import ProsodyFeatures, build_ssml, extract_prosody

_AZURE_TTS_URL = "https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"
_OUTPUT_FORMAT = "raw-24khz-16bit-mono-pcm"
_DEFAULT_VOICE = "en-US-JennyNeural"
_DEFAULT_LANG = "en-US"


class AzureSpeaker:
    def __init__(
        self,
        api_key: str,
        region: str = "eastus",
        *,
        voice_name: str = _DEFAULT_VOICE,
        lang: str = _DEFAULT_LANG,
        output_sample_rate: int = 24000,
        enabled: bool = True,
        client: httpx.Client | None = None,
        stream_factory: Any = None,
    ) -> None:
        self.api_key = api_key
        self.region = region
        self.voice_name = voice_name
        self.lang = lang
        self.output_sample_rate = output_sample_rate
        self.enabled = enabled
        self._client = client or httpx.Client(timeout=30.0)
        self._stream_factory = stream_factory or self._default_stream_factory
        self._pending_text: deque[str] = deque()
        self._is_playing = False
        self._stop_event = threading.Event()
        self._utterance_path: str | Path | None = None

    @staticmethod
    def _default_stream_factory(samplerate: int) -> sd.RawOutputStream:
        return sd.RawOutputStream(samplerate=samplerate, channels=1, dtype="int16")

    def set_utterance_path(self, path: str | Path | None) -> None:
        self._utterance_path = path

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
        ssml = self._build_ssml_for_text(text)
        start = time.perf_counter()
        thread = threading.Thread(
            target=self._stream_audio, args=(ssml,), daemon=True
        )
        thread.start()
        return time.perf_counter() - start

    def _build_ssml_for_text(self, text: str) -> str:
        utterance_path = self._utterance_path
        self._utterance_path = None

        if utterance_path is not None:
            try:
                features = extract_prosody(utterance_path, transcript=text)
                return build_ssml(
                    text,
                    features,
                    voice_name=self.voice_name,
                    lang=self.lang,
                )
            except Exception:
                pass

        # Neutral defaults when no utterance path or extraction fails
        neutral = ProsodyFeatures(
            mean_pitch_hz=0.0,
            pitch_range_hz=0.0,
            mean_intensity_db=0.0,
            speaking_rate_ratio=1.0,
            duration_seconds=0.0,
        )
        return build_ssml(
            text, neutral, voice_name=self.voice_name, lang=self.lang
        )

    def _stream_audio(self, ssml: str) -> None:
        url = _AZURE_TTS_URL.format(region=self.region)
        stream: Any = None
        try:
            with self._client.stream(
                "POST",
                url,
                headers={
                    "Ocp-Apim-Subscription-Key": self.api_key,
                    "Content-Type": "application/ssml+xml",
                    "X-Microsoft-OutputFormat": _OUTPUT_FORMAT,
                },
                content=ssml,
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
