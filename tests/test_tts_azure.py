from __future__ import annotations

import math
import struct
import tempfile
import time
import wave
from pathlib import Path
from unittest.mock import MagicMock

import httpx

from whisper_poc.tts_azure import AzureSpeaker


PCM_CHUNK = b"\x00\x01" * 2048  # 4096 bytes of fake PCM data


class FakeResponse:
    """Simulates an httpx streaming response."""

    def __init__(self, chunks: list[bytes] | None = None, status_code: int = 200):
        self._chunks = chunks if chunks is not None else [PCM_CHUNK]
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "error", request=MagicMock(), response=MagicMock()
            )

    def iter_bytes(self, chunk_size: int = 4096) -> list[bytes]:
        return list(self._chunks)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class FakeClient:
    """Replaces httpx.Client; records calls and yields canned chunks."""

    def __init__(self, chunks: list[bytes] | None = None):
        self.calls: list[dict] = []
        self._chunks = chunks

    def stream(self, method, url, *, headers=None, content=None):
        self.calls.append(
            {"method": method, "url": url, "headers": headers, "content": content}
        )
        return FakeResponse(self._chunks)


class FakeOutputStream:
    """Replaces sounddevice.RawOutputStream."""

    def __init__(self):
        self.written: list[bytes] = []
        self.started = False
        self.stopped = False
        self.closed = False

    def start(self):
        self.started = True

    def write(self, data):
        self.written.append(data)

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


def _make_speaker(
    client: FakeClient | None = None,
    stream: FakeOutputStream | None = None,
    enabled: bool = True,
) -> tuple[AzureSpeaker, FakeClient, FakeOutputStream]:
    client = client or FakeClient()
    stream = stream or FakeOutputStream()
    speaker = AzureSpeaker(
        api_key="test-key",
        region="eastus",
        enabled=enabled,
        client=client,
        stream_factory=lambda sr: stream,
    )
    return speaker, client, stream


def _wait_for_idle(speaker: AzureSpeaker, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while speaker.is_speaking() and time.monotonic() < deadline:
        time.sleep(0.01)


def _write_sine_wav(path: Path, *, frequency: float = 220.0, duration: float = 1.0) -> None:
    sample_rate = 16000
    n_samples = int(sample_rate * duration)
    samples = []
    for i in range(n_samples):
        t = i / sample_rate
        value = int(16000.0 * math.sin(2 * math.pi * frequency * t))
        value = max(-32768, min(32767, value))
        samples.append(value)
    pcm = struct.pack(f"<{n_samples}h", *samples)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


def test_speak_sends_ssml_and_streams_audio():
    speaker, client, stream = _make_speaker()

    runtime = speaker.speak("hello world")
    _wait_for_idle(speaker)

    assert runtime >= 0.0
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["method"] == "POST"
    assert "eastus.tts.speech.microsoft.com" in call["url"]
    assert call["headers"]["Ocp-Apim-Subscription-Key"] == "test-key"
    assert call["headers"]["Content-Type"] == "application/ssml+xml"
    assert call["headers"]["X-Microsoft-OutputFormat"] == "raw-24khz-16bit-mono-pcm"
    # Content should be SSML
    assert "<speak" in call["content"]
    assert "hello world" in call["content"]
    assert stream.started
    assert stream.written == [PCM_CHUNK]
    assert stream.stopped
    assert stream.closed
    assert speaker.is_speaking() is False


def test_speak_with_utterance_path_uses_prosody(tmp_path: Path):
    speaker, client, stream = _make_speaker()
    wav_path = tmp_path / "test.wav"
    _write_sine_wav(wav_path)

    speaker.set_utterance_path(wav_path)
    speaker.speak("hello world")
    _wait_for_idle(speaker)

    assert len(client.calls) == 1
    ssml = client.calls[0]["content"]
    assert "<prosody" in ssml
    assert "hello world" in ssml


def test_speak_returns_zero_when_disabled():
    speaker, client, stream = _make_speaker(enabled=False)

    assert speaker.speak("hello") == 0.0
    assert len(client.calls) == 0
    assert speaker.is_speaking() is False


def test_speak_returns_zero_for_empty_text():
    speaker, client, stream = _make_speaker()

    assert speaker.speak("") == 0.0
    assert len(client.calls) == 0


def test_process_queue_drains_pending():
    speaker, client, stream = _make_speaker()

    speaker._pending_text.append("deferred")
    assert speaker.has_pending() is True

    runtime = speaker.process_queue()
    _wait_for_idle(speaker)

    assert runtime >= 0.0
    assert speaker.has_pending() is False
    assert len(client.calls) == 1
    assert "deferred" in client.calls[0]["content"]


def test_stop_clears_pending():
    speaker, client, stream = _make_speaker()

    speaker._pending_text.append("one")
    speaker._pending_text.append("two")
    assert speaker.has_pending() is True

    speaker.stop()

    assert speaker.has_pending() is False
    assert len(speaker._pending_text) == 0


def test_set_utterance_path_cleared_after_speak(tmp_path: Path):
    speaker, client, stream = _make_speaker()
    wav_path = tmp_path / "test.wav"
    _write_sine_wav(wav_path)

    speaker.set_utterance_path(wav_path)
    speaker.speak("first")
    _wait_for_idle(speaker)

    # Path should be consumed
    assert speaker._utterance_path is None

    # Second speak should use neutral defaults
    speaker.speak("second")
    _wait_for_idle(speaker)

    assert len(client.calls) == 2
    # First call used prosody from wav
    assert "<prosody" in client.calls[0]["content"]
    # Second call also has prosody but with neutral values
    assert "<prosody" in client.calls[1]["content"]
