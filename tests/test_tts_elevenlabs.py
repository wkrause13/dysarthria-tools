from __future__ import annotations

import threading
import time
from collections import deque
from unittest.mock import MagicMock

import httpx

from whisper_poc.tts_elevenlabs import ElevenLabsSpeaker


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

    def stream(self, method, url, *, headers=None, params=None, json=None):
        self.calls.append(
            {"method": method, "url": url, "headers": headers, "params": params, "json": json}
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
) -> tuple[ElevenLabsSpeaker, FakeClient, FakeOutputStream]:
    client = client or FakeClient()
    stream = stream or FakeOutputStream()
    speaker = ElevenLabsSpeaker(
        api_key="test-key",
        voice_id="test-voice",
        enabled=enabled,
        client=client,
        stream_factory=lambda sr: stream,
    )
    return speaker, client, stream


def _wait_for_idle(speaker: ElevenLabsSpeaker, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while speaker.is_speaking() and time.monotonic() < deadline:
        time.sleep(0.01)


def test_speak_streams_audio_and_returns_launch_time():
    speaker, client, stream = _make_speaker()

    runtime = speaker.speak("hello world")
    _wait_for_idle(speaker)

    assert runtime >= 0.0
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["method"] == "POST"
    assert "test-voice" in call["url"]
    assert call["headers"]["xi-api-key"] == "test-key"
    assert call["json"]["text"] == "hello world"
    assert call["json"]["model_id"] == "eleven_flash_v2_5"
    assert stream.started
    assert stream.written == [PCM_CHUNK]
    assert stream.stopped
    assert stream.closed
    assert speaker.is_speaking() is False


def test_speak_queues_when_already_playing():
    client = FakeClient()
    stream = FakeOutputStream()
    speaker = ElevenLabsSpeaker(
        api_key="k",
        voice_id="v",
        enabled=True,
        client=client,
        stream_factory=lambda sr: stream,
    )

    # Simulate _is_playing being True so speak() queues instead of starting
    speaker._is_playing = True

    queued_runtime = speaker.speak("second")
    assert queued_runtime == 0.0
    assert speaker.has_pending() is True
    assert len(client.calls) == 0


def test_stop_interrupts_playback():
    # Use multiple chunks so we have time to stop mid-stream
    slow_chunks = [PCM_CHUNK] * 50
    barrier = threading.Event()

    class SlowClient:
        def __init__(self):
            self.calls = []

        def stream(self, method, url, *, headers=None, params=None, json=None):
            self.calls.append({"method": method, "url": url})

            class SlowResponse:
                status_code = 200

                def raise_for_status(self):
                    pass

                def iter_bytes(self_, chunk_size=4096):
                    for chunk in slow_chunks:
                        barrier.set()
                        time.sleep(0.01)
                        yield chunk

                def __enter__(self_):
                    return self_

                def __exit__(self_, *a):
                    pass

            return SlowResponse()

    client = SlowClient()
    stream = FakeOutputStream()
    speaker = ElevenLabsSpeaker(
        api_key="k",
        voice_id="v",
        enabled=True,
        client=client,
        stream_factory=lambda sr: stream,
    )

    speaker.speak("hello")
    barrier.wait(timeout=2.0)
    speaker.stop()
    _wait_for_idle(speaker)

    # Should have been interrupted before all 50 chunks were written
    assert len(stream.written) < 50
    assert speaker.is_speaking() is False


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

    # Manually queue text
    speaker._pending_text.append("deferred")
    assert speaker.has_pending() is True

    runtime = speaker.process_queue()
    _wait_for_idle(speaker)

    assert runtime >= 0.0
    assert speaker.has_pending() is False
    assert len(client.calls) == 1
    assert client.calls[0]["json"]["text"] == "deferred"


def test_process_queue_noop_when_playing():
    speaker, client, stream = _make_speaker()
    speaker._is_playing = True
    speaker._pending_text.append("queued")

    assert speaker.process_queue() == 0.0
    assert speaker.has_pending() is True
    assert len(client.calls) == 0
