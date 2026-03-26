from __future__ import annotations

import argparse
import time
from collections import deque
from unittest.mock import MagicMock

from whisper_poc.tts_fallback import FallbackSpeaker, build_speaker
from whisper_poc.tts_local import MacSpeaker


class StubSpeaker:
    """Minimal speaker stub for testing the wrapper."""

    def __init__(self, *, fail_on_speak: bool = False):
        self.spoken: list[str] = []
        self.stopped = False
        self.queue_called = False
        self._speaking = False
        self._pending: deque[str] = deque()
        self._fail_on_speak = fail_on_speak
        self._on_error = None

    def speak(self, text: str) -> float:
        if self._fail_on_speak:
            if self._on_error is not None:
                self._on_error(text)
            return 0.0
        self.spoken.append(text)
        self._speaking = True
        return 0.001

    def is_speaking(self) -> bool:
        return self._speaking

    def stop(self) -> None:
        self.stopped = True
        self._speaking = False

    def process_queue(self) -> float:
        self.queue_called = True
        return 0.0

    def has_pending(self) -> bool:
        return bool(self._pending)


# --- FallbackSpeaker tests ---


def test_fallback_speak_delegates_to_primary():
    primary = StubSpeaker()
    fallback = StubSpeaker()
    speaker = FallbackSpeaker(primary=primary, fallback=fallback)

    speaker.speak("hello")

    assert primary.spoken == ["hello"]
    assert fallback.spoken == []


def test_fallback_speak_uses_fallback_on_primary_error():
    primary = StubSpeaker(fail_on_speak=True)
    fallback = StubSpeaker()
    speaker = FallbackSpeaker(primary=primary, fallback=fallback)

    speaker.speak("hello")
    # Give the error callback a moment if async
    time.sleep(0.05)

    assert fallback.spoken == ["hello"]


def test_fallback_is_speaking_checks_both():
    primary = StubSpeaker()
    fallback = StubSpeaker()
    speaker = FallbackSpeaker(primary=primary, fallback=fallback)

    assert speaker.is_speaking() is False

    primary._speaking = True
    assert speaker.is_speaking() is True

    primary._speaking = False
    fallback._speaking = True
    assert speaker.is_speaking() is True


def test_fallback_stop_stops_both():
    primary = StubSpeaker()
    fallback = StubSpeaker()
    speaker = FallbackSpeaker(primary=primary, fallback=fallback)

    primary._speaking = True
    speaker.stop()

    assert primary.stopped is True
    assert fallback.stopped is True


def test_fallback_process_queue_delegates_to_both():
    primary = StubSpeaker()
    fallback = StubSpeaker()
    speaker = FallbackSpeaker(primary=primary, fallback=fallback)

    speaker.process_queue()

    assert primary.queue_called is True
    assert fallback.queue_called is True


def test_fallback_has_pending_checks_both():
    primary = StubSpeaker()
    fallback = StubSpeaker()
    speaker = FallbackSpeaker(primary=primary, fallback=fallback)

    assert speaker.has_pending() is False

    primary._pending.append("x")
    assert speaker.has_pending() is True

    primary._pending.clear()
    fallback._pending.append("y")
    assert speaker.has_pending() is True


def test_fallback_set_utterance_path_forwards_when_supported():
    primary = StubSpeaker()
    primary.set_utterance_path = MagicMock()
    fallback = StubSpeaker()
    speaker = FallbackSpeaker(primary=primary, fallback=fallback)

    speaker.set_utterance_path("/some/path")

    primary.set_utterance_path.assert_called_once_with("/some/path")


def test_fallback_set_utterance_path_noop_when_unsupported():
    primary = StubSpeaker()  # no set_utterance_path method
    fallback = StubSpeaker()
    speaker = FallbackSpeaker(primary=primary, fallback=fallback)

    # Should not raise
    speaker.set_utterance_path("/some/path")


# --- build_speaker tests ---


def _make_args(**overrides) -> argparse.Namespace:
    defaults = {
        "tts_provider": "native",
        "voice": "Samantha",
        "elevenlabs_voice_id": None,
        "elevenlabs_model": "eleven_flash_v2_5",
        "azure_speech_region": "eastus",
        "azure_voice": "en-US-JennyNeural",
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _make_config(**overrides):
    from whisper_poc.config import AppConfig

    return AppConfig(**overrides)


def test_build_speaker_native_returns_mac_speaker():
    args = _make_args(tts_provider="native")
    config = _make_config(tts_enabled=True)

    speaker = build_speaker(args, config)

    assert isinstance(speaker, MacSpeaker)


def test_build_speaker_elevenlabs_missing_key_returns_mac_speaker(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    args = _make_args(tts_provider="elevenlabs")
    config = _make_config(tts_enabled=True)

    logs: list[str] = []
    speaker = build_speaker(args, config, log=logs.append)

    assert isinstance(speaker, MacSpeaker)
    assert any("fallback" in msg.lower() or "local" in msg.lower() for msg in logs)


def test_build_speaker_elevenlabs_with_key_returns_fallback_speaker(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
    args = _make_args(tts_provider="elevenlabs")
    config = _make_config(tts_enabled=True)

    speaker = build_speaker(args, config)

    assert isinstance(speaker, FallbackSpeaker)
