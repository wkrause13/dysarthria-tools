from __future__ import annotations

import time
from collections import deque
from typing import Any


def _build_native_synthesizer(voice: str | None = None) -> Any:
    from AppKit import NSSpeechSynthesizer

    synthesizer = NSSpeechSynthesizer.alloc().init()
    if synthesizer is None:
        raise RuntimeError("Failed to initialize NSSpeechSynthesizer")
    if voice:
        try:
            synthesizer.setVoice_(voice)
        except Exception:
            pass
    return synthesizer


class MacSpeaker:
    def __init__(
        self,
        voice: str = "Samantha",
        enabled: bool = True,
        synthesizer: Any | None = None,
    ) -> None:
        self.voice = voice
        self.enabled = enabled
        self._synthesizer = synthesizer
        self._pending_text: deque[str] = deque()
        if self._synthesizer is not None and self.voice:
            self._apply_voice()

    def _ensure_synthesizer(self) -> Any:
        if self._synthesizer is None:
            self._synthesizer = _build_native_synthesizer(self.voice)
        return self._synthesizer

    def _apply_voice(self) -> None:
        try:
            self._synthesizer.setVoice_(self.voice)
        except Exception:
            pass

    def speak(self, text: str) -> float:
        if not self.enabled or not text:
            return 0.0
        if self.is_speaking():
            self._pending_text.append(text)
            return 0.0
        return self._start_speaking(text)

    def _start_speaking(self, text: str) -> float:
        synthesizer = self._ensure_synthesizer()
        start = time.perf_counter()
        synthesizer.startSpeakingString_(text)
        return time.perf_counter() - start

    def process_queue(self) -> float:
        if self.is_speaking() or not self._pending_text:
            return 0.0
        return self._start_speaking(self._pending_text.popleft())

    def has_pending(self) -> bool:
        return bool(self._pending_text)

    def stop(self) -> None:
        if self._synthesizer is None or not self.enabled:
            return
        self._pending_text.clear()
        self._synthesizer.stopSpeaking()

    def is_speaking(self) -> bool:
        if self._synthesizer is None or not self.enabled:
            return False
        return bool(self._synthesizer.isSpeaking())
