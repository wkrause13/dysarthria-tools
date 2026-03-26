from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from whisper_poc.config import AppConfig
from whisper_poc.tts_local import MacSpeaker


class FallbackSpeaker:
    def __init__(
        self,
        primary: Any,
        fallback: Any,
        *,
        log: Callable[[str], None] = print,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._log = log
        if hasattr(self._primary, "_on_error"):
            self._primary._on_error = self._on_primary_error

    def _on_primary_error(self, text: str) -> None:
        self._log("[tts] primary failed, falling back to local")
        self._fallback.speak(text)

    def speak(self, text: str) -> float:
        return self._primary.speak(text)

    def is_speaking(self) -> bool:
        return self._primary.is_speaking() or self._fallback.is_speaking()

    def stop(self) -> None:
        self._primary.stop()
        self._fallback.stop()

    def process_queue(self) -> float:
        self._primary.process_queue()
        return self._fallback.process_queue()

    def has_pending(self) -> bool:
        return self._primary.has_pending() or self._fallback.has_pending()

    def set_utterance_path(self, path: str | Path | None) -> None:
        if hasattr(self._primary, "set_utterance_path"):
            self._primary.set_utterance_path(path)


def build_speaker(
    args: Any,
    config: AppConfig,
    *,
    log: Callable[[str], None] = print,
) -> Any:
    if args.tts_provider == "elevenlabs":
        api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            log("[tts] ELEVENLABS_API_KEY not set, falling back to local TTS")
            return MacSpeaker(voice=config.tts_voice, enabled=config.tts_enabled)

        from whisper_poc.tts_elevenlabs import ElevenLabsSpeaker

        primary = ElevenLabsSpeaker(
            api_key=api_key,
            voice_id=args.elevenlabs_voice_id or "21m00Tcm4TlvDq8ikWAM",
            model_id=args.elevenlabs_model,
            enabled=config.tts_enabled,
        )
        fallback = MacSpeaker(voice=config.tts_voice, enabled=config.tts_enabled)
        return FallbackSpeaker(primary=primary, fallback=fallback, log=log)

    if args.tts_provider == "azure":
        azure_key = os.environ.get("AZURE_SPEECH_KEY")
        if not azure_key:
            raise SystemExit(
                "AZURE_SPEECH_KEY env var required with --tts-provider azure"
            )
        from whisper_poc.tts_azure import AzureSpeaker

        return AzureSpeaker(
            api_key=azure_key,
            region=args.azure_speech_region,
            voice_name=args.azure_voice,
            enabled=config.tts_enabled,
        )

    return MacSpeaker(voice=config.tts_voice, enabled=config.tts_enabled)
