# TTS Fallback: ElevenLabs to Local

## Problem

When `--tts-provider elevenlabs` is selected, the app crashes if the API key is missing and silently produces no audio if the API is unreachable. The user hears nothing with no indication of what went wrong.

## Design

### FallbackSpeaker wrapper (`src/whisper_poc/tts_fallback.py`)

A new class implementing the same duck-typed interface as all other speakers (`speak`, `stop`, `is_speaking`, `process_queue`, `has_pending`).

- Holds a `primary` and `fallback` speaker.
- `speak(text)` delegates to primary. If primary's streaming fails at runtime, an error callback dispatches `fallback.speak(text)` so the utterance is not lost.
- `is_speaking()` returns `True` if either speaker is active.
- `process_queue()` and `has_pending()` delegate to both.
- `stop()` stops both.
- `set_utterance_path()` forwards to primary if it supports it.
- Logs a message on fallback: `[tts] elevenlabs failed, falling back to local`.

A factory function `build_speaker(args, config)` consolidates speaker construction:

1. `elevenlabs` provider: if `ELEVENLABS_API_KEY` missing, log and return `MacSpeaker`. If present, return `FallbackSpeaker(primary=ElevenLabsSpeaker, fallback=MacSpeaker)`.
2. `azure` provider: unchanged (hard error on missing key).
3. `native` provider: return `MacSpeaker` directly.

### ElevenLabsSpeaker changes

Add an optional `on_error: Callable[[str], None] | None = None` parameter to `__init__`.

In `_stream_audio`, replace `except Exception: pass` with:

```python
except Exception:
    if self._on_error is not None:
        self._on_error(text)
```

This surfaces the original text to the wrapper on failure. No callback = silent ignore (backward compatible).

### app.py changes

Replace the inline provider selection block (`run_endpointed_mode` lines 198-227) with a single call to `build_speaker(args, config)`. Everything downstream is unchanged since the wrapper's interface is identical.

### Testing (`tests/test_tts_fallback.py`)

All tests use mock/stub speakers (no HTTP or NSSpeechSynthesizer).

- Missing API key returns MacSpeaker, not an error.
- Primary error callback triggers fallback `speak()` with the same text.
- Happy path delegates to primary only; fallback is never called.
- `is_speaking()`, `stop()`, `process_queue()` delegate correctly to both speakers.
- `set_utterance_path` forwarded only when primary supports it.

Existing `test_tts_elevenlabs.py` and `test_tts_local.py` are unchanged.
