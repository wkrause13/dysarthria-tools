# Selective Rollback To Endpointed ASR + Final TTS Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove the streaming/preview/partial-decode path and return the app to a single endpointed ASR -> final TTS flow, while keeping native macOS TTS, replay support, `min_speech_ms`, and metrics logging.

**Architecture:** Keep one runtime path: capture or replay audio, endpoint with VAD, transcribe the completed utterance once, and optionally speak the final transcript. Preserve local utilities that help testing and evaluation, but delete streaming-only state, files, flags, and tests.

**Tech Stack:** Python 3, `whisper.cpp` CLI, `webrtcvad`, native macOS TTS (`NSSpeechSynthesizer`), pytest

---

### Task 1: Remove streaming CLI/runtime behavior with TDD

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `src/whisper_poc/app.py`

**Step 1: Write the failing test**

Add/update parser tests showing:
- `--streaming` and other streaming-only flags are rejected
- `--input-wav`, `--min-speech-ms`, and `--metrics-csv` still work

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_cli.py -q`
Expected: FAIL because streaming flags still exist

**Step 3: Write minimal implementation**

Remove:
- `--streaming`
- `--decode-interval-ms`
- `--partial-window-ms`
- `--stability-passes`
- `--min-commit-words`
- `--streaming-start-ms`

Keep:
- `--input-wav`
- `--min-speech-ms`
- `--metrics-csv`
- `--no-tts`

Refactor `app.py` to a single endpointed run path only.

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_cli.py -q`
Expected: PASS

### Task 2: Remove streaming-only code and tests, keep replay utilities

**Files:**
- Delete: `src/whisper_poc/streaming_state.py`
- Delete: `tests/test_streaming_state.py`
- Delete: `tests/test_streaming_runtime.py`
- Modify: `src/whisper_poc/audio_capture.py`
- Modify: `tests/test_audio_capture.py`

**Step 1: Verify replay tests still cover the kept utility**

Keep `iter_wav_frames` and its tests because replay remains in scope.

**Step 2: Remove streaming-only implementation**

Delete:
- `RollingPcmBuffer`
- streaming imports/usages
- streaming-only tests and plan references if needed

**Step 3: Run targeted tests**

Run: `python3 -m pytest tests/test_audio_capture.py tests/test_cli.py -q`
Expected: PASS

### Task 3: Simplify ASR/TTS docs and verify full suite

**Files:**
- Modify: `README.md`
- Modify: `tests/test_asr_whispercpp.py`
- Modify: `src/whisper_poc/asr_whispercpp.py`

**Step 1: Keep only endpointed behavior in docs**

Document:
- live mic endpointed ASR -> final TTS
- replay mode through `--input-wav`
- native macOS TTS queue behavior

Remove streaming references from the README.

**Step 2: Keep or simplify ASR flags**

If `no_speech_thold` still helps the endpointed path and is low-cost, keep it. Otherwise remove it too. Ensure tests reflect the kept interface.

**Step 3: Run targeted verification**

Run: `python3 -m pytest tests/test_asr_whispercpp.py tests/test_tts_local.py tests/test_audio_capture.py tests/test_cli.py -q`
Expected: PASS

**Step 4: Run full verification**

Run: `python3 -m pytest -q`
Expected: PASS

**Step 5: Manual replay smoke**

Run endpointed replay on `audio_001.wav` and verify:
- one final transcript is produced
- no preview/streaming output appears
- TTS can still be disabled with `--no-tts`
