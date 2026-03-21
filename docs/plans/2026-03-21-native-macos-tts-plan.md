# Native macOS TTS Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the blocking `say` subprocess TTS path with native macOS speech synthesis that starts asynchronously and can be interrupted when the user begins speaking again.

**Architecture:** Keep the current local VAD and `whisper.cpp` ASR flow unchanged. Replace the TTS wrapper with a small `NSSpeechSynthesizer`-based adapter loaded lazily via PyObjC, expose `speak()` as a fast launch operation and `stop()` for barge-in, and update the app loop to interrupt TTS as soon as incoming speech is detected.

**Tech Stack:** Python 3, PyObjC/Cocoa (`NSSpeechSynthesizer`), pytest, existing sounddevice/webrtcvad/whisper.cpp stack

---

### Task 1: Add native TTS dependency metadata

**Files:**
- Modify: `pyproject.toml`

**Step 1: Write the failing test**

There is no code test here; this is dependency metadata needed for runtime.

**Step 2: Write minimal implementation**

Add `pyobjc-framework-Cocoa` to project dependencies.

**Step 3: Verify the file contains the dependency**

Run: `python3 - <<'PY'
from pathlib import Path
print('pyobjc-framework-Cocoa' in Path('pyproject.toml').read_text())
PY`
Expected: `True`

### Task 2: Replace the `say` wrapper with a native speech backend using TDD

**Files:**
- Modify: `tests/test_tts_local.py`
- Modify: `src/whisper_poc/tts_local.py`

**Step 1: Write the failing tests**

```python
def test_speaker_starts_native_speech_and_returns_launch_time():
    ...

def test_stop_interrupts_active_speech():
    ...
```

The tests should assert that:
- `speak()` calls `startSpeakingString_()` on the synthesizer and returns a non-negative runtime
- `stop()` forwards to the synthesizer stop method
- disabled or empty speech still returns `0.0`

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_tts_local.py -v`
Expected: FAIL because the current class only shells out to `say`

**Step 3: Write minimal implementation**

Implement a lazy-loading native backend:
- import `NSSpeechSynthesizer` only when constructing the real speaker
- allow backend injection for tests
- set the requested voice if provided
- call `startSpeakingString_()` asynchronously in `speak()`
- add `stop()` and `is_speaking()` helpers
- keep `speak()` fast and non-blocking by returning launch time only

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_tts_local.py -v`
Expected: PASS

### Task 3: Interrupt speech playback when the user starts talking, using TDD

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `src/whisper_poc/app.py`

**Step 1: Write the failing test**

```python
def test_should_interrupt_tts_on_new_speech_frame():
    assert should_interrupt_tts(is_speech=True, speaker_is_speaking=True) is True
    assert should_interrupt_tts(is_speech=False, speaker_is_speaking=True) is False
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_cli.py::test_should_interrupt_tts_on_new_speech_frame -v`
Expected: FAIL because helper does not exist

**Step 3: Write minimal implementation**

Add a pure helper and update the app loop so that before VAD segmentation is processed, any new speech frame stops current TTS playback.

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_cli.py::test_should_interrupt_tts_on_new_speech_frame -v`
Expected: PASS

### Task 4: Update live metrics and docs

**Files:**
- Modify: `src/whisper_poc/app.py`
- Modify: `README.md`

**Step 1: Write the failing test**

```python
def test_build_metrics_row_accepts_async_tts_launch_time():
    row = build_metrics_row(..., tts_runtime_seconds=0.02, ...)
    assert row['tts_runtime_seconds'] == 0.02
```

If this already passes, skip new test code and keep current helper.

**Step 2: Write minimal implementation**

Update README to describe the new TTS behavior and recommended run commands for:
- ASR-only timing runs
- local clear-speech runs with native TTS enabled

**Step 3: Run verification**

Run: `PYTHONPATH=src python3 -m whisper_poc.app --help`
Expected: help still prints successfully

### Task 5: Verify the complete local POC

**Files:**
- Modify: `README.md`

**Step 1: Run targeted tests**

Run: `python3 -m pytest tests/test_tts_local.py tests/test_cli.py -q`
Expected: all targeted tests pass

**Step 2: Run full test suite**

Run: `python3 -m pytest -q`
Expected: all tests pass

**Step 3: Inspect git diff**

Run: `git diff -- src/whisper_poc/tts_local.py src/whisper_poc/app.py README.md pyproject.toml tests/test_tts_local.py tests/test_cli.py`
Expected: only the intended native TTS changes appear
