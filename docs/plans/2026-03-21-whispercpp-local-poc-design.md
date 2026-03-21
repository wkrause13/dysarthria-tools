# Whisper.cpp Local POC Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Mac-local proof of concept that captures microphone audio, endpoints utterances with VAD, transcribes them with whisper.cpp, and speaks the transcript back with local macOS TTS while logging latency.

**Architecture:** The POC is a small Python application composed of independent modules for audio capture, endpointing, whisper.cpp invocation, and local TTS. Live mic I/O stays thin and configurable, while core behavior such as segmentation rules and CLI construction is covered by tests so the first iteration stays debuggable.

**Tech Stack:** Python 3, pytest, sounddevice, soundfile, webrtcvad, subprocess integration with whisper.cpp, macOS `say`

---

### Task 1: Scaffold the local POC package

**Files:**
- Create: `pyproject.toml`
- Create: `src/whisper_poc/__init__.py`
- Create: `src/whisper_poc/config.py`
- Create: `src/whisper_poc/app.py`

**Step 1: Write the failing test**

```python
def test_default_config_exposes_expected_paths():
    from whisper_poc.config import AppConfig

    cfg = AppConfig.default()

    assert cfg.sample_rate == 16000
    assert cfg.channels == 1
    assert cfg.tts_enabled is True
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with import error / missing module

**Step 3: Write minimal implementation**

Create a small `AppConfig` dataclass with sane defaults and a `default()` constructor for the local POC.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS

### Task 2: Build endpointing logic with TDD

**Files:**
- Create: `src/whisper_poc/vad_segmenter.py`
- Test: `tests/test_vad_segmenter.py`

**Step 1: Write the failing test**

```python
def test_segmenter_emits_utterance_after_configured_silence():
    segmenter = VadSegmenter(sample_rate=16000, frame_ms=30, silence_ms=300)
    # feed voiced then silent frames
    ...
    assert utterance is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_vad_segmenter.py -v`
Expected: FAIL with missing class

**Step 3: Write minimal implementation**

Implement a frame-based segmenter that is agnostic to the underlying VAD model. It should accept boolean `is_speech` decisions plus raw bytes and emit completed utterances after enough trailing silence.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_vad_segmenter.py -v`
Expected: PASS

### Task 3: Build the whisper.cpp runner with TDD

**Files:**
- Create: `src/whisper_poc/asr_whispercpp.py`
- Test: `tests/test_asr_whispercpp.py`

**Step 1: Write the failing test**

```python
def test_runner_builds_expected_whisper_cpp_command(tmp_path):
    runner = WhisperCppRunner(binary_path="/tmp/main", model_path="/tmp/model.bin")
    cmd = runner.build_command(tmp_path / "sample.wav")
    assert cmd[:2] == ["/tmp/main", "-m"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_asr_whispercpp.py -v`
Expected: FAIL with missing module/class

**Step 3: Write minimal implementation**

Implement a small wrapper that builds the CLI command, executes whisper.cpp, and extracts transcript text from stdout.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_asr_whispercpp.py -v`
Expected: PASS

### Task 4: Add local TTS wrapper with TDD

**Files:**
- Create: `src/whisper_poc/tts_local.py`
- Test: `tests/test_tts_local.py`

**Step 1: Write the failing test**

```python
def test_tts_command_uses_macos_say_binary():
    speaker = MacSpeaker(voice="Samantha")
    assert speaker.build_command("hello") == ["say", "-v", "Samantha", "hello"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_tts_local.py -v`
Expected: FAIL with missing class

**Step 3: Write minimal implementation**

Implement a wrapper around the `say` command with an option to disable TTS for measurement runs.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_tts_local.py -v`
Expected: PASS

### Task 5: Wire an end-to-end CLI loop

**Files:**
- Create: `src/whisper_poc/audio_capture.py`
- Modify: `src/whisper_poc/app.py`
- Create: `README.md`

**Step 1: Write the failing test**

```python
def test_cli_requires_whisper_binary_and_model_paths():
    ...
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Create a terminal-only app that:
- captures mic frames
- runs VAD decisions
- writes utterances to temp WAV files
- calls whisper.cpp
- prints transcript and latency
- optionally speaks result with `say`

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli.py -v`
Expected: PASS

### Task 6: Verify the local POC manually

**Files:**
- Modify: `README.md`

**Step 1: Write a smoke command**

Document the exact command to run the app locally, including required whisper.cpp binary/model paths.

**Step 2: Run smoke verification**

Run: `PYTHONPATH=src python3 -m whisper_poc.app --help`
Expected: CLI help text prints successfully

**Step 3: Run test suite**

Run: `pytest -q`
Expected: all tests pass
