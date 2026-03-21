# Live Metrics Logging Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend the local whisper.cpp POC so a real mic session logs per-utterance latency metrics to CSV while preserving the current terminal workflow.

**Architecture:** Keep the current capture/VAD/ASR/TTS loop intact and add a thin metrics layer in `app.py`. The VAD segmenter will expose enough timing detail to compute endpointing delay, the TTS wrapper will report launch time, and the app will optionally append one CSV row per completed utterance.

**Tech Stack:** Python 3, pytest, csv module, existing sounddevice/webrtcvad/subprocess-based whisper.cpp and macOS `say`

---

### Task 1: Add a failing CLI/config test for metrics logging

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `src/whisper_poc/app.py`

**Step 1: Write the failing test**

```python
def test_cli_accepts_metrics_csv_path(tmp_path):
    parser = build_parser()

    args = parser.parse_args([
        "--whisper-binary", "/tmp/whisper-cli",
        "--model-path", "/tmp/model.bin",
        "--metrics-csv", str(tmp_path / "metrics.csv"),
    ])

    assert args.metrics_csv == str(tmp_path / "metrics.csv")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py::test_cli_accepts_metrics_csv_path -v`
Expected: FAIL because `--metrics-csv` is not defined

**Step 3: Write minimal implementation**

Add a `--metrics-csv` optional argument to the CLI parser.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli.py::test_cli_accepts_metrics_csv_path -v`
Expected: PASS

### Task 2: Add segment timing data with TDD

**Files:**
- Modify: `tests/test_vad_segmenter.py`
- Modify: `src/whisper_poc/vad_segmenter.py`

**Step 1: Write the failing test**

```python
def test_segmenter_reports_trailing_silence_frames_on_emit():
    segmenter = VadSegmenter(sample_rate=16000, frame_ms=30, silence_ms=300)
    voiced = b"\x00\x00" * 480
    silent = b"\x00\x00" * 480

    for _ in range(3):
        assert segmenter.push(voiced, is_speech=True) is None

    utterance = None
    for _ in range(10):
        utterance = segmenter.push(silent, is_speech=False)
        if utterance is not None:
            break

    assert utterance is not None
    assert utterance.trailing_silence_frames == 10
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_vad_segmenter.py::test_segmenter_reports_trailing_silence_frames_on_emit -v`
Expected: FAIL because `trailing_silence_frames` does not exist

**Step 3: Write minimal implementation**

Extend `CompletedUtterance` with `trailing_silence_frames` and populate it when an utterance is emitted.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_vad_segmenter.py::test_segmenter_reports_trailing_silence_frames_on_emit -v`
Expected: PASS

### Task 3: Add TTS timing with TDD

**Files:**
- Modify: `tests/test_tts_local.py`
- Modify: `src/whisper_poc/tts_local.py`

**Step 1: Write the failing test**

```python
def test_speak_returns_runtime_when_enabled(monkeypatch):
    calls = []

    def fake_run(cmd, check):
        calls.append(cmd)
        return None

    monkeypatch.setattr("whisper_poc.tts_local.subprocess.run", fake_run)
    speaker = MacSpeaker(voice="Samantha", enabled=True)

    runtime = speaker.speak("hello")

    assert calls == [["say", "-v", "Samantha", "hello"]]
    assert runtime >= 0.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_tts_local.py::test_speak_returns_runtime_when_enabled -v`
Expected: FAIL because `speak()` returns `None`

**Step 3: Write minimal implementation**

Measure and return the runtime of the blocking `say` invocation. Return `0.0` when disabled or the input text is empty.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_tts_local.py::test_speak_returns_runtime_when_enabled -v`
Expected: PASS

### Task 4: Add CSV metrics helpers with TDD

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `src/whisper_poc/app.py`

**Step 1: Write the failing test**

```python
def test_append_metrics_row_writes_header_once(tmp_path):
    metrics_path = tmp_path / "metrics.csv"

    append_metrics_row(metrics_path, {"utterance_index": 1, ...})
    append_metrics_row(metrics_path, {"utterance_index": 2, ...})

    lines = metrics_path.read_text().splitlines()
    assert lines[0].startswith("utterance_index,")
    assert len(lines) == 3
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py::test_append_metrics_row_writes_header_once -v`
Expected: FAIL because helper does not exist

**Step 3: Write minimal implementation**

Add a small `append_metrics_row()` helper in `app.py` that creates parent directories, writes a header once, and appends rows.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli.py::test_append_metrics_row_writes_header_once -v`
Expected: PASS

### Task 5: Wire end-to-end metrics into the app loop

**Files:**
- Modify: `src/whisper_poc/app.py`
- Modify: `README.md`

**Step 1: Write the failing test**

```python
def test_build_metrics_row_computes_expected_timings():
    row = build_metrics_row(...)
    assert row["speech_duration_seconds"] == 0.9
    assert row["endpoint_delay_seconds"] == 0.3
    assert row["total_post_speech_seconds"] == 0.8
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py::test_build_metrics_row_computes_expected_timings -v`
Expected: FAIL because helper does not exist

**Step 3: Write minimal implementation**

Add a pure helper that computes a metrics row from utterance/frame data and measured runtimes. Update the live loop to:
- compute speech duration from voiced frames
- compute endpoint delay from trailing silence frames
- log ASR runtime, TTS runtime, transcript length, and total post-speech delay
- append rows when `--metrics-csv` is provided
- print a compact summary that includes endpoint delay

Update the README with a recommended live-run command that includes `--metrics-csv`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli.py::test_build_metrics_row_computes_expected_timings -v`
Expected: PASS

### Task 6: Verify the local metrics workflow

**Files:**
- Modify: `README.md`

**Step 1: Run targeted tests**

Run: `pytest tests/test_cli.py tests/test_tts_local.py tests/test_vad_segmenter.py -q`
Expected: all relevant tests pass

**Step 2: Run full test suite**

Run: `pytest -q`
Expected: all tests pass

**Step 3: Verify CLI help**

Run: `PYTHONPATH=src python3 -m whisper_poc.app --help`
Expected: help output includes `--metrics-csv`
