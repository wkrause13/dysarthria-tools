# Enrollment Speaker Gate Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a simple enrollment-based speaker gate that filters whole utterances to the intended speaker before Whisper ASR runs.

**Architecture:** Keep the current endpointed ASR -> final TTS flow. Add an optional speaker-gate module that builds an enrollment profile from one or more reference WAVs, computes a lightweight embedding for each candidate utterance, and rejects non-matching utterances before ASR.

**Tech Stack:** Python 3, `numpy`, `soundfile`, `webrtcvad`, `whisper.cpp`, pytest

---

### Task 1: Add pure speaker-gate module with TDD

**Files:**
- Create: `src/whisper_poc/speaker_gate.py`
- Create: `tests/test_speaker_gate.py`

**Step 1: Write the failing tests**

Add tests covering:
- enrollment path resolution from repeated WAV args and/or a directory
- embedding/similarity prefers same synthetic speaker over a different one
- gate raises on missing enrollment data

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_speaker_gate.py -q`
Expected: FAIL because module does not exist

**Step 3: Write minimal implementation**

Implement:
- enrollment path loading
- lightweight utterance embedding
- cosine similarity scoring
- `EnrollmentSpeakerGate.matches_wav(...)`

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_speaker_gate.py -q`
Expected: PASS

### Task 2: Add CLI wiring and endpointed runtime gating with TDD

**Files:**
- Modify: `src/whisper_poc/app.py`
- Modify: `tests/test_cli.py`
- Create: `tests/test_endpointed_runtime.py`

**Step 1: Write the failing tests**

Add tests showing:
- CLI accepts enrollment-related flags
- endpointed mode skips ASR when a fake gate rejects an utterance
- endpointed mode still transcribes when a fake gate accepts it

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_cli.py tests/test_endpointed_runtime.py -q`
Expected: FAIL because flags/runtime gate do not exist

**Step 3: Write minimal implementation**

Add CLI support for:
- `--enrollment-wav` (repeatable)
- `--enrollment-dir`
- `--speaker-threshold`

In endpointed mode:
- build gate only if enrollment inputs are provided
- score utterance after duration/RMS checks
- skip Whisper if speaker match fails

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_cli.py tests/test_endpointed_runtime.py -q`
Expected: PASS

### Task 3: Document and verify full slice

**Files:**
- Modify: `README.md`

**Step 1: Update docs**

Document:
- how to prepare enrollment WAVs
- how to run with `--enrollment-wav` or `--enrollment-dir`
- that the gate is a lightweight heuristic, not full diarization

**Step 2: Run targeted verification**

Run: `python3 -m pytest tests/test_speaker_gate.py tests/test_cli.py tests/test_endpointed_runtime.py -q`
Expected: PASS

**Step 3: Run full verification**

Run: `python3 -m pytest -q`
Expected: PASS
