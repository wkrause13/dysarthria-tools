from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from whisper_poc.speaker_gate import (
    EnrollmentSpeakerGate,
    resolve_enrollment_paths,
)


def _write_tone(path: Path, *, frequency: float, seconds: float = 1.0, sample_rate: int = 16000) -> None:
    t = np.linspace(0.0, seconds, int(sample_rate * seconds), endpoint=False)
    audio = 0.2 * np.sin(2 * np.pi * frequency * t)
    sf.write(path, audio, sample_rate)


def test_resolve_enrollment_paths_supports_explicit_paths_and_directory(tmp_path):
    wav_a = tmp_path / "a.wav"
    wav_b = tmp_path / "b.wav"
    subdir = tmp_path / "enroll"
    subdir.mkdir()
    wav_c = subdir / "c.wav"
    for wav in [wav_a, wav_b, wav_c]:
        _write_tone(wav, frequency=220.0)

    paths = resolve_enrollment_paths(
        enrollment_wavs=[str(wav_a), str(wav_b)],
        enrollment_dir=str(subdir),
    )

    assert paths == [wav_a, wav_b, wav_c]


def test_gate_scores_same_speaker_higher_than_different_speaker(tmp_path):
    enroll_a = tmp_path / "enroll_a.wav"
    enroll_b = tmp_path / "enroll_b.wav"
    same = tmp_path / "same.wav"
    different = tmp_path / "different.wav"
    _write_tone(enroll_a, frequency=220.0)
    _write_tone(enroll_b, frequency=223.0)
    _write_tone(same, frequency=221.0)
    _write_tone(different, frequency=660.0)

    gate = EnrollmentSpeakerGate.from_paths(
        [enroll_a, enroll_b],
        threshold=0.0,
    )

    same_score = gate.score_wav(same)
    different_score = gate.score_wav(different)

    assert same_score > different_score


def test_gate_raises_when_no_enrollment_audio_is_provided(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(ValueError):
        EnrollmentSpeakerGate.from_paths([], threshold=0.8)

    with pytest.raises(ValueError):
        resolve_enrollment_paths(enrollment_wavs=None, enrollment_dir=str(empty_dir))
