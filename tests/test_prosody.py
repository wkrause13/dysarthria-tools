from __future__ import annotations

import struct
import math
import tempfile
import wave
import xml.etree.ElementTree as ET
from pathlib import Path

from whisper_poc.prosody import ProsodyFeatures, build_ssml, extract_prosody


def _write_sine_wav(
    path: Path,
    *,
    frequency: float = 220.0,
    duration: float = 2.0,
    sample_rate: int = 16000,
    amplitude: float = 16000.0,
) -> None:
    """Write a synthetic sine wave WAV for testing."""
    n_samples = int(sample_rate * duration)
    samples = []
    for i in range(n_samples):
        t = i / sample_rate
        value = int(amplitude * math.sin(2 * math.pi * frequency * t))
        value = max(-32768, min(32767, value))
        samples.append(value)
    pcm = struct.pack(f"<{n_samples}h", *samples)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


def test_extract_prosody_returns_features_from_wav(tmp_path: Path) -> None:
    wav_path = tmp_path / "sine.wav"
    _write_sine_wav(wav_path, frequency=220.0, duration=2.0)

    features = extract_prosody(wav_path)

    assert isinstance(features, ProsodyFeatures)
    assert features.duration_seconds > 0
    assert features.mean_intensity_db > 0
    # Speaking rate defaults to 1.0 when no transcript given
    assert features.speaking_rate_ratio == 1.0


def test_extract_prosody_speaking_rate_uses_transcript_word_count(tmp_path: Path) -> None:
    wav_path = tmp_path / "sine.wav"
    _write_sine_wav(wav_path, frequency=220.0, duration=2.0)

    # 5 words / 2 seconds = 150 wpm = ratio 1.0
    features = extract_prosody(wav_path, transcript="one two three four five")

    assert features.speaking_rate_ratio > 0
    assert abs(features.speaking_rate_ratio - 1.0) < 0.1


def test_build_ssml_wraps_text_with_prosody() -> None:
    features = ProsodyFeatures(
        mean_pitch_hz=170.0,
        pitch_range_hz=40.0,
        mean_intensity_db=60.0,
        speaking_rate_ratio=0.8,
        duration_seconds=2.0,
    )

    ssml = build_ssml("hello world", features, voice_name="en-US-JennyNeural")

    assert "<prosody" in ssml
    assert 'rate="' in ssml
    assert 'pitch="' in ssml
    assert 'volume="' in ssml
    assert "hello world" in ssml
    assert "en-US-JennyNeural" in ssml


def test_build_ssml_produces_valid_xml() -> None:
    features = ProsodyFeatures(
        mean_pitch_hz=200.0,
        pitch_range_hz=50.0,
        mean_intensity_db=70.0,
        speaking_rate_ratio=1.2,
        duration_seconds=3.0,
    )

    ssml = build_ssml("test text", features)

    # Should parse without error
    root = ET.fromstring(ssml)
    assert root.tag == "{http://www.w3.org/2001/10/synthesis}speak"


def test_build_ssml_clamps_extreme_rate() -> None:
    features = ProsodyFeatures(
        mean_pitch_hz=190.0,
        pitch_range_hz=0.0,
        mean_intensity_db=65.0,
        speaking_rate_ratio=5.0,  # Way above 2.0x max
        duration_seconds=1.0,
    )

    ssml = build_ssml("fast speech", features)

    # Rate should be clamped to +100% (2.0x)
    assert 'rate="+100%"' in ssml


def test_build_ssml_neutral_defaults() -> None:
    features = ProsodyFeatures(
        mean_pitch_hz=0.0,
        pitch_range_hz=0.0,
        mean_intensity_db=0.0,
        speaking_rate_ratio=1.0,
        duration_seconds=0.0,
    )

    ssml = build_ssml("neutral", features)

    assert 'rate="+0%"' in ssml
    assert 'pitch="+0%"' in ssml
    assert 'volume="+0%"' in ssml
