from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import parselmouth

_BASELINE_WPM = 150.0
_BASELINE_PITCH_HZ = 190.0
_BASELINE_INTENSITY_DB = 65.0

_MIN_RATE = 0.5
_MAX_RATE = 2.0
_MAX_PITCH_OFFSET_PCT = 50.0


@dataclass(slots=True)
class ProsodyFeatures:
    mean_pitch_hz: float
    pitch_range_hz: float
    mean_intensity_db: float
    speaking_rate_ratio: float  # 1.0 = 150 wpm baseline
    duration_seconds: float


def extract_prosody(
    wav_path: str | Path,
    *,
    transcript: str | None = None,
) -> ProsodyFeatures:
    """Extract prosody features from a WAV file using parselmouth."""
    snd = parselmouth.Sound(str(wav_path))
    duration = snd.duration

    # Pitch (F0)
    pitch = snd.to_pitch()
    pitch_values = pitch.selected_array["frequency"]
    voiced = pitch_values[pitch_values > 0]
    if len(voiced) > 0:
        mean_pitch = float(np.mean(voiced))
        pitch_range = float(np.max(voiced) - np.min(voiced))
    else:
        mean_pitch = 0.0
        pitch_range = 0.0

    # Intensity
    intensity = snd.to_intensity()
    intensity_values = intensity.values.flatten()
    valid_intensity = intensity_values[np.isfinite(intensity_values)]
    if len(valid_intensity) > 0:
        mean_intensity = float(np.mean(valid_intensity))
    else:
        mean_intensity = 0.0

    # Speaking rate
    if transcript is not None and duration > 0:
        word_count = len(transcript.split())
        wpm = word_count / (duration / 60.0)
        speaking_rate_ratio = wpm / _BASELINE_WPM
    else:
        speaking_rate_ratio = 1.0

    return ProsodyFeatures(
        mean_pitch_hz=mean_pitch,
        pitch_range_hz=pitch_range,
        mean_intensity_db=mean_intensity,
        speaking_rate_ratio=speaking_rate_ratio,
        duration_seconds=duration,
    )


def build_ssml(
    text: str,
    features: ProsodyFeatures,
    *,
    voice_name: str = "en-US-JennyNeural",
    lang: str = "en-US",
) -> str:
    """Build SSML with prosody attributes from extracted features."""
    # Rate: percentage offset from 1.0 ratio, clamped to 0.5x-2.0x
    clamped_rate = max(_MIN_RATE, min(_MAX_RATE, features.speaking_rate_ratio))
    rate_pct = round((clamped_rate - 1.0) * 100)

    # Pitch: percentage offset from baseline, clamped to +/-50%
    if features.mean_pitch_hz > 0:
        pitch_pct = round(
            (features.mean_pitch_hz - _BASELINE_PITCH_HZ) / _BASELINE_PITCH_HZ * 100
        )
        pitch_pct = max(-_MAX_PITCH_OFFSET_PCT, min(_MAX_PITCH_OFFSET_PCT, pitch_pct))
        pitch_pct = int(pitch_pct)
    else:
        pitch_pct = 0

    # Volume: dB offset from baseline, mapped to percentage
    if features.mean_intensity_db > 0:
        vol_pct = round(
            (features.mean_intensity_db - _BASELINE_INTENSITY_DB)
            / _BASELINE_INTENSITY_DB
            * 100
        )
        vol_pct = max(-50, min(50, vol_pct))
    else:
        vol_pct = 0

    rate_str = f"+{rate_pct}%" if rate_pct >= 0 else f"{rate_pct}%"
    pitch_str = f"+{pitch_pct}%" if pitch_pct >= 0 else f"{pitch_pct}%"
    vol_str = f"+{vol_pct}%" if vol_pct >= 0 else f"{vol_pct}%"

    return (
        f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"'
        f' xml:lang="{lang}">\n'
        f'  <voice name="{voice_name}">\n'
        f'    <prosody rate="{rate_str}" pitch="{pitch_str}" volume="{vol_str}">\n'
        f"      {text}\n"
        f"    </prosody>\n"
        f"  </voice>\n"
        f"</speak>"
    )
