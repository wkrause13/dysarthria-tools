from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


def resolve_enrollment_paths(
    *,
    enrollment_wavs: list[str] | None,
    enrollment_dir: str | None,
) -> list[Path]:
    paths: list[Path] = []
    if enrollment_wavs:
        paths.extend(Path(path) for path in enrollment_wavs)
    if enrollment_dir:
        paths.extend(sorted(Path(enrollment_dir).glob("*.wav")))
    if not paths:
        raise ValueError("no enrollment audio found")
    return paths


def _load_mono_audio(path: Path, target_sample_rate: int = 16000) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sample_rate != target_sample_rate:
        raise ValueError(
            f"expected enrollment audio at {target_sample_rate} Hz, got {sample_rate} Hz"
        )
    return audio.astype(np.float32)


def _speaker_embedding(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    if audio.size == 0:
        raise ValueError("empty audio cannot be embedded")

    frame_length = int(0.025 * sample_rate)
    hop_length = int(0.010 * sample_rate)
    if audio.size < frame_length:
        audio = np.pad(audio, (0, frame_length - audio.size))

    frame_count = 1 + max(0, (audio.size - frame_length) // hop_length)
    frames = np.stack(
        [
            audio[index * hop_length : index * hop_length + frame_length]
            for index in range(frame_count)
        ]
    )
    windowed = frames * np.hamming(frame_length)
    spectrum = np.abs(np.fft.rfft(windowed, axis=1))
    log_spectrum = np.log1p(spectrum)
    mean_features = log_spectrum.mean(axis=0)
    std_features = log_spectrum.std(axis=0)
    embedding = np.concatenate([mean_features, std_features]).astype(np.float32)
    norm = np.linalg.norm(embedding)
    if norm == 0.0:
        return embedding
    return embedding / norm


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


@dataclass(slots=True)
class EnrollmentSpeakerGate:
    enrollment_embedding: np.ndarray
    threshold: float

    @classmethod
    def from_paths(
        cls,
        paths: list[Path],
        *,
        threshold: float,
        sample_rate: int = 16000,
    ) -> "EnrollmentSpeakerGate":
        if not paths:
            raise ValueError("at least one enrollment path is required")
        embeddings = [_speaker_embedding(_load_mono_audio(path, sample_rate)) for path in paths]
        profile = np.mean(np.stack(embeddings), axis=0)
        norm = np.linalg.norm(profile)
        if norm != 0.0:
            profile = profile / norm
        return cls(enrollment_embedding=profile.astype(np.float32), threshold=threshold)

    def score_wav(self, wav_path: Path, *, sample_rate: int = 16000) -> float:
        embedding = _speaker_embedding(_load_mono_audio(wav_path, sample_rate))
        return _cosine_similarity(self.enrollment_embedding, embedding)

    def matches_wav(self, wav_path: Path, *, sample_rate: int = 16000) -> tuple[bool, float]:
        score = self.score_wav(wav_path, sample_rate=sample_rate)
        return score >= self.threshold, score
