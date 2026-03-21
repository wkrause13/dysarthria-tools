import wave

import pytest

from whisper_poc.audio_capture import frame_byte_count, iter_wav_frames, pcm16_rms


def test_iter_wav_frames_yields_fixed_size_frames_and_trailing_silence(tmp_path):
    wav_path = tmp_path / "sample.wav"
    frame_bytes = frame_byte_count(sample_rate=16000, frame_ms=30, channels=1)
    pcm = (b"\x01\x00" * 480) + (b"\x02\x00" * 480)

    with wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(pcm)

    frames = list(
        iter_wav_frames(
            wav_path,
            frame_bytes=frame_bytes,
            sample_rate=16000,
            channels=1,
            trailing_silence_frames=2,
        )
    )

    assert frames == [
        b"\x01\x00" * 480,
        b"\x02\x00" * 480,
        b"\x00" * frame_bytes,
        b"\x00" * frame_bytes,
    ]


def test_iter_wav_frames_rejects_mismatched_wav_format(tmp_path):
    wav_path = tmp_path / "bad.wav"

    with wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(22050)
        handle.writeframes(b"\x00\x00" * 100)

    with pytest.raises(ValueError):
        list(
            iter_wav_frames(
                wav_path,
                frame_bytes=frame_byte_count(sample_rate=16000, frame_ms=30, channels=1),
                sample_rate=16000,
                channels=1,
            )
        )


def test_pcm16_rms_reports_energy_for_pcm_samples():
    quiet = (b"\x01\x00" * 10)
    loud = (b"\x10\x00" * 10)

    assert pcm16_rms(b"") == 0.0
    assert pcm16_rms(loud) > pcm16_rms(quiet)
