import csv

import pytest

from whisper_poc.app import (
    append_metrics_row,
    build_metrics_row,
    build_parser,
    should_process_utterance,
    wait_for_speaker_idle,
    write_pcm16_wav,
)


def test_cli_requires_whisper_binary_and_model_paths():
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_cli_accepts_metrics_csv_path(tmp_path):
    parser = build_parser()

    args = parser.parse_args(
        [
            "--whisper-binary",
            "/tmp/whisper-cli",
            "--model-path",
            "/tmp/model.bin",
            "--metrics-csv",
            str(tmp_path / "metrics.csv"),
        ]
    )

    assert args.metrics_csv == str(tmp_path / "metrics.csv")


def test_cli_accepts_input_wav_path(tmp_path):
    parser = build_parser()

    args = parser.parse_args(
        [
            "--whisper-binary",
            "/tmp/whisper-cli",
            "--model-path",
            "/tmp/model.bin",
            "--input-wav",
            str(tmp_path / "sample.wav"),
        ]
    )

    assert args.input_wav == str(tmp_path / "sample.wav")


def test_cli_accepts_enrollment_flags(tmp_path):
    parser = build_parser()
    enroll_dir = tmp_path / "enroll"
    enroll_dir.mkdir()

    args = parser.parse_args(
        [
            "--whisper-binary",
            "/tmp/whisper-cli",
            "--model-path",
            "/tmp/model.bin",
            "--enrollment-wav",
            str(tmp_path / "a.wav"),
            "--enrollment-wav",
            str(tmp_path / "b.wav"),
            "--enrollment-dir",
            str(enroll_dir),
            "--speaker-threshold",
            "0.82",
        ]
    )

    assert args.enrollment_wav == [str(tmp_path / "a.wav"), str(tmp_path / "b.wav")]
    assert args.enrollment_dir == str(enroll_dir)
    assert args.speaker_threshold == 0.82


def test_cli_accepts_min_speech_ms_override():
    parser = build_parser()

    args = parser.parse_args(
        [
            "--whisper-binary",
            "/tmp/whisper-cli",
            "--model-path",
            "/tmp/model.bin",
            "--min-speech-ms",
            "500",
        ]
    )

    assert args.min_speech_ms == 500


def test_cli_accepts_max_utterance_ms_override():
    parser = build_parser()

    args = parser.parse_args(
        [
            "--whisper-binary",
            "/tmp/whisper-cli",
            "--model-path",
            "/tmp/model.bin",
            "--max-utterance-ms",
            "5000",
        ]
    )

    assert args.max_utterance_ms == 5000


def test_cli_accepts_min_rms_override():
    parser = build_parser()

    args = parser.parse_args(
        [
            "--whisper-binary",
            "/tmp/whisper-cli",
            "--model-path",
            "/tmp/model.bin",
            "--min-rms",
            "350",
        ]
    )

    assert args.min_rms == 350.0


def test_cli_rejects_streaming_flags():
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--whisper-binary",
                "/tmp/whisper-cli",
                "--model-path",
                "/tmp/model.bin",
                "--streaming",
            ]
        )

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--whisper-binary",
                "/tmp/whisper-cli",
                "--model-path",
                "/tmp/model.bin",
                "--decode-interval-ms",
                "300",
            ]
        )

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--whisper-binary",
                "/tmp/whisper-cli",
                "--model-path",
                "/tmp/model.bin",
                "--no-speech-thold",
                "0.85",
            ]
        )


def test_cli_rejects_removed_tts_interrupt_ms_flag():
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--whisper-binary",
                "/tmp/whisper-cli",
                "--model-path",
                "/tmp/model.bin",
                "--tts-interrupt-ms",
                "180",
            ]
        )


def test_write_pcm16_wav_creates_file(tmp_path):
    pcm = (b"\x00\x00" * 1600)
    wav_path = tmp_path / "sample.wav"

    write_pcm16_wav(wav_path, pcm, sample_rate=16000, channels=1)

    assert wav_path.exists()
    assert wav_path.stat().st_size > 44


def test_append_metrics_row_writes_header_once(tmp_path):
    metrics_path = tmp_path / "metrics.csv"
    first_row = {
        "utterance_index": 1,
        "speech_duration_seconds": 0.9,
        "endpoint_delay_seconds": 0.3,
        "asr_runtime_seconds": 0.4,
        "tts_runtime_seconds": 0.0,
        "total_post_speech_seconds": 0.7,
        "frame_count": 30,
        "trailing_silence_frames": 10,
        "transcript_characters": 5,
        "transcript": "hello",
    }
    second_row = {
        "utterance_index": 2,
        "speech_duration_seconds": 1.2,
        "endpoint_delay_seconds": 0.3,
        "asr_runtime_seconds": 0.5,
        "tts_runtime_seconds": 0.0,
        "total_post_speech_seconds": 0.8,
        "frame_count": 40,
        "trailing_silence_frames": 10,
        "transcript_characters": 7,
        "transcript": "goodbye",
    }

    append_metrics_row(metrics_path, first_row)
    append_metrics_row(metrics_path, second_row)

    lines = metrics_path.read_text().splitlines()
    assert lines[0].startswith("utterance_index,")
    assert len(lines) == 3

    with metrics_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["transcript"] == "hello"
    assert rows[1]["transcript"] == "goodbye"


def test_build_metrics_row_computes_expected_timings():
    row = build_metrics_row(
        utterance_index=3,
        frame_count=30,
        trailing_silence_frames=10,
        frame_ms=30,
        asr_runtime_seconds=0.4,
        tts_runtime_seconds=0.1,
        transcript="hello world",
    )

    assert row["utterance_index"] == 3
    assert row["speech_duration_seconds"] == 0.9
    assert row["endpoint_delay_seconds"] == 0.3
    assert row["asr_runtime_seconds"] == 0.4
    assert row["tts_runtime_seconds"] == 0.1
    assert row["total_post_speech_seconds"] == 0.8
    assert row["transcript_characters"] == 11


def test_should_process_utterance_filters_short_segments():
    assert should_process_utterance(frame_count=3, frame_ms=30, min_speech_ms=300) is False
    assert should_process_utterance(frame_count=17, frame_ms=30, min_speech_ms=300) is True


class FakeQueuedSpeaker:
    def __init__(self):
        self.pending = True
        self.process_calls = 0

    def process_queue(self):
        self.process_calls += 1
        self.pending = False
        return 0.0

    def is_speaking(self):
        return False

    def has_pending(self):
        return self.pending


def test_wait_for_speaker_idle_drains_pending_queue():
    speaker = FakeQueuedSpeaker()

    wait_for_speaker_idle(speaker, poll_interval_seconds=0.0, timeout_seconds=0.01)

    assert speaker.process_calls >= 1
    assert speaker.has_pending() is False
