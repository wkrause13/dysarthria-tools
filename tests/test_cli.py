import csv

import pytest

from whisper_poc.app import (
    append_metrics_row,
    build_metrics_row,
    build_parser,
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
