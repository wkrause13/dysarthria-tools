from __future__ import annotations

import argparse
import csv
import time
import wave
from pathlib import Path

import webrtcvad

from whisper_poc.asr_whispercpp import WhisperCppRunner
from whisper_poc.audio_capture import frame_byte_count, microphone_queue
from whisper_poc.config import AppConfig
from whisper_poc.tts_local import MacSpeaker
from whisper_poc.vad_segmenter import VadSegmenter

METRICS_FIELDNAMES = [
    "utterance_index",
    "speech_duration_seconds",
    "endpoint_delay_seconds",
    "asr_runtime_seconds",
    "tts_runtime_seconds",
    "total_post_speech_seconds",
    "frame_count",
    "trailing_silence_frames",
    "transcript_characters",
    "transcript",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local whisper.cpp proof of concept")
    parser.add_argument("--whisper-binary", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--metrics-csv")
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--frame-ms", type=int, default=30)
    parser.add_argument("--silence-ms", type=int, default=400)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--vad-mode", type=int, choices=[0, 1, 2, 3], default=2)
    parser.add_argument("--language", default="en")
    parser.add_argument("--voice", default="Samantha")
    parser.add_argument("--temp-dir", default="tmp_audio")
    parser.add_argument("--no-tts", action="store_true")
    return parser


def write_pcm16_wav(path: Path, pcm_bytes: bytes, sample_rate: int, channels: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)


def build_metrics_row(
    *,
    utterance_index: int,
    frame_count: int,
    trailing_silence_frames: int,
    frame_ms: int,
    asr_runtime_seconds: float,
    tts_runtime_seconds: float,
    transcript: str,
) -> dict[str, int | float | str]:
    speech_duration_seconds = round(frame_count * frame_ms / 1000.0, 3)
    endpoint_delay_seconds = round(trailing_silence_frames * frame_ms / 1000.0, 3)
    asr_runtime_seconds = round(asr_runtime_seconds, 3)
    tts_runtime_seconds = round(tts_runtime_seconds, 3)
    total_post_speech_seconds = round(
        endpoint_delay_seconds + asr_runtime_seconds + tts_runtime_seconds,
        3,
    )
    return {
        "utterance_index": utterance_index,
        "speech_duration_seconds": speech_duration_seconds,
        "endpoint_delay_seconds": endpoint_delay_seconds,
        "asr_runtime_seconds": asr_runtime_seconds,
        "tts_runtime_seconds": tts_runtime_seconds,
        "total_post_speech_seconds": total_post_speech_seconds,
        "frame_count": frame_count,
        "trailing_silence_frames": trailing_silence_frames,
        "transcript_characters": len(transcript),
        "transcript": transcript,
    }


def append_metrics_row(path: Path, row: dict[str, int | float | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    should_write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRICS_FIELDNAMES)
        if should_write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = AppConfig(
        sample_rate=args.sample_rate,
        channels=args.channels,
        frame_ms=args.frame_ms,
        silence_ms=args.silence_ms,
        tts_enabled=not args.no_tts,
        tts_voice=args.voice,
        temp_dir=Path(args.temp_dir),
    )
    config.temp_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics_csv) if args.metrics_csv else None

    vad = webrtcvad.Vad(args.vad_mode)
    segmenter = VadSegmenter(
        sample_rate=config.sample_rate,
        frame_ms=config.frame_ms,
        silence_ms=config.silence_ms,
    )
    runner = WhisperCppRunner(
        binary_path=args.whisper_binary,
        model_path=args.model_path,
        language=args.language,
    )
    speaker = MacSpeaker(voice=config.tts_voice, enabled=config.tts_enabled)

    blocksize = config.sample_rate * config.frame_ms // 1000
    expected_bytes = frame_byte_count(config.sample_rate, config.frame_ms, config.channels)

    print("[poc] listening; press Ctrl-C to stop")
    utterance_index = 0

    try:
        with microphone_queue(
            sample_rate=config.sample_rate,
            channels=config.channels,
            blocksize=blocksize,
            device=args.device,
        ) as queue:
            while True:  # pragma: no branch - manual interactive loop
                frame = queue.get()
                if len(frame) != expected_bytes:
                    continue
                is_speech = vad.is_speech(frame, config.sample_rate)
                utterance = segmenter.push(frame, is_speech=is_speech)
                if utterance is None:
                    continue

                utterance_index += 1
                utterance_path = config.temp_dir / f"utterance_{utterance_index:04d}.wav"
                write_pcm16_wav(
                    utterance_path,
                    utterance.audio_bytes,
                    sample_rate=config.sample_rate,
                    channels=config.channels,
                )
                started = time.perf_counter()
                result = runner.transcribe(utterance_path)
                asr_wall_seconds = time.perf_counter() - started
                tts_runtime_seconds = speaker.speak(result.transcript)
                metrics_row = build_metrics_row(
                    utterance_index=utterance_index,
                    frame_count=utterance.frame_count,
                    trailing_silence_frames=utterance.trailing_silence_frames,
                    frame_ms=config.frame_ms,
                    asr_runtime_seconds=result.runtime_seconds or asr_wall_seconds,
                    tts_runtime_seconds=tts_runtime_seconds,
                    transcript=result.transcript,
                )
                if metrics_path is not None:
                    append_metrics_row(metrics_path, metrics_row)
                print(
                    f"[utterance {utterance_index}] "
                    f"frames={utterance.frame_count} "
                    f"endpoint={metrics_row['endpoint_delay_seconds']:.3f}s "
                    f"asr={metrics_row['asr_runtime_seconds']:.3f}s "
                    f"tts={metrics_row['tts_runtime_seconds']:.3f}s "
                    f"post_speech={metrics_row['total_post_speech_seconds']:.3f}s"
                )
                print(f"[utterance {utterance_index}] {result.transcript}")
    except KeyboardInterrupt:
        print("\n[poc] stopped")


if __name__ == "__main__":
    main()
