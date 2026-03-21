from pathlib import Path

from whisper_poc.asr_whispercpp import AsrResult
from whisper_poc.app import build_parser, run_endpointed_mode, write_pcm16_wav
from whisper_poc.config import AppConfig


class FakeVad:
    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        return any(frame)


class FakeVadModule:
    def Vad(self, mode: int) -> FakeVad:
        return FakeVad()


class FakeRunner:
    instances = []

    def __init__(self, **kwargs):
        self.calls = []
        FakeRunner.instances.append(self)

    def transcribe(self, audio_path: Path) -> AsrResult:
        self.calls.append(audio_path)
        return AsrResult(transcript="accepted", stdout="accepted", runtime_seconds=0.1)


class FakeSpeaker:
    instances = []

    def __init__(self, **kwargs):
        self.spoken = []
        FakeSpeaker.instances.append(self)

    def speak(self, text: str) -> float:
        self.spoken.append(text)
        return 0.01

    def process_queue(self) -> float:
        return 0.0

    def is_speaking(self) -> bool:
        return False

    def has_pending(self) -> bool:
        return False


class FakeGate:
    instances = []
    should_match = True

    def __init__(self):
        self.calls = []
        FakeGate.instances.append(self)

    @classmethod
    def from_paths(cls, paths, *, threshold, sample_rate=16000):
        return cls()

    def matches_wav(self, wav_path: Path, *, sample_rate: int = 16000):
        self.calls.append(wav_path)
        return self.should_match, 0.9 if self.should_match else 0.1


def _build_args(tmp_path: Path):
    wav_path = tmp_path / "sample.wav"
    write_pcm16_wav(
        wav_path,
        b"\x01\x00" * (480 * 40),
        sample_rate=16000,
        channels=1,
    )
    args = build_parser().parse_args(
        [
            "--whisper-binary",
            "/tmp/whisper-cli",
            "--model-path",
            "/tmp/model.bin",
            "--input-wav",
            str(wav_path),
            "--enrollment-wav",
            str(wav_path),
            "--speaker-threshold",
            "0.8",
            "--min-speech-ms",
            "300",
            "--min-rms",
            "0",
            "--no-tts",
        ]
    )
    config = AppConfig(
        sample_rate=16000,
        channels=1,
        frame_ms=30,
        silence_ms=300,
        tts_enabled=False,
        temp_dir=tmp_path / "tmp_audio",
    )
    return args, config


def test_endpointed_mode_skips_asr_when_speaker_gate_rejects(tmp_path, monkeypatch):
    args, config = _build_args(tmp_path)
    FakeGate.should_match = False
    FakeGate.instances.clear()
    FakeRunner.instances.clear()

    monkeypatch.setattr("whisper_poc.app.webrtcvad", FakeVadModule())
    monkeypatch.setattr("whisper_poc.app.WhisperCppRunner", FakeRunner)
    monkeypatch.setattr("whisper_poc.app.MacSpeaker", FakeSpeaker)
    monkeypatch.setattr("whisper_poc.app.EnrollmentSpeakerGate", FakeGate)
    monkeypatch.setattr("whisper_poc.app.resolve_enrollment_paths", lambda **kwargs: [Path(args.input_wav)])

    run_endpointed_mode(args, config)

    assert FakeGate.instances
    assert FakeGate.instances[0].calls
    assert FakeRunner.instances[0].calls == []


def test_endpointed_mode_runs_asr_when_speaker_gate_accepts(tmp_path, monkeypatch):
    args, config = _build_args(tmp_path)
    FakeGate.should_match = True
    FakeGate.instances.clear()
    FakeRunner.instances.clear()
    FakeSpeaker.instances.clear()

    monkeypatch.setattr("whisper_poc.app.webrtcvad", FakeVadModule())
    monkeypatch.setattr("whisper_poc.app.WhisperCppRunner", FakeRunner)
    monkeypatch.setattr("whisper_poc.app.MacSpeaker", FakeSpeaker)
    monkeypatch.setattr("whisper_poc.app.EnrollmentSpeakerGate", FakeGate)
    monkeypatch.setattr("whisper_poc.app.resolve_enrollment_paths", lambda **kwargs: [Path(args.input_wav)])

    run_endpointed_mode(args, config)

    assert FakeGate.instances
    assert FakeGate.instances[0].calls
    assert len(FakeRunner.instances[0].calls) == 1
