import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class AsrResult:
    transcript: str
    stdout: str
    runtime_seconds: float


class WhisperCppRunner:
    def __init__(
        self,
        binary_path: str,
        model_path: str,
        language: str = "en",
        no_speech_threshold: float = 0.85,
        suppress_non_speech: bool = True,
        no_fallback: bool = True,
    ) -> None:
        self.binary_path = binary_path
        self.model_path = model_path
        self.language = language
        self.no_speech_threshold = no_speech_threshold
        self.suppress_non_speech = suppress_non_speech
        self.no_fallback = no_fallback

    def build_command(self, audio_path: Path) -> list[str]:
        command = [
            self.binary_path,
            "-m",
            self.model_path,
            "-f",
            str(audio_path),
            "-l",
            self.language,
            "-nt",
            "-nth",
            f"{self.no_speech_threshold:.2f}",
        ]
        if self.suppress_non_speech:
            command.append("-sns")
        if self.no_fallback:
            command.append("-nf")
        return command

    def transcribe(self, audio_path: Path) -> AsrResult:
        start = time.perf_counter()
        completed = subprocess.run(
            self.build_command(audio_path),
            check=True,
            capture_output=True,
            text=True,
        )
        runtime_seconds = time.perf_counter() - start
        transcript = ""
        for line in reversed(completed.stdout.splitlines()):
            candidate = line.strip()
            if not candidate or candidate.startswith("[") or candidate.startswith("system_info:"):
                continue
            transcript = candidate
            break
        return AsrResult(transcript=transcript, stdout=completed.stdout, runtime_seconds=runtime_seconds)
