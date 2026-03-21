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
    def __init__(self, binary_path: str, model_path: str, language: str = "en") -> None:
        self.binary_path = binary_path
        self.model_path = model_path
        self.language = language

    def build_command(self, audio_path: Path) -> list[str]:
        return [
            self.binary_path,
            "-m",
            self.model_path,
            "-f",
            str(audio_path),
            "-l",
            self.language,
            "-nt",
        ]

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
