from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class AppConfig:
    sample_rate: int = 16000
    channels: int = 1
    frame_ms: int = 30
    silence_ms: int = 300
    tts_enabled: bool = True
    tts_voice: str = "Samantha"
    temp_dir: Path = Path("tmp_audio")

    @classmethod
    def default(cls) -> "AppConfig":
        return cls()
