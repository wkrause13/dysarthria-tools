import subprocess
import time


class MacSpeaker:
    def __init__(self, voice: str = "Samantha", enabled: bool = True) -> None:
        self.voice = voice
        self.enabled = enabled

    def build_command(self, text: str) -> list[str]:
        return ["say", "-v", self.voice, text]

    def speak(self, text: str) -> float:
        if not self.enabled or not text:
            return 0.0
        start = time.perf_counter()
        subprocess.run(self.build_command(text), check=True)
        return time.perf_counter() - start
