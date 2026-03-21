from dataclasses import dataclass


@dataclass(slots=True)
class CompletedUtterance:
    audio_bytes: bytes
    frame_count: int
    trailing_silence_frames: int


class VadSegmenter:
    def __init__(
        self,
        sample_rate: int,
        frame_ms: int,
        silence_ms: int,
        max_utterance_ms: int = 0,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.silence_frames = max(1, silence_ms // frame_ms)
        self.max_utterance_frames = (
            max_utterance_ms // frame_ms if max_utterance_ms > 0 else 0
        )
        self._buffer = bytearray()
        self._speech_frames = 0
        self._trailing_silence_frames = 0
        self._active = False

    def push(self, frame_bytes: bytes, is_speech: bool) -> CompletedUtterance | None:
        if is_speech:
            self._active = True
            self._trailing_silence_frames = 0
            self._speech_frames += 1
            self._buffer.extend(frame_bytes)
            if self.max_utterance_frames and self._speech_frames >= self.max_utterance_frames:
                utterance = CompletedUtterance(
                    bytes(self._buffer),
                    self._speech_frames,
                    0,
                )
                self._buffer.clear()
                self._speech_frames = 0
                self._trailing_silence_frames = 0
                self._active = False
                return utterance
            return None

        if not self._active:
            return None

        self._trailing_silence_frames += 1
        if self._trailing_silence_frames < self.silence_frames:
            return None

        utterance = CompletedUtterance(
            bytes(self._buffer),
            self._speech_frames,
            self._trailing_silence_frames,
        )
        self._buffer.clear()
        self._speech_frames = 0
        self._trailing_silence_frames = 0
        self._active = False
        return utterance
