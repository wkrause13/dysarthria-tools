from whisper_poc.tts_local import MacSpeaker


class FakeSynthesizer:
    def __init__(self):
        self.started = []
        self.stopped = False
        self.voice = None
        self.speaking = False

    def startSpeakingString_(self, text):
        self.started.append(text)
        self.speaking = True
        return True

    def stopSpeaking(self):
        self.stopped = True
        self.speaking = False

    def isSpeaking(self):
        return self.speaking

    def setVoice_(self, voice):
        self.voice = voice
        return True


def test_speaker_starts_native_speech_and_returns_launch_time():
    synth = FakeSynthesizer()
    speaker = MacSpeaker(voice="Samantha", enabled=True, synthesizer=synth)

    runtime = speaker.speak("hello")

    assert synth.voice == "Samantha"
    assert synth.started == ["hello"]
    assert runtime >= 0.0
    assert speaker.is_speaking() is True


def test_stop_interrupts_active_speech():
    synth = FakeSynthesizer()
    speaker = MacSpeaker(enabled=True, synthesizer=synth)

    speaker.speak("hello")
    speaker.stop()

    assert synth.stopped is True
    assert speaker.is_speaking() is False


def test_speak_returns_zero_when_disabled_or_empty():
    synth = FakeSynthesizer()
    disabled = MacSpeaker(enabled=False, synthesizer=synth)
    enabled = MacSpeaker(enabled=True, synthesizer=synth)

    assert disabled.speak("hello") == 0.0
    assert enabled.speak("") == 0.0


def test_followup_speech_is_queued_until_current_audio_finishes():
    synth = FakeSynthesizer()
    speaker = MacSpeaker(enabled=True, synthesizer=synth)

    speaker.speak("first")
    queued_runtime = speaker.speak("second")

    assert synth.started == ["first"]
    assert queued_runtime == 0.0
    assert speaker.has_pending() is True

    synth.speaking = False
    launch_runtime = speaker.process_queue()

    assert synth.started == ["first", "second"]
    assert launch_runtime >= 0.0
    assert speaker.has_pending() is False
