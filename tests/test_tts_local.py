from whisper_poc.tts_local import MacSpeaker


def test_tts_command_uses_macos_say_binary():
    speaker = MacSpeaker(voice="Samantha")

    assert speaker.build_command("hello") == ["say", "-v", "Samantha", "hello"]


def test_speak_returns_runtime_when_enabled(monkeypatch):
    calls = []

    def fake_run(cmd, check):
        calls.append(cmd)
        return None

    monkeypatch.setattr("whisper_poc.tts_local.subprocess.run", fake_run)
    speaker = MacSpeaker(voice="Samantha", enabled=True)

    runtime = speaker.speak("hello")

    assert calls == [["say", "-v", "Samantha", "hello"]]
    assert runtime >= 0.0
