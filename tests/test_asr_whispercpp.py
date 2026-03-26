from pathlib import Path

import pytest

from whisper_poc.asr_whispercpp import AsrResult, WhisperCppRunner, is_hallucination


def test_runner_builds_expected_whisper_cpp_command(tmp_path: Path):
    runner = WhisperCppRunner(binary_path="/tmp/main", model_path="/tmp/model.bin")

    cmd = runner.build_command(tmp_path / "sample.wav")

    assert cmd[:2] == ["/tmp/main", "-m"]
    assert cmd[2] == "/tmp/model.bin"
    assert str(tmp_path / "sample.wav") in cmd


def test_runner_builds_command_with_live_safety_flags(tmp_path: Path):
    runner = WhisperCppRunner(
        binary_path="/tmp/main",
        model_path="/tmp/model.bin",
        no_speech_threshold=0.85,
        suppress_non_speech=True,
        no_fallback=True,
    )

    cmd = runner.build_command(tmp_path / "sample.wav")

    assert "-nth" in cmd
    assert cmd[cmd.index("-nth") + 1] == "0.85"
    assert "-sns" in cmd
    assert "-nf" in cmd


@pytest.mark.parametrize(
    "text",
    [
        "Thank you for watching.",
        "Thanks for watching!",
        "Thank you for watching",
        "THANK YOU FOR WATCHING",
        "Please subscribe.",
        "Subscribe",
        "Like and subscribe.",
        "See you next time.",
        "See you in the next video",
        "you",
        " You. ",
        "Subtitles by the Amara.org community",
    ],
)
def test_is_hallucination_catches_known_phrases(text: str):
    assert is_hallucination(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "thank you",
        "Thank you!",
        "Thank you very much",
        "Hello, how are you?",
        "I want to subscribe to the newsletter",
        "Can you help me please",
        "Thanks for your help",
        "You are welcome",
    ],
)
def test_is_hallucination_allows_real_speech(text: str):
    assert is_hallucination(text) is False

