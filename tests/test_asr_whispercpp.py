from pathlib import Path

from whisper_poc.asr_whispercpp import AsrResult, WhisperCppRunner


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

