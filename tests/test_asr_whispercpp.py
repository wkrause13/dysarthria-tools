from pathlib import Path

from whisper_poc.asr_whispercpp import WhisperCppRunner


def test_runner_builds_expected_whisper_cpp_command(tmp_path: Path):
    runner = WhisperCppRunner(binary_path="/tmp/main", model_path="/tmp/model.bin")

    cmd = runner.build_command(tmp_path / "sample.wav")

    assert cmd[:2] == ["/tmp/main", "-m"]
    assert cmd[2] == "/tmp/model.bin"
    assert str(tmp_path / "sample.wav") in cmd
