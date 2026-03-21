from whisper_poc.config import AppConfig


def test_default_config_exposes_expected_paths():
    cfg = AppConfig.default()

    assert cfg.sample_rate == 16000
    assert cfg.channels == 1
    assert cfg.tts_enabled is True
