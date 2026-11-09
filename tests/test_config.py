from worker.config import Settings


def test_settings_defaults(monkeypatch):
    monkeypatch.setenv("PARSEC_WORKER_TOKEN", "secret")
    monkeypatch.delenv("PARSEC_AUDIO_TRANSPORT_TOKEN", raising=False)
    settings = Settings.from_env()
    assert settings.worker_type == "gpu"
    assert settings.worker_id
    assert settings.gateway_url == "http://127.0.0.1:8766"
    assert settings.max_concurrent_jobs == 1


def test_cpu_profile(monkeypatch):
    monkeypatch.setenv("PARSEC_WORKER_TOKEN", "secret")
    monkeypatch.setenv("PARSEC_WORKER_TYPE", "cpu")
    settings = Settings.from_env()
    assert settings.worker_type == "cpu"
