import pytest

from worker.config import Settings


def test_settings_defaults(monkeypatch):
    monkeypatch.setenv("PARSEC_WORKER_TOKEN", "secret")
    monkeypatch.delenv("PARSEC_AUDIO_TRANSPORT_TOKEN", raising=False)
    monkeypatch.delenv("PARSEC_WORKER_TYPE", raising=False)
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


def test_rejects_unsupported_concurrency(monkeypatch):
    monkeypatch.setenv("PARSEC_WORKER_TOKEN", "secret")
    monkeypatch.setenv("PARSEC_WORKER_MAX_CONCURRENT_JOBS", "2")
    with pytest.raises(ValueError, match="one job at a time"):
        Settings.from_env()
