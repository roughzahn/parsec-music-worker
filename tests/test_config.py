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


def test_gpu_profile_does_not_require_static_token(monkeypatch):
    monkeypatch.setenv("PARSEC_WORKER_TYPE", "gpu")
    monkeypatch.delenv("PARSEC_WORKER_TOKEN", raising=False)
    monkeypatch.delenv("PARSEC_AUDIO_TRANSPORT_TOKEN", raising=False)

    settings = Settings.from_env()

    assert settings.token == ""
    assert settings.heartbeat_seconds == 10


def test_cpu_profile_still_requires_static_token(monkeypatch):
    monkeypatch.setenv("PARSEC_WORKER_TYPE", "cpu")
    monkeypatch.delenv("PARSEC_WORKER_TOKEN", raising=False)
    monkeypatch.delenv("PARSEC_AUDIO_TRANSPORT_TOKEN", raising=False)

    with pytest.raises(ValueError, match="CPU workers require"):
        Settings.from_env()
