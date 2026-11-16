import pytest

from worker.config import Settings


def test_gpu_settings_default_to_twenty_prefetched_and_four_inference_jobs(monkeypatch):
    monkeypatch.delenv("PARSEC_WORKER_TYPE", raising=False)
    monkeypatch.delenv("PARSEC_WORKER_MAX_CONCURRENT_JOBS", raising=False)
    monkeypatch.delenv("PARSEC_GPU_PREFETCH_BATCH_SIZE", raising=False)
    monkeypatch.delenv("PARSEC_GPU_INFERENCE_BATCH_SIZE", raising=False)
    monkeypatch.delenv("PARSEC_WORKER_TOKEN", raising=False)
    monkeypatch.delenv("PARSEC_AUDIO_TRANSPORT_TOKEN", raising=False)

    settings = Settings.from_env()

    assert settings.worker_type == "gpu"
    assert settings.token == ""
    assert settings.prefetch_batch_size == 20
    assert settings.inference_batch_size == 4
    assert settings.max_concurrent_jobs == 20
    assert settings.heartbeat_seconds == 10


def test_cpu_profile_defaults_to_ten_job_batches_and_five_minute_cooldown(monkeypatch):
    monkeypatch.setenv("PARSEC_WORKER_TOKEN", "secret")
    monkeypatch.setenv("PARSEC_WORKER_TYPE", "cpu")
    monkeypatch.delenv("PARSEC_GPU_PREFETCH_BATCH_SIZE", raising=False)
    monkeypatch.delenv("PARSEC_GPU_INFERENCE_BATCH_SIZE", raising=False)
    settings = Settings.from_env()
    assert settings.worker_type == "cpu"
    assert settings.prefetch_batch_size == 10
    assert settings.inference_batch_size == 1
    assert settings.max_concurrent_jobs == 10
    assert settings.cooldown_seconds == 300


def test_rejects_gpu_concurrency_not_matching_prefetch(monkeypatch):
    monkeypatch.setenv("PARSEC_WORKER_TYPE", "gpu")
    monkeypatch.setenv("PARSEC_WORKER_MAX_CONCURRENT_JOBS", "2")
    with pytest.raises(ValueError, match="equal the prefetch batch size"):
        Settings.from_env()


def test_rejects_invalid_inference_batch_size(monkeypatch):
    monkeypatch.setenv("PARSEC_GPU_PREFETCH_BATCH_SIZE", "3")
    monkeypatch.setenv("PARSEC_GPU_INFERENCE_BATCH_SIZE", "4")
    with pytest.raises(ValueError, match="between 1 and the prefetch batch size"):
        Settings.from_env()


def test_cpu_profile_still_requires_static_token(monkeypatch):
    monkeypatch.setenv("PARSEC_WORKER_TYPE", "cpu")
    monkeypatch.delenv("PARSEC_WORKER_TOKEN", raising=False)
    monkeypatch.delenv("PARSEC_AUDIO_TRANSPORT_TOKEN", raising=False)
    with pytest.raises(ValueError, match="CPU workers require"):
        Settings.from_env()


def test_gpu_profile_ignores_legacy_static_token(monkeypatch):
    monkeypatch.setenv("PARSEC_WORKER_TYPE", "gpu")
    monkeypatch.setenv("PARSEC_WORKER_TOKEN", "old-shared-secret")
    settings = Settings.from_env()
    assert settings.token == ""


def test_cpu_profile_ignores_gpu_batch_tuning(monkeypatch):
    monkeypatch.setenv("PARSEC_WORKER_TYPE", "cpu")
    monkeypatch.setenv("PARSEC_WORKER_TOKEN", "secret")
    monkeypatch.setenv("PARSEC_GPU_PREFETCH_BATCH_SIZE", "0")
    monkeypatch.setenv("PARSEC_GPU_INFERENCE_BATCH_SIZE", "99")

    settings = Settings.from_env()

    assert settings.prefetch_batch_size == 10
    assert settings.inference_batch_size == 1
    assert settings.max_concurrent_jobs == 10
    assert settings.cooldown_seconds == 300

def test_cpu_batch_and_cooldown_can_be_configured(monkeypatch):
    monkeypatch.setenv("PARSEC_WORKER_TYPE", "cpu")
    monkeypatch.setenv("PARSEC_WORKER_TOKEN", "secret")
    monkeypatch.setenv("PARSEC_CPU_BATCH_SIZE", "7")
    monkeypatch.setenv("PARSEC_CPU_COOLDOWN_SECONDS", "120")
    settings = Settings.from_env()
    assert settings.prefetch_batch_size == 7
    assert settings.max_concurrent_jobs == 7
    assert settings.inference_batch_size == 1
    assert settings.cooldown_seconds == 120


def test_rejects_cpu_capacity_that_differs_from_batch_size(monkeypatch):
    monkeypatch.setenv("PARSEC_WORKER_TYPE", "cpu")
    monkeypatch.setenv("PARSEC_WORKER_TOKEN", "secret")
    monkeypatch.setenv("PARSEC_CPU_BATCH_SIZE", "10")
    monkeypatch.setenv("PARSEC_WORKER_MAX_CONCURRENT_JOBS", "1")
    with pytest.raises(ValueError, match="equal the CPU batch size"):
        Settings.from_env()
