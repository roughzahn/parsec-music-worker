from dataclasses import dataclass
import os
import socket


@dataclass(frozen=True)
class Settings:
    gateway_url: str
    token: str | None
    worker_id: str
    worker_type: str
    poll_seconds: float
    heartbeat_seconds: float
    request_timeout: float
    model_path: str
    model_id: str
    max_concurrent_jobs: int
    software_version: str = "dev"
    health_port: int = 8081
    prefetch_batch_size: int = 1
    inference_batch_size: int = 1
    cooldown_seconds: float = 300.0
    download_concurrency: int = 1

    @classmethod
    def from_env(cls):
        worker_type = os.getenv("PARSEC_WORKER_TYPE", "gpu").strip().lower()
        if worker_type not in {"cpu", "gpu"}:
            raise ValueError("PARSEC_WORKER_TYPE must be cpu or gpu")
        token = os.getenv("PARSEC_WORKER_TOKEN") or os.getenv("PARSEC_AUDIO_TRANSPORT_TOKEN", "")
        if worker_type == "gpu":
            token = ""  # GPU authorization is acquired through one-time Admin pairing.
        if worker_type == "cpu" and not token.strip():
            raise ValueError("CPU workers require PARSEC_WORKER_TOKEN (or PARSEC_AUDIO_TRANSPORT_TOKEN)")
        worker_id = os.getenv("PARSEC_WORKER_ID", socket.gethostname()).strip()
        if not worker_id:
            raise ValueError("PARSEC_WORKER_ID must not be empty")

        if worker_type == "gpu":
            prefetch_batch_size = int(os.getenv("PARSEC_GPU_PREFETCH_BATCH_SIZE", "20"))
            if not 1 <= prefetch_batch_size <= 50:
                raise ValueError("PARSEC_GPU_PREFETCH_BATCH_SIZE must be between 1 and 50")
            inference_batch_size = 1
            concurrency_default = prefetch_batch_size
        else:
            prefetch_batch_size = int(os.getenv("PARSEC_CPU_BATCH_SIZE", "10"))
            if not 1 <= prefetch_batch_size <= 50:
                raise ValueError("PARSEC_CPU_BATCH_SIZE must be between 1 and 50")
            # CPU inference remains one file at a time to limit transient RAM usage.
            inference_batch_size = 1
            concurrency_default = prefetch_batch_size

        max_concurrent_jobs = int(os.getenv("PARSEC_WORKER_MAX_CONCURRENT_JOBS", str(concurrency_default)))
        if worker_type == "cpu" and max_concurrent_jobs != prefetch_batch_size:
            raise ValueError("CPU max concurrent jobs must equal the CPU batch size")
        if worker_type == "gpu" and max_concurrent_jobs != prefetch_batch_size:
            raise ValueError("GPU max concurrent jobs must equal the prefetch batch size")

        download_concurrency = int(os.getenv("PARSEC_WORKER_DOWNLOAD_CONCURRENCY", str(prefetch_batch_size)))
        if not 1 <= download_concurrency <= prefetch_batch_size:
            raise ValueError("PARSEC_WORKER_DOWNLOAD_CONCURRENCY must be between 1 and the prefetch batch size")

        return cls(
            gateway_url=os.getenv("PARSEC_GATEWAY_BASE_URL", "http://127.0.0.1:8766").rstrip("/"),
            token=token,
            worker_id=worker_id,
            worker_type=worker_type,
            poll_seconds=float(os.getenv("PARSEC_WORKER_POLL_SECONDS", "3")),
            heartbeat_seconds=float(os.getenv("PARSEC_WORKER_HEARTBEAT_SECONDS", "10" if worker_type == "gpu" else "5")),
            request_timeout=float(os.getenv("PARSEC_WORKER_REQUEST_TIMEOUT", "60")),
            model_path=os.getenv("PARSEC_CLAP_MODEL_PATH", "/models/630k-audioset-best.pt"),
            model_id=os.getenv("PARSEC_CLAP_MODEL_ID", "laion-clap-630k-audioset-v1"),
            max_concurrent_jobs=max_concurrent_jobs,
            software_version=os.getenv("PARSEC_SOFTWARE_VERSION", "dev").strip() or "dev",
            health_port=int(os.getenv("PARSEC_WORKER_HEALTH_PORT", "8081")),
            prefetch_batch_size=prefetch_batch_size,
            inference_batch_size=inference_batch_size,
            cooldown_seconds=max(0.0, float(os.getenv("PARSEC_CPU_COOLDOWN_SECONDS", "300"))) if worker_type == "cpu" else 0.0,
            download_concurrency=download_concurrency,
        )
