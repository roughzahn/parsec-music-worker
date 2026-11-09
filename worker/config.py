from dataclasses import dataclass
import os
import socket


@dataclass(frozen=True)
class Settings:
    gateway_url: str
    token: str
    worker_id: str
    worker_type: str
    poll_seconds: float
    heartbeat_seconds: float
    request_timeout: float
    model_path: str
    model_id: str
    max_concurrent_jobs: int

    @classmethod
    def from_env(cls):
        worker_type = os.getenv("PARSEC_WORKER_TYPE", "gpu").strip().lower()
        if worker_type not in {"cpu", "gpu"}:
            raise ValueError("PARSEC_WORKER_TYPE must be cpu or gpu")
        token = os.getenv("PARSEC_WORKER_TOKEN") or os.getenv("PARSEC_AUDIO_TRANSPORT_TOKEN", "")
        if not token.strip():
            raise ValueError("PARSEC_WORKER_TOKEN (or PARSEC_AUDIO_TRANSPORT_TOKEN) is required")
        return cls(
            gateway_url=os.getenv("PARSEC_GATEWAY_BASE_URL", "http://127.0.0.1:8766").rstrip("/"),
            token=token,
            worker_id=os.getenv("PARSEC_WORKER_ID", socket.gethostname()).strip(),
            worker_type=worker_type,
            poll_seconds=float(os.getenv("PARSEC_WORKER_POLL_SECONDS", "3")),
            heartbeat_seconds=float(os.getenv("PARSEC_WORKER_HEARTBEAT_SECONDS", "20")),
            request_timeout=float(os.getenv("PARSEC_WORKER_REQUEST_TIMEOUT", "60")),
            model_path=os.getenv("PARSEC_CLAP_MODEL_PATH", "/models/630k-audioset-best.pt"),
            model_id=os.getenv("PARSEC_CLAP_MODEL_ID", "laion-clap-630k-audioset-v1"),
            max_concurrent_jobs=int(os.getenv("PARSEC_WORKER_MAX_CONCURRENT_JOBS", "1")),
        )
