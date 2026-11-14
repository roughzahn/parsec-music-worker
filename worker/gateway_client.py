from urllib.parse import quote

import requests


class GatewayError(RuntimeError):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


class GatewayClient:
    """Pull-based client for the private analysis-worker API."""

    def __init__(self, base_url: str, token: str, timeout: float = 60):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "User-Agent": "parsec-music-worker/0.1",
        })

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _json(self, method: str, path: str, **kwargs):
        response = self.session.request(method, self._url(path), timeout=self.timeout, **kwargs)
        if response.status_code >= 400:
            raise GatewayError(response.status_code,
                               f"Gateway returned HTTP {response.status_code}: {response.text[:500]}")
        if response.status_code == 204 or not response.content:
            return None
        return response.json()

    def register(self, worker_id: str, worker_type: str, model_id: str, max_concurrent_jobs: int,
                 worker_status: str = "READY"):
        return self._json("POST", "/internal/v1/analysis/workers/register", json={
            "workerId": worker_id,
            "workerType": worker_type.upper(),
            "capabilities": ["audio-embedding", "clap-embedding-512"],
            "modelId": model_id,
            "maxConcurrentJobs": max_concurrent_jobs,
            "workerStatus": worker_status.upper(),
        })

    def update_worker_status(self, worker_id: str, status: str):
        worker_path = quote(worker_id, safe="")
        return self._json("POST", f"/internal/v1/analysis/workers/{worker_path}/status",
                          json={"status": status.upper()})

    def unregister(self, worker_id: str):
        return self._json("DELETE", f"/internal/v1/analysis/workers/{worker_id}")

    def claim_job(self, worker_id: str):
        return self._json("POST", f"/internal/v1/analysis/workers/{worker_id}/jobs/claim")

    def heartbeat(self, worker_id: str, job_id: str, lease_token: str):
        return self._json("POST", f"/internal/v1/analysis/workers/{worker_id}/jobs/{job_id}/heartbeat",
                          json={"leaseToken": lease_token})

    def download_audio(self, audio_path: str, expected_size: int, destination: str):
        response = self.session.post(
            self._url("/internal/v1/audio/content"),
            json={"path": audio_path, "start": 0, "length": expected_size},
            headers={"Accept": "application/octet-stream"},
            timeout=(15, max(self.timeout, 180)),
            stream=True,
        )
        if response.status_code >= 400:
            raise GatewayError(response.status_code,
                               f"Audio download failed with HTTP {response.status_code}: {response.text[:500]}")
        received = 0
        with open(destination, "wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)
                    received += len(chunk)
        if received != expected_size:
            raise GatewayError(502, f"Truncated audio download: received {received} of {expected_size} bytes")

    def complete_job(self, worker_id: str, job_id: str, lease_token: str, result: dict):
        return self._json("POST", f"/internal/v1/analysis/workers/{worker_id}/jobs/{job_id}/complete",
                          json={"leaseToken": lease_token, "result": result})

    def fail_job(self, worker_id: str, job_id: str, lease_token: str, error: str, retryable: bool = True):
        return self._json("POST", f"/internal/v1/analysis/workers/{worker_id}/jobs/{job_id}/fail",
                          json={"leaseToken": lease_token, "retryable": retryable, "error": error[:1000]})
