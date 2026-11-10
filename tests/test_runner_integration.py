import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

from worker.config import Settings
from worker.gateway_client import GatewayClient
from worker.runner import Worker


AUDIO_BYTES = b"fake-audio-content"


class FakeEngine:
    def __init__(self, fail=False):
        self.fail = fail
        self.loaded = False
        self.analyzed_bytes = None

    def load(self):
        self.loaded = True

    def analyze(self, audio_path):
        with open(audio_path, "rb") as source:
            self.analyzed_bytes = source.read()
        if self.fail:
            raise RuntimeError("synthetic model failure")
        return {
            "modelId": "clap-test-model",
            "embeddingVersion": 1,
            "dimensions": 512,
            "embedding": [0.125] * 512,
            "processingSeconds": 0.01,
            "runtime": "cpu",
        }


def run_worker_integration(fail=False):
    state = {
        "registered": None,
        "completion": None,
        "failure": None,
        "claim_count": 0,
    }
    finished = threading.Event()
    job = {
        "jobId": "job-test-1",
        "leaseToken": "lease-test-1",
        "leaseExpiresAt": "2026-11-10T20:00:00+01:00",
        "attempt": 1,
        "audio": {
            "path": "Artist/Album/track.wav",
            "sizeBytes": len(AUDIO_BYTES),
            "lastModifiedUtc": "2026-11-10T18:00:00Z",
            "contentHash": "sha256-test",
        },
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def send_json(self, status, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_empty(self, status=204):
            self.send_response(status)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_POST(self):
            if self.headers.get("Authorization") != "Bearer integration-secret":
                self.send_empty(401)
                return

            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else b""

            if self.path == "/internal/v1/analysis/workers/register":
                state["registered"] = json.loads(body)
                self.send_empty()
            elif self.path == "/internal/v1/analysis/workers/cpu-integration/jobs/claim":
                state["claim_count"] += 1
                if state["claim_count"] == 1:
                    self.send_json(200, job)
                else:
                    self.send_empty()
            elif self.path == "/internal/v1/audio/content":
                request = json.loads(body)
                assert request["path"] == job["audio"]["path"]
                assert request["length"] == len(AUDIO_BYTES)
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(AUDIO_BYTES)))
                self.end_headers()
                self.wfile.write(AUDIO_BYTES)
            elif self.path.endswith("/complete"):
                state["completion"] = json.loads(body)
                finished.set()
                self.send_empty()
            elif self.path.endswith("/fail"):
                state["failure"] = json.loads(body)
                finished.set()
                self.send_empty()
            elif self.path.endswith("/heartbeat"):
                self.send_json(200, {"leaseValid": True})
            else:
                self.send_empty(404)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server_thread = Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    settings = Settings(
        gateway_url=f"http://127.0.0.1:{server.server_port}",
        token="integration-secret",
        worker_id="cpu-integration",
        worker_type="cpu",
        poll_seconds=0.02,
        heartbeat_seconds=10,
        request_timeout=2,
        model_path="/unused/in/test",
        model_id="clap-test-model",
        max_concurrent_jobs=1,
    )
    engine = FakeEngine(fail=fail)
    worker = Worker(settings, GatewayClient(settings.gateway_url, settings.token, settings.request_timeout), engine)
    worker_thread = Thread(target=worker.run, daemon=True)
    worker_thread.start()

    try:
        assert finished.wait(timeout=5), "worker did not report job completion/failure"
    finally:
        worker.stop()
        worker_thread.join(timeout=2)
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)

    return state, engine


def test_worker_downloads_audio_analyzes_and_completes_job_over_http():
    state, engine = run_worker_integration()

    assert engine.loaded is True
    assert engine.analyzed_bytes == AUDIO_BYTES
    assert state["registered"]["workerId"] == "cpu-integration"
    assert state["registered"]["workerType"] == "CPU"
    assert state["completion"]["leaseToken"] == "lease-test-1"
    assert state["completion"]["result"]["dimensions"] == 512
    assert len(state["completion"]["result"]["embedding"]) == 512
    assert state["completion"]["result"]["audio"] == {
        "path": "Artist/Album/track.wav",
        "sizeBytes": len(AUDIO_BYTES),
        "lastModifiedUtc": "2026-11-10T18:00:00Z",
        "contentHash": "sha256-test",
    }
    assert state["failure"] is None


def test_worker_reports_analysis_failure_to_gateway():
    state, engine = run_worker_integration(fail=True)

    assert engine.analyzed_bytes == AUDIO_BYTES
    assert state["completion"] is None
    assert state["failure"]["leaseToken"] == "lease-test-1"
    assert state["failure"]["retryable"] is True
    assert "synthetic model failure" in state["failure"]["error"]
