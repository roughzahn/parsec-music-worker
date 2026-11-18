import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

from worker.config import Settings
from worker.gateway_client import GatewayClient, GatewayError
from worker.runner import Worker


AUDIO_BYTES = b"fake-audio-content"


class FakeEngine:
    def __init__(self, fail=False):
        self.fail = fail
        self.loaded = False
        self.analyzed_bytes = None

    def load(self):
        self.loaded = True

    def is_loaded(self):
        return self.loaded

    def unload(self):
        self.loaded = False

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
        "statuses": [],
        "unregistered": None,
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
            elif self.path == "/internal/v1/analysis/workers/cpu-integration/status":
                state["statuses"].append(json.loads(body)["status"])
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

        def do_DELETE(self):
            if self.headers.get("Authorization") != "Bearer integration-secret":
                self.send_empty(401)
                return
            state["unregistered"] = self.path
            self.send_empty()

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
        max_concurrent_jobs=10,
        prefetch_batch_size=10,
        inference_batch_size=1,
        cooldown_seconds=0,
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

    assert engine.loaded is False
    assert engine.analyzed_bytes == AUDIO_BYTES
    assert state["registered"]["workerId"] == "cpu-integration"
    assert state["registered"]["workerType"] == "CPU"
    assert state["registered"]["workerStatus"] == "STARTING"
    assert "READY" in state["statuses"]
    assert state["unregistered"] == "/internal/v1/analysis/workers/cpu-integration"
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
    assert state["unregistered"] == "/internal/v1/analysis/workers/cpu-integration"
    assert state["failure"]["leaseToken"] == "lease-test-1"
    assert state["failure"]["retryable"] is True
    assert "synthetic model failure" in state["failure"]["error"]


def test_worker_registers_as_starting_before_model_load_and_claims_only_when_ready():
    load_started = threading.Event()
    allow_load = threading.Event()
    registered = threading.Event()
    starting_heartbeat = threading.Event()
    state = {"registration": None, "statuses": [], "claims": 0, "unregistered": False}

    class RecordingClient:
        def register(self, worker_id, worker_type, model_id, capacity, worker_status, stop_event=None, worker_version="dev"):
            state["registration"] = worker_status
            registered.set()
            return True

        def update_worker_status(self, worker_id, status, **metadata):
            state["statuses"].append(status)
            if status == "STARTING":
                starting_heartbeat.set()

        def session_heartbeat(self, worker_id):
            pass

        def claim_job(self, worker_id):
            state["claims"] += 1
            return None

        def unregister(self, worker_id):
            state["unregistered"] = True

    class BlockingEngine(FakeEngine):
        def load(self):
            load_started.set()
            assert allow_load.wait(timeout=5)
            self.loaded = True

    settings = Settings(
        gateway_url="http://unused",
        token="test",
        worker_id="gpu-startup-test",
        worker_type="gpu",
        poll_seconds=0.02,
        heartbeat_seconds=0.05,
        request_timeout=2,
        model_path="/unused/in/test",
        model_id="clap-test-model",
        max_concurrent_jobs=1,
    )
    worker = Worker(settings, RecordingClient(), BlockingEngine())
    worker_thread = Thread(target=worker.run, daemon=True)
    worker_thread.start()

    try:
        assert registered.wait(timeout=2)
        assert load_started.wait(timeout=2)
        assert state["registration"] == "STARTING"
        assert starting_heartbeat.wait(timeout=1)
        assert not worker.ready.is_set()
        assert state["claims"] == 0

        allow_load.set()
        assert worker.ready.wait(timeout=3)
        assert "READY" in state["statuses"]
    finally:
        allow_load.set()
        worker.stop()
        worker_thread.join(timeout=3)

    assert not worker_thread.is_alive()
    assert state["unregistered"] is True


def test_worker_stopped_during_model_load_never_transitions_to_ready():
    load_started = threading.Event()
    allow_load = threading.Event()
    state = {"statuses": [], "unregistered": False}

    class RecordingClient:
        def register(self, worker_id, worker_type, model_id, capacity, worker_status, stop_event=None, worker_version="dev"):
            return True

        def update_worker_status(self, worker_id, status, **metadata):
            state["statuses"].append(status)

        def session_heartbeat(self, worker_id):
            pass

        def unregister(self, worker_id):
            state["unregistered"] = True

    class BlockingEngine(FakeEngine):
        def load(self):
            load_started.set()
            assert allow_load.wait(timeout=5)
            self.loaded = True

    settings = Settings(
        gateway_url="http://unused",
        token="test",
        worker_id="gpu-stopped-during-load",
        worker_type="gpu",
        poll_seconds=0.02,
        heartbeat_seconds=0.05,
        request_timeout=2,
        model_path="/unused/in/test",
        model_id="clap-test-model",
        max_concurrent_jobs=1,
    )
    worker = Worker(settings, RecordingClient(), BlockingEngine())
    worker_thread = Thread(target=worker.run, daemon=True)
    worker_thread.start()

    try:
        assert load_started.wait(timeout=2)
        worker.stop()
        allow_load.set()
        worker_thread.join(timeout=3)
    finally:
        allow_load.set()
        worker.stop()
        worker_thread.join(timeout=3)

    assert not worker_thread.is_alive()
    assert "READY" not in state["statuses"]
    assert state["unregistered"] is True



def test_cpu_worker_processes_a_reserved_batch_with_one_model_load_then_unloads():
    finished = threading.Event()
    paused = threading.Event()
    state = {"claims": 0, "loads": 0, "unloads": 0, "completed": [], "statuses": [], "status_updates": []}
    jobs = [
        {
            "jobId": "cpu-batch-job-" + str(index),
            "leaseToken": "lease-" + str(index),
            "audio": {"path": "Artist/Album/" + str(index) + ".wav", "sizeBytes": len(AUDIO_BYTES)},
        }
        for index in range(3)
    ]

    class BatchClient:
        def register(self, worker_id, worker_type, model_id, capacity, worker_status, stop_event=None, worker_version="dev"):
            state["capacity"] = capacity
            return True

        def update_worker_status(self, worker_id, status, **metadata):
            state["statuses"].append(status)
            state["status_updates"].append({"status": status, **metadata})
            if status == "PAUSED":
                paused.set()

        def claim_job(self, worker_id):
            state["claims"] += 1
            if state["claims"] <= len(jobs):
                return jobs[state["claims"] - 1]
            return None

        def heartbeat(self, worker_id, job_id, lease_token):
            return {"leaseValid": True}

        def download_audio(self, audio_path, expected_size, destination):
            with open(destination, "wb") as target:
                target.write(AUDIO_BYTES)

        def complete_job(self, worker_id, job_id, lease_token, result):
            state["completed"].append(job_id)
            if len(state["completed"]) == len(jobs):
                finished.set()

        def fail_job(self, *args, **kwargs):
            raise AssertionError("batch jobs should complete successfully")

        def unregister(self, worker_id):
            pass

    class CountingEngine:
        def __init__(self):
            self.loaded = False

        def load(self):
            assert not self.loaded
            self.loaded = True
            state["loads"] += 1

        def analyze_batch(self, audio_paths):
            assert self.loaded
            return [
                {
                    "modelId": "clap-test-model",
                    "embeddingVersion": 1,
                    "dimensions": 512,
                    "embedding": [0.125] * 512,
                    "processingSeconds": 0.01,
                    "runtime": "cpu",
                }
                for _ in audio_paths
            ]

        def unload(self):
            self.loaded = False
            state["unloads"] += 1

    settings = Settings(
        gateway_url="http://unused", token="test", worker_id="cpu-batch-test",
        worker_type="cpu", poll_seconds=0.01, heartbeat_seconds=10,
        request_timeout=2, model_path="/unused/model", model_id="clap-test-model",
        max_concurrent_jobs=3, prefetch_batch_size=3, inference_batch_size=1,
        cooldown_seconds=0,
    )
    engine = CountingEngine()
    worker = Worker(settings, BatchClient(), engine)
    thread = Thread(target=worker.run, daemon=True)
    thread.start()

    try:
        assert finished.wait(timeout=5), "CPU worker did not complete its reserved batch"
        assert paused.wait(timeout=1), "CPU worker did not publish its cooldown state"
    finally:
        worker.stop()
        thread.join(timeout=2)

    assert not thread.is_alive()
    assert state["capacity"] == 3
    assert state["claims"] in (3, 4)
    assert len(state["completed"]) == 3
    assert state["loads"] == 1
    assert state["unloads"] == 1
    assert engine.loaded is False
    assert "PAUSED" in state["statuses"]
    assert state["statuses"][-1] in ("READY", "PAUSED")


def test_cpu_worker_pauses_on_gateway_gpu_lock_and_resumes_when_gpu_disappears():
    paused = threading.Event()
    resumed = threading.Event()
    state = {"claims": 0, "statuses": []}
    worker_holder = {}

    class PriorityClient:
        def register(self, *args, **kwargs):
            return True

        def update_worker_status(self, worker_id, status, **metadata):
            state["statuses"].append(status)
            if status == "PAUSED_BY_GPU":
                paused.set()
            if status == "READY" and paused.is_set():
                resumed.set()

        def claim_job(self, worker_id):
            state["claims"] += 1
            if state["claims"] == 1:
                raise GatewayError(423, "GPU_WORKER_ACTIVE", retry_after_seconds=1)
            return None

        def unregister(self, worker_id):
            pass

    settings = Settings(
        gateway_url="http://unused", token="test", worker_id="cpu-priority-test",
        worker_type="cpu", poll_seconds=0.01, heartbeat_seconds=10,
        request_timeout=2, model_path="/unused/model", model_id="clap-test-model",
        max_concurrent_jobs=1, prefetch_batch_size=1, inference_batch_size=1,
        cooldown_seconds=0,
    )
    worker = Worker(settings, PriorityClient(), FakeEngine())
    worker_holder["worker"] = worker
    thread = Thread(target=worker.run, daemon=True)
    thread.start()

    try:
        assert paused.wait(timeout=2), "CPU worker did not report GPU preemption"
        assert worker.is_ready(), "Intentional GPU pause must not make the CPU worker unhealthy"
        assert resumed.wait(timeout=3), "CPU worker did not resume after GPU disappeared"
        assert worker.is_ready(), "Resumed CPU worker should remain healthy"
    finally:
        worker.stop()
        thread.join(timeout=2)

    assert not thread.is_alive()
    assert state["claims"] == 2
    assert "PAUSED_BY_GPU" in state["statuses"]
    assert state["statuses"][-1] == "READY"


def test_cpu_batch_stops_after_inflight_inference_when_gpu_preempts_all_leases():
    gpu_started = threading.Event()
    revoked = threading.Event()
    state = {"completed": [], "statuses": [], "loads": 0, "unloads": 0, "inferences": 0}
    jobs = [
        {
            "jobId": f"preempt-job-{index}",
            "leaseToken": f"preempt-lease-{index}",
            "audio": {"path": f"Artist/Album/{index}.wav", "sizeBytes": len(AUDIO_BYTES)},
        }
        for index in range(3)
    ]

    class PreemptingClient:
        def register(self, *args, **kwargs):
            return True

        def update_worker_status(self, worker_id, status, **metadata):
            state["statuses"].append(status)

        def claim_job(self, worker_id):
            if not hasattr(self, "next_job"):
                self.next_job = 0
            if self.next_job >= len(jobs):
                return None
            job = jobs[self.next_job]
            self.next_job += 1
            return job

        def heartbeat(self, worker_id, job_id, lease_token):
            if not gpu_started.is_set():
                return {"leaseValid": True}
            state["revoked_heartbeats"] = state.get("revoked_heartbeats", 0) + 1
            if state["revoked_heartbeats"] == len(jobs):
                revoked.set()
            return {"leaseValid": False}

        def download_audio(self, audio_path, expected_size, destination):
            with open(destination, "wb") as target:
                target.write(AUDIO_BYTES)

        def complete_job(self, worker_id, job_id, lease_token, result):
            state["completed"].append(job_id)

        def fail_job(self, *args, **kwargs):
            raise AssertionError("Preempted jobs must not be failed by the CPU worker")

        def unregister(self, worker_id):
            pass

    class PreemptAwareEngine:
        def load(self):
            state["loads"] += 1

        def analyze(self, audio_path):
            state["inferences"] += 1
            # Simulate GPU approval while the first file is being processed.
            gpu_started.set()
            assert revoked.wait(timeout=2), "CPU lease heartbeat did not observe GPU preemption"
            return {
                "modelId": "clap-test-model",
                "embeddingVersion": 1,
                "dimensions": 512,
                "embedding": [0.125] * 512,
                "processingSeconds": 0.01,
                "runtime": "cpu",
            }

        def unload(self):
            state["unloads"] += 1

    settings = Settings(
        gateway_url="http://unused", token="test", worker_id="cpu-preempt-test",
        worker_type="cpu", poll_seconds=0.01, heartbeat_seconds=0.02,
        request_timeout=2, model_path="/unused/model", model_id="clap-test-model",
        max_concurrent_jobs=3, prefetch_batch_size=3, inference_batch_size=1,
        cooldown_seconds=0,
    )
    worker = Worker(settings, PreemptingClient(), PreemptAwareEngine())
    jobs_claimed = worker._claim_batch()
    worker._process_cpu_batch(jobs_claimed)

    assert state["completed"] == []
    assert state["inferences"] == 1
    assert state["loads"] == 1
    assert state["unloads"] == 1
    assert "PAUSED_BY_GPU" in state["statuses"]
