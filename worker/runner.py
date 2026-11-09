import os
import tempfile
import threading
from urllib.parse import quote

from .gateway_client import GatewayClient, GatewayError


class Worker:
    def __init__(self, settings, client: GatewayClient, engine):
        self.settings = settings
        self.client = client
        self.engine = engine
        self.stopping = threading.Event()

    def run(self):
        self.engine.load()
        print(f"Registering worker {self.settings.worker_id} ({self.settings.worker_type})", flush=True)
        self.client.register(self.settings.worker_id, self.settings.worker_type,
                             self.settings.model_id, self.settings.max_concurrent_jobs)
        print("Worker registered; polling for analysis jobs", flush=True)
        while not self.stopping.is_set():
            try:
                job = self.client.claim_job(quote(self.settings.worker_id, safe=""))
                if not job:
                    self.stopping.wait(self.settings.poll_seconds)
                    continue
                self._process(job)
            except GatewayError as exc:
                print(f"Gateway communication error: {exc}", flush=True)
                self.stopping.wait(self.settings.poll_seconds)
            except Exception as exc:
                print(f"Worker loop error: {type(exc).__name__}: {exc}", flush=True)
                self.stopping.wait(self.settings.poll_seconds)

    def stop(self):
        self.stopping.set()

    def _process(self, job: dict):
        job_id = str(job["jobId"])
        lease_token = str(job["leaseToken"])
        audio = job["audio"]
        audio_path = str(audio["path"])
        expected_size = int(audio["sizeBytes"])
        lease_lost = threading.Event()
        heartbeat_stop = threading.Event()

        def heartbeat_loop():
            while not heartbeat_stop.wait(self.settings.heartbeat_seconds):
                try:
                    response = self.client.heartbeat(self.settings.worker_id, job_id, lease_token)
                    if response and response.get("leaseValid") is False:
                        lease_lost.set()
                        print(f"Lease lost for job {job_id}", flush=True)
                        return
                except GatewayError as exc:
                    if exc.status_code in (404, 409, 410):
                        lease_lost.set()
                        print(f"Lease expired for job {job_id}", flush=True)
                        return
                    print(f"Heartbeat warning for {job_id}: {exc}", flush=True)
                except Exception as exc:
                    print(f"Heartbeat warning for {job_id}: {exc}", flush=True)

        heartbeat = threading.Thread(target=heartbeat_loop, name=f"heartbeat-{job_id}", daemon=True)
        heartbeat.start()
        print(f"Claimed job {job_id}: {audio_path}", flush=True)
        try:
            with tempfile.TemporaryDirectory(prefix="parsec-job-") as temp_dir:
                source = os.path.join(temp_dir, "source-audio")
                self.client.download_audio(audio_path, expected_size, source)
                result = self.engine.analyze(source)
            if lease_lost.is_set():
                print(f"Discarding result for expired lease {job_id}", flush=True)
                return
            result["audio"] = {
                "path": audio_path,
                "sizeBytes": expected_size,
                "lastModifiedUtc": audio.get("lastModifiedUtc"),
                "contentHash": audio.get("contentHash"),
            }
            self.client.complete_job(self.settings.worker_id, job_id, lease_token, result)
            print(f"Completed job {job_id} ({result['processingSeconds']}s)", flush=True)
        except Exception as exc:
            if not lease_lost.is_set():
                try:
                    self.client.fail_job(self.settings.worker_id, job_id, lease_token,
                                         f"{type(exc).__name__}: {exc}", retryable=True)
                except Exception as report_error:
                    print(f"Could not report job failure {job_id}: {report_error}", flush=True)
            print(f"Job {job_id} failed: {type(exc).__name__}: {exc}", flush=True)
        finally:
            heartbeat_stop.set()
            heartbeat.join(timeout=2)
