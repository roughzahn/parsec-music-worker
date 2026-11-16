import os
import tempfile
import threading
import time
from .gateway_client import GatewayClient, GatewayError


class Worker:
    def __init__(self, settings, client: GatewayClient, engine):
        self.settings = settings
        self.client = client
        self.engine = engine
        self.stopping = threading.Event()
        self.ready = threading.Event()
        self.worker_status_lock = threading.Lock()
        self.worker_status = "STARTING"
        self.registration_heartbeat_stop = threading.Event()
        self.last_gateway_contact = None
        self.load_paused = False
        self.idle_since = None

    def is_ready(self):
        if not (self.ready.is_set() or self.load_paused) or self.stopping.is_set() or self.last_gateway_contact is None:
            return False
        stale_after = max(60.0, self.settings.heartbeat_seconds * 3, self.settings.poll_seconds * 5)
        return time.monotonic() - self.last_gateway_contact < stale_after

    def _mark_gateway_contact(self):
        self.last_gateway_contact = time.monotonic()

    def run(self):
        registered = False
        status_stop = self.registration_heartbeat_stop
        status_thread = None
        try:
            print(f"Registering worker {self.settings.worker_id} ({self.settings.worker_type}) as STARTING", flush=True)
            if self.settings.worker_type == "gpu":
                registered = self.client.register(
                    self.settings.worker_id, self.settings.worker_type,
                    self.settings.model_id, self.settings.max_concurrent_jobs,
                    worker_status="STARTING", stop_event=self.stopping)
            else:
                self.client.register(self.settings.worker_id, self.settings.worker_type,
                                     self.settings.model_id, self.settings.max_concurrent_jobs,
                                     worker_status="STARTING")
                registered = True
            if not registered:
                return
            self._mark_gateway_contact()
            status_thread = threading.Thread(
                target=self._registration_heartbeat_loop, args=(status_stop,),
                name=f"worker-status-{self.settings.worker_id}", daemon=True)
            status_thread.start()

            if self.settings.worker_type == "cpu":
                # Register and stay healthy without loading CLAP. The Gateway grants
                # LOAD_ALLOWED only when a queued job exists and host RAM has headroom.
                self.load_paused = True
                with self.worker_status_lock:
                    self.worker_status = "IDLE"
                    self.client.update_worker_status(self.settings.worker_id, "IDLE")
                    self._mark_gateway_contact()
                print("CPU worker is IDLE; CLAP stays unloaded until a job is available", flush=True)
            else:
                print("Loading CLAP model and initializing runtime", flush=True)
                self.engine.load()
                if self.stopping.is_set():
                    return
                with self.worker_status_lock:
                    self.worker_status = "READY"
                    self.client.update_worker_status(self.settings.worker_id, "READY")
                    self._mark_gateway_contact()
                self.ready.set()
                print("Worker is READY; polling for analysis jobs", flush=True)
            if self.settings.worker_type == "gpu":
                print(
                    f"GPU batching enabled: prefetch={self.settings.prefetch_batch_size}, "
                    f"inference_batch={self.settings.inference_batch_size}",
                    flush=True,
                )
            while not self.stopping.is_set():
                try:
                    if self.settings.worker_type == "gpu":
                        jobs = self._claim_batch()
                        if not jobs:
                            self.stopping.wait(self.settings.poll_seconds)
                            continue
                        self._process_batch(jobs)
                    else:
                        job = self.client.claim_job(self.settings.worker_id)
                        self._mark_gateway_contact()
                        if job and job.get("status") == "LOAD_ALLOWED":
                            self.idle_since = None
                            self._ensure_engine_loaded()
                            continue
                        if not job:
                            self._unload_after_idle()
                            self.stopping.wait(self.settings.poll_seconds)
                            continue
                        self.idle_since = None
                        self._ensure_engine_loaded()
                        self._process(job)
                except GatewayError as exc:
                    if self.settings.worker_type == "cpu" and exc.status_code == 429:
                        self._pause_and_unload(exc)
                        continue
                    print(f"Gateway communication error: {exc}", flush=True)
                    if self.settings.worker_type == "gpu" and exc.status_code in (401, 403):
                        print("GPU worker session is no longer valid; stopping. Restart and approve the new pairing request.", flush=True)
                        self.stop()
                    else:
                        self.stopping.wait(self.settings.poll_seconds)
                except Exception as exc:
                    print(f"Worker loop error: {type(exc).__name__}: {exc}", flush=True)
                    self.stopping.wait(self.settings.poll_seconds)
        finally:
            self.ready.clear()
            status_stop.set()
            if status_thread is not None:
                status_thread.join(timeout=2)
            if registered:
                try:
                    self.client.unregister(self.settings.worker_id)
                    print(f"Worker {self.settings.worker_id} unregistered", flush=True)
                except Exception as exc:
                    print(f"Could not unregister worker {self.settings.worker_id}: {exc}", flush=True)

    def _unload_after_idle(self):
        if self.settings.worker_type != "cpu":
            return
        checker = getattr(self.engine, "is_loaded", None)
        loaded = checker() if callable(checker) else getattr(self.engine, "loaded", True)
        if not loaded:
            self.idle_since = None
            return
        now = time.monotonic()
        if self.idle_since is None:
            self.idle_since = now
            return
        if now - self.idle_since < self.settings.idle_unload_seconds:
            return
        if hasattr(self.engine, "unload"):
            self.engine.unload()
        self.load_paused = True
        self.ready.clear()
        self.idle_since = None
        with self.worker_status_lock:
            self.worker_status = "IDLE"
            try:
                self.client.update_worker_status(self.settings.worker_id, "IDLE")
                self._mark_gateway_contact()
            except Exception:
                # Never continue polling as READY after unloading the model.
                self.stop()
                raise
        print("CPU worker idle timeout reached; CLAP unloaded and worker returned to IDLE", flush=True)

    def _pause_and_unload(self, error):
        self.load_paused = True
        self._mark_gateway_contact()
        checker = getattr(self.engine, "is_loaded", None)
        loaded = checker() if callable(checker) else getattr(self.engine, "loaded", True)
        if loaded and hasattr(self.engine, "unload"):
            self.engine.unload()
        with self.worker_status_lock:
            self.worker_status = "PAUSED"
            try:
                self.client.update_worker_status(self.settings.worker_id, "PAUSED")
                self._mark_gateway_contact()
            except Exception as exc:
                print(f"Could not report PAUSED worker status: {exc}", flush=True)
        delay = error.retry_after_seconds or 30
        action = "unloaded" if loaded else "remains unloaded"
        print(f"Host load protection paused CPU worker; CLAP {action}, retry in {delay}s", flush=True)
        self.stopping.wait(max(1, delay))

    def _ensure_engine_loaded(self):
        checker = getattr(self.engine, "is_loaded", None)
        loaded = checker() if callable(checker) else getattr(self.engine, "loaded", True)
        if loaded:
            return
        # Keep health/readiness UP while an intentionally unloaded worker loads CLAP.
        self.ready.clear()
        print("CPU job available; loading CLAP model", flush=True)
        self.engine.load()
        self.load_paused = False
        with self.worker_status_lock:
            self.worker_status = "READY"
            self.client.update_worker_status(self.settings.worker_id, "READY")
            self._mark_gateway_contact()
        self.ready.set()

    def _claim_batch(self):
        jobs = []
        for _ in range(self.settings.prefetch_batch_size):
            if self.stopping.is_set():
                break
            job = self.client.claim_job(self.settings.worker_id)
            self._mark_gateway_contact()
            if not job:
                break
            jobs.append(job)
        if jobs:
            print(f"Reserved {len(jobs)} GPU jobs", flush=True)
        return jobs

    def _registration_heartbeat_loop(self, stop_event):
        while not stop_event.wait(self.settings.heartbeat_seconds):
            try:
                with self.worker_status_lock:
                    self.client.update_worker_status(self.settings.worker_id, self.worker_status)
                    if self.settings.worker_type == "gpu":
                        self.client.session_heartbeat(self.settings.worker_id)
                    self._mark_gateway_contact()
            except GatewayError as exc:
                print(f"Worker session heartbeat warning: {exc}", flush=True)
                if self.settings.worker_type == "gpu" and exc.status_code in (401, 403):
                    print("GPU worker session expired; stopping for a fresh manual pairing.", flush=True)
                    self.stop()
                    return
            except Exception as exc:
                print(f"Worker status heartbeat warning: {exc}", flush=True)

    def stop(self):
        self.ready.clear()
        self.stopping.set()
        self.registration_heartbeat_stop.set()

    def _start_job_heartbeat(self, job):
        job_id = str(job["jobId"])
        lease_token = str(job["leaseToken"])
        lease_lost = threading.Event()
        heartbeat_stop = threading.Event()

        def heartbeat_loop():
            while not heartbeat_stop.wait(self.settings.heartbeat_seconds):
                try:
                    response = self.client.heartbeat(self.settings.worker_id, job_id, lease_token)
                    self._mark_gateway_contact()
                    if response and response.get("leaseValid") is False:
                        lease_lost.set()
                        print(f"Lease lost for job {job_id}", flush=True)
                        return
                except GatewayError as exc:
                    if exc.status_code in (404, 409, 410):
                        lease_lost.set()
                        print(f"Lease expired for job {job_id}", flush=True)
                        return
                    if self.settings.worker_type == "gpu" and exc.status_code in (401, 403):
                        self.stop()
                        lease_lost.set()
                        print("GPU worker authorization expired during lease heartbeat.", flush=True)
                        return
                    print(f"Heartbeat warning for {job_id}: {exc}", flush=True)
                except Exception as exc:
                    print(f"Heartbeat warning for {job_id}: {exc}", flush=True)

        heartbeat = threading.Thread(target=heartbeat_loop, name=f"heartbeat-{job_id}", daemon=True)
        heartbeat.start()
        return heartbeat_stop, lease_lost, heartbeat

    @staticmethod
    def _audio_metadata(job):
        audio = job["audio"]
        return {
            "path": str(audio["path"]),
            "sizeBytes": int(audio["sizeBytes"]),
            "lastModifiedUtc": audio.get("lastModifiedUtc"),
            "contentHash": audio.get("contentHash"),
        }

    def _process_batch(self, jobs):
        heartbeats = {}
        downloaded = {}
        outcomes = {}
        try:
            for job in jobs:
                job_id = str(job["jobId"])
                heartbeats[job_id] = self._start_job_heartbeat(job)

            with tempfile.TemporaryDirectory(prefix="parsec-gpu-batch-") as temp_dir:
                for index, job in enumerate(jobs):
                    job_id = str(job["jobId"])
                    if self.stopping.is_set():
                        outcomes[job_id] = RuntimeError("Worker is stopping")
                        continue
                    if heartbeats[job_id][1].is_set():
                        continue
                    audio = job["audio"]
                    source = os.path.join(temp_dir, f"source-{index}")
                    try:
                        self.client.download_audio(
                            str(audio["path"]), int(audio["sizeBytes"]), source
                        )
                        self._mark_gateway_contact()
                        downloaded[job_id] = source
                    except Exception as exc:
                        outcomes[job_id] = exc

                valid_jobs = [
                    job for job in jobs
                    if str(job["jobId"]) in downloaded
                    and not heartbeats[str(job["jobId"])][1].is_set()
                ]
                for offset in range(0, len(valid_jobs), self.settings.inference_batch_size):
                    chunk = valid_jobs[offset:offset + self.settings.inference_batch_size]
                    try:
                        if hasattr(self.engine, "analyze_batch"):
                            chunk_results = self.engine.analyze_batch(
                                [downloaded[str(job["jobId"])] for job in chunk]
                            )
                        else:
                            chunk_results = [
                                self.engine.analyze(downloaded[str(job["jobId"])])
                                for job in chunk
                            ]
                        if len(chunk_results) != len(chunk):
                            raise RuntimeError("CLAP returned an unexpected number of batch results")
                        for job, result in zip(chunk, chunk_results):
                            outcomes[str(job["jobId"])] = result
                    except Exception as exc:
                        for job in chunk:
                            outcomes[str(job["jobId"])] = exc

            for job in jobs:
                job_id = str(job["jobId"])
                lease_lost = heartbeats[job_id][1]
                if lease_lost.is_set():
                    print(f"Discarding result for expired lease {job_id}", flush=True)
                    continue
                if self.stopping.is_set():
                    continue
                outcome = outcomes.get(job_id, RuntimeError("No analysis result was produced"))
                lease_token = str(job["leaseToken"])
                try:
                    if isinstance(outcome, Exception):
                        raise outcome
                    result = dict(outcome)
                    result["audio"] = self._audio_metadata(job)
                    self.client.complete_job(self.settings.worker_id, job_id, lease_token, result)
                    self._mark_gateway_contact()
                    print(
                        f"Completed job {job_id} ({result.get('processingSeconds', 0)}s)",
                        flush=True,
                    )
                except Exception as exc:
                    if isinstance(exc, GatewayError) and exc.status_code in (401, 403):
                        print("GPU worker authorization expired while processing; stopping for re-approval.", flush=True)
                        self.stop()
                    elif not lease_lost.is_set():
                        try:
                            self.client.fail_job(
                                self.settings.worker_id, job_id, lease_token,
                                f"{type(exc).__name__}: {exc}", retryable=True,
                            )
                        except Exception as report_error:
                            print(f"Could not report job failure {job_id}: {report_error}", flush=True)
                    print(f"Job {job_id} failed: {type(exc).__name__}: {exc}", flush=True)
        finally:
            for heartbeat_stop, _, heartbeat in heartbeats.values():
                heartbeat_stop.set()
                heartbeat.join(timeout=2)

    def _process(self, job: dict):
        job_id = str(job["jobId"])
        lease_token = str(job["leaseToken"])
        audio = job["audio"]
        audio_path = str(audio["path"])
        expected_size = int(audio["sizeBytes"])
        heartbeat_stop, lease_lost, heartbeat = self._start_job_heartbeat(job)
        print(f"Claimed job {job_id}: {audio_path}", flush=True)
        try:
            with tempfile.TemporaryDirectory(prefix="parsec-job-") as temp_dir:
                source = os.path.join(temp_dir, "source-audio")
                self.client.download_audio(audio_path, expected_size, source)
                result = self.engine.analyze(source)
            if lease_lost.is_set():
                print(f"Discarding result for expired lease {job_id}", flush=True)
                return
            result["audio"] = self._audio_metadata(job)
            self.client.complete_job(self.settings.worker_id, job_id, lease_token, result)
            print(f"Completed job {job_id} ({result['processingSeconds']}s)", flush=True)
        except Exception as exc:
            if self.settings.worker_type == "gpu" and isinstance(exc, GatewayError) and exc.status_code in (401, 403):
                print("GPU worker authorization expired while processing; stopping for re-approval.", flush=True)
                self.stop()
            if not lease_lost.is_set() and not (self.settings.worker_type == "gpu" and isinstance(exc, GatewayError) and exc.status_code in (401, 403)):
                try:
                    self.client.fail_job(self.settings.worker_id, job_id, lease_token,
                                         f"{type(exc).__name__}: {exc}", retryable=True)
                except Exception as report_error:
                    print(f"Could not report job failure {job_id}: {report_error}", flush=True)
            print(f"Job {job_id} failed: {type(exc).__name__}: {exc}", flush=True)
        finally:
            heartbeat_stop.set()
            heartbeat.join(timeout=2)
