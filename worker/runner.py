import os
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
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
        self.pause_started_at = None
        self.cooldown_until = None
        self.registration_heartbeat_stop = threading.Event()
        self.last_gateway_contact = None


    def is_ready(self):
        if not self.ready.is_set() or self.stopping.is_set() or self.last_gateway_contact is None:
            return False
        stale_after = max(60.0, self.settings.heartbeat_seconds * 3, self.settings.poll_seconds * 5)
        return time.monotonic() - self.last_gateway_contact < stale_after

    def _mark_gateway_contact(self):
        self.last_gateway_contact = time.monotonic()

    def _set_worker_status(self, status):
        with self.worker_status_lock:
            if self.worker_status == status:
                return
            self.worker_status = status
            self.pause_started_at = None
            self.cooldown_until = None
            self.client.update_worker_status(self.settings.worker_id, status)
            self._mark_gateway_contact()

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
                    worker_status="STARTING", stop_event=self.stopping,
                    worker_version=self.settings.software_version)
            else:
                self.client.register(self.settings.worker_id, self.settings.worker_type,
                                     self.settings.model_id, self.settings.max_concurrent_jobs,
                                     worker_status="STARTING", worker_version=self.settings.software_version)
                registered = True
            if not registered:
                return
            self._mark_gateway_contact()
            status_thread = threading.Thread(
                target=self._registration_heartbeat_loop, args=(status_stop,),
                name=f"worker-status-{self.settings.worker_id}", daemon=True)
            status_thread.start()

            if self.settings.worker_type == "cpu":
                # Stay registered and healthy with CLAP unloaded between fixed batches.
                with self.worker_status_lock:
                    self.worker_status = "READY"
                    self.client.update_worker_status(self.settings.worker_id, "READY")
                    self._mark_gateway_contact()
                self.ready.set()
                print(
                    f"CPU worker is READY; batch_size={self.settings.prefetch_batch_size}, "
                    f"cooldown={self.settings.cooldown_seconds}s; CLAP stays unloaded between batches",
                    flush=True,
                )
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
                        jobs = self._claim_batch()
                        if self.worker_status == "PAUSED_BY_GPU":
                            self._set_worker_status("READY")
                            print("GPU worker is no longer active; CPU worker resumed", flush=True)
                        if not jobs:
                            self.stopping.wait(self.settings.poll_seconds)
                            continue
                        self._process_cpu_batch(jobs)
                except GatewayError as exc:
                    if self.settings.worker_type == "cpu" and exc.status_code == 423:
                        try:
                            self._set_worker_status("PAUSED_BY_GPU")
                        except Exception as status_error:
                            print(f"Could not publish GPU pause status: {status_error}", flush=True)
                        print("CPU worker paused because a GPU worker is active", flush=True)
                        self.stopping.wait(max(self.settings.poll_seconds, exc.retry_after_seconds))
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

    def _claim_batch(self):
        jobs = []
        for _ in range(self.settings.prefetch_batch_size):
            if self.stopping.is_set():
                break
            try:
                job = self.client.claim_job(self.settings.worker_id)
            except GatewayError as exc:
                if exc.status_code == 423:
                    # The Gateway has already returned preempted CPU leases to the queue.
                    raise
                if jobs:
                    print(f"Gateway interrupted batch reservation after {len(jobs)} jobs; processing the partial batch: {exc}", flush=True)
                    break
                raise
            self._mark_gateway_contact()
            if not job:
                break
            jobs.append(job)
        if jobs:
            print(f"Reserved {len(jobs)} {self.settings.worker_type.upper()} jobs", flush=True)
        return jobs

    def _registration_heartbeat_loop(self, stop_event):
        while not stop_event.wait(self.settings.heartbeat_seconds):
            try:
                with self.worker_status_lock:
                    self.client.update_worker_status(
                        self.settings.worker_id, self.worker_status,
                        pause_started_at=self.pause_started_at, cooldown_until=self.cooldown_until)
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

    def _process_cpu_batch(self, jobs):
        """Load CLAP once, process a reserved CPU batch, unload, then cool down."""
        heartbeats = {}
        for job in jobs:
            job_id = str(job["jobId"])
            heartbeats[job_id] = self._start_job_heartbeat(job)

        try:
            if any(lease_lost.is_set() for _, lease_lost, _ in heartbeats.values()):
                print("CPU batch was preempted before model load; skipping CLAP initialization", flush=True)
            else:
                print(f"CPU batch reserved: {len(jobs)} jobs; loading CLAP once", flush=True)
                self.engine.load()
                if not self.stopping.is_set():
                    self._process_batch(jobs, heartbeats=heartbeats)
        except Exception as exc:
            print(f"CPU batch failed during model initialization: {type(exc).__name__}: {exc}", flush=True)
            for job in jobs:
                job_id = str(job["jobId"])
                if heartbeats[job_id][1].is_set() or self.stopping.is_set():
                    continue
                try:
                    self.client.fail_job(
                        self.settings.worker_id, job_id, str(job["leaseToken"]),
                        f"{type(exc).__name__}: {exc}", retryable=True,
                    )
                except Exception as report_error:
                    print(f"Could not report batch failure for {job_id}: {report_error}", flush=True)
        finally:
            for heartbeat_stop, _, heartbeat in heartbeats.values():
                heartbeat_stop.set()
                heartbeat.join(timeout=2)
            unload_succeeded = False
            if hasattr(self.engine, "unload"):
                try:
                    self.engine.unload()
                    unload_succeeded = True
                    print("CPU batch finished; CLAP model unloaded", flush=True)
                except Exception as exc:
                    print(f"Could not fully unload CLAP after CPU batch: {exc}", flush=True)
            if not unload_succeeded:
                # Never start another batch when releasing the model could not be confirmed.
                print("CPU worker stopping because CLAP could not be safely unloaded", flush=True)
                self.stop()

        if self.stopping.is_set():
            return
        preempted = any(lease_lost.is_set() for _, lease_lost, _ in heartbeats.values())
        if preempted:
            try:
                self._set_worker_status("PAUSED_BY_GPU")
            except Exception as exc:
                print(f"Could not publish GPU preemption status: {exc}", flush=True)
            print("CPU batch preempted by GPU worker; skipping cooldown", flush=True)
            return
        pause_started = datetime.now(timezone.utc)
        cooldown_ends = pause_started + timedelta(seconds=self.settings.cooldown_seconds)
        self.pause_started_at = pause_started.isoformat()
        self.cooldown_until = cooldown_ends.isoformat()
        with self.worker_status_lock:
            self.worker_status = "PAUSED"
            try:
                self.client.update_worker_status(
                    self.settings.worker_id, "PAUSED",
                    pause_started_at=self.pause_started_at, cooldown_until=self.cooldown_until)
                self._mark_gateway_contact()
            except Exception as exc:
                print(f"Could not publish CPU cooldown status: {exc}", flush=True)
        print(f"CPU worker cooling down for {self.settings.cooldown_seconds}s", flush=True)
        self.stopping.wait(self.settings.cooldown_seconds)
        if not self.stopping.is_set():
            with self.worker_status_lock:
                self.worker_status = "READY"
                self.pause_started_at = None
                self.cooldown_until = None
                self.client.update_worker_status(self.settings.worker_id, "READY")
                self._mark_gateway_contact()
            print("CPU worker cooldown complete; READY for the next batch", flush=True)
    def _process_batch(self, jobs, heartbeats=None):
        owns_heartbeats = heartbeats is None
        if heartbeats is None:
            heartbeats = {}
            for job in jobs:
                job_id = str(job["jobId"])
                heartbeats[job_id] = self._start_job_heartbeat(job)
        downloaded = {}
        outcomes = {}
        try:

            with tempfile.TemporaryDirectory(prefix="parsec-worker-batch-") as temp_dir:
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
                    # A GPU registration revokes all outstanding CPU leases. Re-check
                    # immediately before every inference chunk so the CPU stops at the
                    # current file boundary rather than finishing its prefetched batch.
                    chunk = [
                        job for job in valid_jobs[offset:offset + self.settings.inference_batch_size]
                        if not heartbeats[str(job["jobId"])][1].is_set()
                    ]
                    if not chunk:
                        continue
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
                    if isinstance(exc, GatewayError) and exc.status_code in (404, 409, 410):
                        lease_lost.set()
                        print(f"Lease was revoked while completing job {job_id}", flush=True)
                    elif isinstance(exc, GatewayError) and exc.status_code in (401, 403):
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
            if owns_heartbeats:
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
