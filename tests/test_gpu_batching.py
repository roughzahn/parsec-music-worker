import threading
import time

from worker.runner import Worker


def make_job(index):
    return {
        "jobId": f"job-{index}",
        "leaseToken": f"lease-{index}",
        "audio": {
            "path": f"artist/track-{index}.wav",
            "sizeBytes": 4,
            "lastModifiedUtc": "2026-11-15T18:00:00Z",
            "contentHash": f"hash-{index}",
        },
    }


class FakeClient:
    def __init__(self, jobs=None, fail_download=None):
        self.jobs = list(jobs or [])
        self.claim_calls = 0
        self.downloaded = []
        self.completed = []
        self.failed = []
        self.heartbeats = []
        self.fail_download = fail_download

    def claim_job(self, worker_id):
        self.claim_calls += 1
        return self.jobs.pop(0) if self.jobs else None

    def download_audio(self, path, expected_size, destination):
        if self.fail_download == path:
            raise RuntimeError("download failed")
        with open(destination, "wb") as output:
            output.write(b"data")
        self.downloaded.append(path)

    def heartbeat(self, worker_id, job_id, lease_token):
        self.heartbeats.append(job_id)
        return {"leaseValid": True}

    def complete_job(self, worker_id, job_id, lease_token, result):
        self.completed.append((job_id, lease_token, result))

    def fail_job(self, worker_id, job_id, lease_token, error, retryable=True):
        self.failed.append((job_id, lease_token, error, retryable))


class FakeBatchEngine:
    def __init__(self, failed_index=None):
        self.batch_sizes = []
        self.failed_index = failed_index

    def analyze_batch(self, paths):
        self.batch_sizes.append(len(paths))
        results = []
        for index, _ in enumerate(paths):
            if index == self.failed_index:
                results.append(RuntimeError("synthetic inference failure"))
            else:
                results.append({
                    "modelId": "test",
                    "embeddingVersion": 1,
                    "dimensions": 512,
                    "embedding": [0.0] * 512,
                    "processingSeconds": 0.1,
                    "runtime": "gpu",
                })
        return results


def make_worker(client, engine, prefetch=20, inference_batch=4):
    settings = type("Settings", (), {
        "worker_id": "gpu-test",
        "worker_type": "gpu",
        "poll_seconds": 0.01,
        "heartbeat_seconds": 10,
        "prefetch_batch_size": prefetch,
        "inference_batch_size": inference_batch,
    })()
    return Worker(settings, client, engine)


def test_gpu_claims_up_to_prefetch_batch_size():
    jobs = [make_job(i) for i in range(20)]
    client = FakeClient(jobs)
    worker = make_worker(client, FakeBatchEngine())

    claimed = worker._claim_batch()

    assert len(claimed) == 20
    assert client.claim_calls == 20


def test_gpu_processes_prefetched_jobs_in_inference_batches_and_completes_each():
    jobs = [make_job(i) for i in range(9)]
    client = FakeClient()
    engine = FakeBatchEngine()
    worker = make_worker(client, engine, prefetch=20, inference_batch=4)

    worker._process_batch(jobs)

    assert engine.batch_sizes == [4, 4, 1]
    assert len(client.completed) == 9
    assert client.failed == []
    assert all(item[2]["audio"]["path"] for item in client.completed)


def test_gpu_batch_failure_is_reported_per_job_without_losing_other_results():
    jobs = [make_job(i) for i in range(3)]
    client = FakeClient(fail_download=jobs[1]["audio"]["path"])
    engine = FakeBatchEngine()
    worker = make_worker(client, engine, prefetch=3, inference_batch=3)

    worker._process_batch(jobs)

    assert len(client.completed) == 2
    assert len(client.failed) == 1
    assert client.failed[0][0] == "job-1"
    assert "download failed" in client.failed[0][2]


def test_gpu_inference_failure_is_reported_for_that_job_only():
    jobs = [make_job(i) for i in range(3)]
    client = FakeClient()
    engine = FakeBatchEngine(failed_index=1)
    worker = make_worker(client, engine, prefetch=3, inference_batch=3)

    worker._process_batch(jobs)

    assert [item[0] for item in client.completed] == ["job-0", "job-2"]
    assert len(client.failed) == 1
    assert client.failed[0][0] == "job-1"
    assert "synthetic inference failure" in client.failed[0][2]


def test_gpu_renews_leases_for_all_prefetched_jobs_while_batch_is_processing():
    jobs = [make_job(i) for i in range(3)]
    client = FakeClient()
    entered = threading.Event()
    release = threading.Event()

    class BlockingEngine(FakeBatchEngine):
        def analyze_batch(self, paths):
            entered.set()
            assert release.wait(timeout=3)
            return super().analyze_batch(paths)

    worker = make_worker(client, BlockingEngine(), prefetch=3, inference_batch=3)
    thread = threading.Thread(target=worker._process_batch, args=(jobs,), daemon=True)
    worker.settings.heartbeat_seconds = 0.02
    thread.start()
    try:
        assert entered.wait(timeout=2)
        deadline = time.monotonic() + 2
        while len(set(client.heartbeats)) < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert set(client.heartbeats) == {"job-0", "job-1", "job-2"}
    finally:
        release.set()
        thread.join(timeout=3)
    assert not thread.is_alive()
    assert len(client.completed) == 3
