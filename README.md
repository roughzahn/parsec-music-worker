# parsec-music-worker

Distributed CPU/GPU worker for generating 512-dimensional LAION-CLAP audio embeddings for Parsec.

## Responsibilities

The worker is intentionally a pull-only execution agent:

- Registers its identity, runtime type and capabilities with the Gateway.
- CPU reserves up to 10 jobs per work cycle; GPU prefetches a configurable queue of jobs from the Gateway.
- Renews every reserved job lease while downloads and batched inference are running.
- Downloads the requested audio through the Gateway's existing private reverse-audio endpoint.
- Generates a 512-dimensional LAION-CLAP embedding.
- Submits the result or reports a failure.
- Does not scan the music library, decide what needs analysis, maintain a local index, or write directly to the Search service.

The Gateway owns inventory synchronization, job creation, scheduling, retry policy and job state. The Search service remains the sole owner of the persistent search index.

## Worker health endpoint

The CPU worker reports 200 UP after registration while CLAP is intentionally unloaded between batches and during the fixed cooldown. The GPU worker reports UP after model initialization. Both workers return 503 during startup, graceful shutdown, or when Gateway contact becomes stale. Docker uses this endpoint for the worker container health check and the Gateway readiness indicator; it is not published to the host or Internet.

## CPU batch cycle

The CPU worker uses a deterministic work/rest cycle instead of host-resource monitoring:

1. Reserve up to PARSEC_CPU_BATCH_SIZE jobs (default 10) through individual atomic Gateway claims.
2. Start lease heartbeats for every reserved job.
3. Load CLAP once and process the reserved jobs sequentially. CPU inference batch size remains 1 to limit transient RAM use.
4. Complete or fail each job independently.
5. Unload CLAP, publish PAUSED, and wait PARSEC_CPU_COOLDOWN_SECONDS (default 300 seconds).
6. Publish READY and begin the next cycle.

The worker stays registered and its health endpoint remains UP while CLAP is unloaded and during the intentional cooldown. No host metrics, LOAD_ALLOWED handshake, or resource-dependent pause response is used. The GPU profile and its batching behavior are unchanged.

## Runtime profiles

Both profiles use the same Python application and analysis implementation. Only the container base image and PyTorch runtime differ.

| Profile | Runtime | Intended use |
|---|---|---|
| `cpu` | PyTorch CPU | Small incremental updates; CPU-only hosts |
| `gpu` | PyTorch CUDA 12.8 | Batched CLAP inference and full-library reindexing |

The model checkpoint is supplied as a read-only host volume. It is not included in the image or Git.

Expected model file:

```text
/models/630k-audioset-best.pt
```

## Configuration

Copy `.env.example` to `.env` and configure:

| Variable | Meaning |
|---|---|
| `PARSEC_GATEWAY_BASE_URL` | Gateway base URL reachable from inside the container |
| `PARSEC_WORKER_TOKEN` | Same private bearer token configured on the Gateway |
| `PARSEC_WORKER_ID` | Stable, unique identity for this worker instance |
| `PARSEC_MODEL_PATH` | Host directory containing the CLAP checkpoint |

Additional runtime settings:

- `PARSEC_WORKER_TYPE`: `cpu` or `gpu` (set by the compose profile).
- `PARSEC_WORKER_POLL_SECONDS`: delay after an empty queue (default 3).
- `PARSEC_WORKER_HEARTBEAT_SECONDS`: lease renewal interval (CPU: 20 seconds; GPU: 10 seconds).
- `PARSEC_WORKER_MAX_CONCURRENT_JOBS`: advertised Gateway reservation capacity (CPU: 1; GPU: same value as prefetch size).
- `PARSEC_GPU_PREFETCH_BATCH_SIZE`: maximum number of GPU jobs reserved ahead (default 20; range 1–50).
- `PARSEC_GPU_INFERENCE_BATCH_SIZE`: number of audio files passed to CLAP in one GPU inference call (default 4; must not exceed prefetch size).
- `PARSEC_CLAP_MODEL_PATH`: model path inside the container.
- `PARSEC_CLAP_MODEL_ID`: embedding model identifier.

Every worker instance must have its own `PARSEC_WORKER_ID`. The same token can be shared by the Audio Node and workers; a dedicated worker token can be introduced later if the Gateway supports separate credentials.

## Run on Windows

Use the existing checkpoint cache without copying the 1.8 GB model into this repository. In `.env`, set `PARSEC_MODEL_PATH=C:/parsec/models` and set the Gateway URL to an address reachable from Docker Desktop (normally `http://host.docker.internal:8766` for a Gateway published on the Windows host).

GPU:

```powershell
docker compose --env-file .env -f compose.gpu.yaml up -d --build
docker compose --env-file .env -f compose.gpu.yaml logs -f
```

CPU:

```powershell
docker compose --env-file .env -f compose.cpu.yaml up -d --build
docker compose --env-file .env -f compose.cpu.yaml logs -f
```

Stop the selected profile with `docker compose ... down`.

## Run on Linux / vServer

Install Docker Engine with the NVIDIA Container Toolkit for the GPU profile. The CPU profile needs only Docker.

Mount the model directory through `PARSEC_MODEL_PATH`. Set the Gateway URL to the internal Docker service address if the worker joins the Gateway network, or to the reachable public Gateway URL otherwise. No inbound connection to the worker is required.

## Pull protocol and lease semantics

The worker API contract is implemented in `parsec-media-gateway`: durable job creation, worker registration and lifecycle status, atomic claim, lease renewal, completion and failure reporting. Gateway and Worker releases must be coordinated before deployment. The existing `POST /internal/v1/audio/content` endpoint is used for audio transfer.

All analysis endpoints use the existing private bearer-token authentication and live below `/internal/`, so the existing Nginx internal-route forwarding rule can protect them.

Worker routes:

- `POST /internal/v1/analysis/workers/register`: register worker ID, CPU/GPU type, model ID, capabilities, capacity and initial `workerStatus` (`STARTING` or `READY`). Legacy clients omitting the field default to `READY`.
- `POST /internal/v1/analysis/workers/{workerId}/status`: publish `STARTING` or `READY` and renew the worker registration heartbeat. A worker must remain `STARTING` until model loading and runtime initialization have completed.
- `POST /internal/v1/analysis/workers/{workerId}/jobs/claim`: atomically reserve one compatible job; return `204` when none is available. Only `READY` workers can claim jobs.
- `POST /internal/v1/analysis/workers/{workerId}/jobs/{jobId}/heartbeat`: renew the lease using its opaque `leaseToken`.
- `POST /internal/v1/analysis/workers/{workerId}/jobs/{jobId}/complete`: submit embedding and source metadata.
- `POST /internal/v1/analysis/workers/{workerId}/jobs/{jobId}/fail`: report failure and whether retry is appropriate.

A claimed job is expected to look like:

```json
{
  "jobId": "opaque-job-id",
  "leaseToken": "opaque-lease-token",
  "leaseExpiresAt": "2026-11-09T23:15:00+01:00",
  "attempt": 1,
  "audio": {
    "path": "Artist/Album/track.flac",
    "sizeBytes": 12345678,
    "lastModifiedUtc": "2026-11-09T12:00:00Z",
    "contentHash": "optional-content-hash"
  }
}
```

The Gateway reserves the job atomically before returning it. The worker renews its lease periodically. If the worker disappears, the lease expires and the Gateway may offer the job to another worker. Completion and failure requests must validate both job ID and lease token; stale workers must not be able to overwrite a newer attempt. The Gateway must also make completion idempotent.

The worker fetches exactly `sizeBytes` from the existing `POST /internal/v1/audio/content` endpoint. The Gateway uses the reverse Audio Node connection to retrieve bytes from the NAS, so a NAS behind NAT never needs an inbound port.

CPU workers continue to execute one job at a time. GPU workers reserve up to 20 jobs by default, keep each lease alive independently, and process the files in inference batches of 4 by default. Each job is still completed or failed independently. Running multiple containers or hosts concurrently is supported by unique worker IDs and Gateway-side atomic reservations.

## Result contract

A successful result contains:

- `modelId`: stable model/checkpoint identifier.
- `embeddingVersion`: preprocessing/output contract version.
- `dimensions`: exactly 512.
- `embedding`: 512 finite float values.
- `processingSeconds` and `runtime`.
- Original audio path, size, modification timestamp and optional content hash.

The Gateway validates the lease and forwards the embedding to the Search service. Workers never modify the index directly.

## Container releases

Pushing a semantic version tag such as v0.1.0 runs the release workflow. It builds and publishes both runtime profiles to GHCR:

- ghcr.io/roughzahn/parsec-music-worker:0.1.0-cpu
- ghcr.io/roughzahn/parsec-music-worker:0.1.0-gpu

The workflow also updates the moving latest-cpu and latest-gpu tags and creates a matching GitHub Release. The CPU and GPU images are released together from the same source revision. No model checkpoint is included in either image.

Set PARSEC_WORKER_VERSION in .env to the numeric release version (for example 0.1.0) to pin both Compose profiles. Use latest only when intentionally tracking the newest release. A release is created by pushing the version tag after the PR has been merged; merging the PR itself does not publish an image.

## Development and tests

Python 3.11 is used for both images.

```powershell
python -m pip install -r requirements-test.txt
python -m pytest -q
```

The tests do not download PyTorch or the CLAP model. Full model execution is performed in the CPU/GPU containers.

## Operational notes

- Time zone is explicitly set to `Europe/Berlin` in both images and compose profiles.
- Model files, local environment files, generated embeddings and music files must never be committed.
- The old one-shot scanner and CSV-based index builder are intentionally not part of this distributed worker. Inventory discovery and index updates belong to the Gateway and Search service respectively.
- Do not deploy this worker against production until Gateway PR #2 is merged and released, and the full Audio Node → Gateway → worker → Search integration test has passed.

## Windows production GPU worker updater

The production GPU worker can be updated independently of the development and E2E Compose profiles. The versioned, double-click updater is located at:

```text
scripts/windows/gpu-worker/
```

On the Windows host, check out this repository and double-click `update-gpu-worker.cmd` in that directory. The updater pulls `ghcr.io/roughzahn/parsec-music-worker:stable-gpu` first and only recreates the production container after the pull succeeds. It then reports the container state, worker health, software version and recent logs. It does not publish host ports and does not change the worker's registration or manual approval behavior.

The production configuration and persistent data deliberately remain outside the repository:

- `C:\parsec\gpu-worker-prod.env` — local worker settings; never commit this file.
- `C:\parsec\models\630k-audioset-best.pt` — model checkpoint, mounted read-only.
- `C:\parsec\models\huggingface` — persistent Hugging Face cache.

These paths are host data locations, while the updater and Compose file are resolved relative to the checked-out repository. The Windows production updater is separate from `compose.gpu.yaml` (development) and any E2E test configuration. It tracks only the published `stable-gpu` image; it never builds an image from the checked-out source.
