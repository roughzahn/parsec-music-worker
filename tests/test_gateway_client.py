from worker.gateway_client import GatewayClient


def test_gateway_client_auth_and_routes():
    client = GatewayClient("http://gateway/", "secret")
    assert client._url("/internal/v1/analysis/workers/cpu-1/jobs/claim") == (
        "http://gateway/internal/v1/analysis/workers/cpu-1/jobs/claim"
    )
    assert client.session.headers["Authorization"] == "Bearer secret"
    assert client.session.headers["User-Agent"].startswith("parsec-music-worker/")


def test_worker_status_route_url_encodes_worker_id():
    client = GatewayClient("http://gateway", "secret")
    captured = {}
    client._json = lambda method, path, **kwargs: captured.update(
        method=method, path=path, kwargs=kwargs
    )

    client.update_worker_status("gpu worker/1", "starting")

    assert captured["method"] == "POST"
    assert captured["path"] == "/internal/v1/analysis/workers/gpu%20worker%2F1/status"
    assert captured["kwargs"]["json"] == {"status": "STARTING", "pauseStartedAt": None, "cooldownUntil": None}


def test_worker_status_forwards_actual_cooldown_timestamps():
    client = GatewayClient("http://gateway", "secret")
    captured = {}
    client._json = lambda method, path, **kwargs: captured.update(method=method, path=path, kwargs=kwargs)

    client.update_worker_status(
        "cpu-1", "paused", pause_started_at="2026-11-17T18:00:00+00:00",
        cooldown_until="2026-11-17T18:05:00+00:00")

    assert captured["kwargs"]["json"] == {
        "status": "PAUSED",
        "pauseStartedAt": "2026-11-17T18:00:00+00:00",
        "cooldownUntil": "2026-11-17T18:05:00+00:00",
    }


def test_gpu_pairing_sets_only_ephemeral_authorization_after_approval(monkeypatch):
    client = GatewayClient("https://gateway", None)
    calls = []

    def fake_json(method, path, **kwargs):
        calls.append((method, path, kwargs))
        if path.endswith("/register"):
            return {"registrationId": "registration-1", "bootstrapSecret": "bootstrap-only"}
        return {"status": "APPROVED", "sessionToken": "ephemeral-session", "expiresInSeconds": 30}

    monkeypatch.setattr(client, "_json", fake_json)

    assert client.register("gpu-host-1", "gpu", "clap-v1", 1)
    assert client.session.headers["Authorization"] == "Bearer ephemeral-session"
    assert client.session.headers["X-Parsec-Worker-Id"] == "gpu-host-1"
    assert calls[0][2]["json"]["workerId"] == "gpu-host-1"
    assert calls[1][2]["json"]["bootstrapSecret"] == "bootstrap-only"


def test_gpu_audio_download_includes_worker_identity_header(monkeypatch, tmp_path):
    client = GatewayClient("https://gateway", None)
    client.worker_id = "gpu-host-1"
    captured = {}

    class Response:
        status_code = 200
        def iter_content(self, chunk_size):
            yield b"abc"

        def close(self):
            pass

    def post(session, *args, **kwargs):
        captured.update(kwargs)
        return Response()

    monkeypatch.setattr("requests.Session.post", post)
    destination = tmp_path / "audio.bin"
    client.download_audio("track.wav", 3, str(destination))

    assert captured["headers"]["X-Parsec-Worker-Id"] == "gpu-host-1"
    assert destination.read_bytes() == b"abc"
