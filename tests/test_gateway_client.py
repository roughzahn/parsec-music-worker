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
    assert captured["kwargs"]["json"] == {"status": "STARTING"}
