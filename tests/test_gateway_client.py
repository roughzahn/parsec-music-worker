from worker.gateway_client import GatewayClient


def test_gateway_client_auth_and_routes():
    client = GatewayClient("http://gateway/", "secret")
    assert client._url("/internal/v1/analysis/workers/cpu-1/jobs/claim") == (
        "http://gateway/internal/v1/analysis/workers/cpu-1/jobs/claim"
    )
    assert client.session.headers["Authorization"] == "Bearer secret"
    assert client.session.headers["User-Agent"].startswith("parsec-music-worker/")
