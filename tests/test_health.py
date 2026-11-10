import json
import threading
from urllib.error import HTTPError
from urllib.request import urlopen

from worker.health import WorkerHealthServer


def test_health_endpoint_reflects_model_and_registration_readiness():
    state = {"ready": False}
    server = WorkerHealthServer(("127.0.0.1", 0), lambda: state["ready"])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        url = f"http://127.0.0.1:{server.server_port}/health"
        try:
            urlopen(url, timeout=2)
            assert False, "health endpoint must be unavailable before readiness"
        except HTTPError as error:
            assert error.code == 503
            assert json.loads(error.read()) == {"status": "DOWN", "ready": False}

        state["ready"] = True
        with urlopen(url, timeout=2) as response:
            assert response.status == 200
            assert json.loads(response.read()) == {"status": "UP", "ready": True}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
