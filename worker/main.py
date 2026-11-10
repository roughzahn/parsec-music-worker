import logging
import signal
import threading

from .config import Settings
from .engine import ClapEngine
from .gateway_client import GatewayClient
from .health import WorkerHealthServer
from .runner import Worker


def main():
    logging.basicConfig(level=logging.INFO)
    settings = Settings.from_env()
    client = GatewayClient(settings.gateway_url, settings.token, settings.request_timeout)
    engine = ClapEngine(settings.worker_type, settings.model_path, settings.model_id)
    worker = Worker(settings, client, engine)

    def shutdown(signum, frame):
        print(f"Received signal {signum}; stopping after current job", flush=True)
        worker.stop()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    health_server = WorkerHealthServer(("0.0.0.0", settings.health_port), worker.is_ready)
    health_thread = threading.Thread(target=health_server.serve_forever, name="health-server", daemon=True)
    health_thread.start()
    try:
        worker.run()
    finally:
        worker.stop()
        health_server.shutdown()
        health_server.server_close()
        health_thread.join(timeout=2)


if __name__ == "__main__":
    main()
