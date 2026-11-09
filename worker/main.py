import logging
import signal

from .config import Settings
from .engine import ClapEngine
from .gateway_client import GatewayClient
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
    worker.run()


if __name__ == "__main__":
    main()
