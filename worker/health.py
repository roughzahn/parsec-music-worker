import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class WorkerHealthServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, is_ready):
        self.is_ready = is_ready
        super().__init__(address, self._handler())

    def _handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path != "/health":
                    self.send_error(404)
                    return
                ready = server.is_ready()
                payload = json.dumps({"status": "UP" if ready else "DOWN", "ready": ready}).encode()
                self.send_response(200 if ready else 503)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format, *args):
                return

        return Handler
