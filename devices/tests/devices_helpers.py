"""Shared test helpers: a fusion that accepts device ids, a local HTTP server."""

import threading
from contextlib import contextmanager
from http.server import ThreadingHTTPServer

from polaris_devices.paths import ensure_repo_paths

ensure_repo_paths()

from fusion_carte import CarteFusion  # noqa: E402


class CarteAvecIds(CarteFusion):
    """What interface 7 asks of the fusion, emulated for these tests: 'drone:<id>' and
    'rover:<id>' accepted, the role is the part before the colon. Records every source."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recu = []

    def recevoir(self, msg):
        self.recu.append(msg['source'])
        return super().recevoir({**msg, 'source': msg['source'].split(':')[0]})


@contextmanager
def serve(handler, port):
    """Run an HTTP handler on 127.0.0.1:port (ports 8170-8179 are reserved for these tests)."""
    server = ThreadingHTTPServer(('127.0.0.1', port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{port}'
    finally:
        server.shutdown()
        server.server_close()
