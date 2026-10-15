#!/usr/bin/env python3
"""Serve the demo storefront.

A plain stdlib HTTP server is enough and keeps the demo self-contained — the
point is that a *real* browser is driving a *real* HTTP site, not that the site
runs on a particular stack.

    python3 demo/serve.py --port 8099
"""

from __future__ import annotations

import argparse
import functools
import http.server
import socketserver
import threading
from pathlib import Path

WEBAPP = Path(__file__).resolve().parent / "webapp"

#: URL path -> file under webapp/. The demo needs two real pages, because
#: "does your harness handle a normal site" and "does it handle a surface that
#: gives up no text" are different questions and deserve different pages.
ROUTES = {
    "/": "/templates/shell.html",
    "/index.html": "/templates/shell.html",
    "/gateway": "/templates/gateway.html",
    "/gateway/": "/templates/gateway.html",
}


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        path = self.path.split("?", 1)[0]
        if path in ROUTES:
            self.path = ROUTES[path]
        super().do_GET()

    def log_message(self, *_args) -> None:
        """Quiet: the agent's own telemetry is the interesting log."""


def serve(port: int = 8099, host: str = "127.0.0.1") -> socketserver.TCPServer:
    handler = functools.partial(Handler, directory=str(WEBAPP))
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer((host, port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd


def start_background(port: int = 8099) -> tuple[socketserver.TCPServer, str]:
    httpd = serve(port)
    return httpd, f"http://127.0.0.1:{port}/"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve the demo storefront")
    parser.add_argument("--port", type=int, default=8099)
    args = parser.parse_args()
    httpd, url = start_background(args.port)
    print(f"serving {WEBAPP} on {url}  (ctrl-c to stop)")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        httpd.shutdown()
