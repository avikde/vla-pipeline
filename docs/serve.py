#!/usr/bin/env python3
"""Local dev server with COOP/COEP headers required for MuJoCo WASM (SharedArrayBuffer)."""

import http.server
import sys


class CORSHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "credentialless")
        super().end_headers()


port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
print(f"Serving on http://localhost:{port} (with COOP/COEP headers)")
http.server.HTTPServer(("", port), CORSHandler).serve_forever()
