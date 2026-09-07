"""Development server for the web client — no FastAPI, no numpy, stdlib only.

    python portfolio/web/dev/serve.py --port 8899

Serves /static/* from portfolio/web, index.html at / (and any non-API path),
and answers the read-only API from the mock JSON files next to this script.
"""
import argparse, json, pathlib, sys, time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

HERE = pathlib.Path(__file__).resolve().parent
WEB = HERE.parent
SNAPSHOT = json.loads((HERE / "mock-snapshot.json").read_text(encoding="utf-8"))
SETTINGS = json.loads((HERE / "mock-settings.json").read_text(encoding="utf-8"))


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(WEB), **kw)

    def log_message(self, fmt, *args):  # quiet unless it is an error
        if args and str(args[1]).startswith(("4", "5")):
            sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    def _json(self, body, status=200):
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _api(self, method):
        path = self.path.split("?", 1)[0]
        if path == "/api/snapshot" and method == "GET":
            return self._json(SNAPSHOT)
        if path == "/api/settings":
            if method == "PUT":
                n = int(self.headers.get("Content-Length") or 0)
                SETTINGS.update({k: v for k, v in json.loads(self.rfile.read(n) or b"{}").items() if v is not None})
            return self._json(SETTINGS)
        if path == "/api/health":
            return self._json({"status": "ok", "version": SETTINGS["version"], "mode": SETTINGS["mode"], "provider": SETTINGS["provider"]})
        if path == "/api/benchmarks":
            return self._json(SNAPSHOT["benchmarks"])
        if path == "/api/refresh" and method == "POST":
            time.sleep(0.8)  # long enough to see the spinner and the top-bar progress line
            return self._json({"refreshed": True, "elapsed_ms": 800})
        return self._json({"error": {"code": "not_found", "message": "No mock for %s %s" % (method, path)}}, 404)

    def do_GET(self):
        if self.path.startswith("/api/"):
            return self._api("GET")
        if self.path.startswith("/static/"):
            self.path = self.path[len("/static"):]
            return super().do_GET()
        self.path = "/index.html"  # SPA: every non-API path serves the shell
        return super().do_GET()

    def do_POST(self): return self._api("POST")
    def do_PUT(self): return self._api("PUT")
    def do_PATCH(self): return self._api("PATCH")
    def do_DELETE(self): return self._api("DELETE")

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    print("serving %s on http://%s:%d/" % (WEB, args.host, args.port))
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
