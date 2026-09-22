"""Demo origin server: serves tools/demo/sorter/ statically, serves dataset
rows at GET /items, and proxies POST /predict to the live jevalaya server.

Why a proxy: browsers enforce CORS, and the router deliberately has no CORS
headers — the demo stays a pure consumer with zero router changes. Latency
honesty is preserved: the overlay shows the server-measured `latency_ms`
from the routing receipt (inference), plus page-measured round trip.

Usage: serve_demo.py [--demo-port 8000] [--jevalaya http://127.0.0.1:8767]
Then open http://127.0.0.1:8000/index.html?token=<bearer>.
"""

import argparse
import http.server
import json
import os
import urllib.request


class Handler(http.server.SimpleHTTPRequestHandler):
    jevalaya = "http://127.0.0.1:8767"
    jevalaya_jev = ""  # optional separate upstream for backend=jev (golden beat)
    jev_token = ""     # env JEVALAYA_JEV_TOKEN; overrides bearer on jev upstream
    items_path = None

    def log_message(self, *args):
        pass

    def do_GET(self):
        # GET /items?limit=N&offset=M — dataset rows for sorter demos.
        if self.path.startswith("/items"):
            import urllib.parse

            qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            limit = int(qs.get("limit", ["40"])[0])
            offset = int(qs.get("offset", ["0"])[0])
            rows = []
            with open(Handler.items_path, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i < offset:
                        continue
                    if len(rows) >= limit:
                        break
                    if line.strip():
                        rows.append(json.loads(line))
            body = json.dumps({"items": rows}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        return super().do_GET()

    def do_POST(self):
        if self.path != "/predict":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        upstream_base = self.jevalaya
        auth = self.headers.get("Authorization", "")
        if self.jevalaya_jev:
            try:
                if json.loads(raw.decode("utf-8")).get("backend") == "jev":
                    upstream_base = self.jevalaya_jev
                    if self.jev_token:
                        auth = "Bearer " + self.jev_token
            except (ValueError, AttributeError):
                pass
        upstream = urllib.request.Request(
            upstream_base + "/predict",
            data=raw,
            headers={
                "Content-Type": "application/json",
                "Authorization": auth,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(upstream, timeout=90) as r:
                body, status = r.read(), r.status
        except urllib.error.HTTPError as e:
            body, status = e.read(), e.code
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo-port", type=int, default=8000)
    ap.add_argument("--jevalaya", default="http://127.0.0.1:8767")
    ap.add_argument("--jevalaya-jev", default="",
                    help="separate upstream for backend=jev (golden beat); "
                         "unset = same as --jevalaya")
    args = ap.parse_args()
    Handler.jevalaya = args.jevalaya
    Handler.jevalaya_jev = args.jevalaya_jev
    Handler.jev_token = os.environ.get("JEVALAYA_JEV_TOKEN", "")
    Handler.items_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "bench", "datasets", "agnews_test.jsonl",
    )
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sorter"))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.demo_port), Handler)
    print(f"demo at http://127.0.0.1:{args.demo_port}/index.html", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
