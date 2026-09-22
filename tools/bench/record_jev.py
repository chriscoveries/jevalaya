"""Zero-spend Jev wire recorder (T029).

Listens on loopback, appends each raw request (headers + body bytes) to a
file, and answers 200 with a canned payload. Point a jevalaya server at it
via [jev] base_url to capture the EXACT bytes backend-jev sends without
spending a single TypeSafe call.

Usage: record_jev.py --port 9765 --out /tmp/jev-wire.log
"""

import argparse
import http.server
import json


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        auth = self.headers.get("Authorization", "")
        # Redact bearer value length only — never persist secrets.
        safe_auth = "Bearer <redacted>" if auth.startswith("Bearer ") else auth
        with open(self.server.out_path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "path": self.path,
                        "authorization": safe_auth,
                        "content_type": self.headers.get("Content-Type", ""),
                        "body": json.loads(raw.decode("utf-8")),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        body = json.dumps({"model": "jev-recorder", "answers": {}, "usage": {}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9765)
    ap.add_argument("--out", default="/tmp/jev-wire.log")
    args = ap.parse_args()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.out_path = args.out
    print(f"recording to {args.out}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
