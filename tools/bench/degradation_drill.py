#!/usr/bin/env python3
"""T013 degradation drills — run jevalaya with 3/2/1/0 backends live and
verify routing degrades per DESIGN.md §Backend availability / wire contract.

Each configuration boots a fresh service against a mutated bench config, then
runs the same probe battery. The oracle asserts:

- available backends answer; explicitly requested absent backends return
  409 backend_unavailable (never a fabricated answer);
- auto mode excludes absent backends and retains the validated local answer
  with escalated/degraded/escalation_error when Jev escalation is impossible
  (configured-but-down) — vs a clean local answer when Jev is unconfigured
  (the escalation gate does not even fire);
- a single remaining backend is a validated passthrough; zero usable
  backends is 503 not_ready, never an invented result.

Jev is always a local mock (or a dead port) — TYPESAFE_API_KEY is popped by
``Service.start`` and the config points ``api_key_env`` at JEV_DRILL_KEY.

Usage::

    python3 tools/bench/degradation_drill.py \
        --config tools/bench/jevalaya.bench.toml \
        --output /tmp/t013-degradation.json
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_local_backends import PHRASE, QUESTION, Service  # noqa: E402

ROOT = HERE.parents[1]
DEFAULT_BINARY = ROOT / "target/release/jevalaya"
DEFAULT_BASE_URL = "http://127.0.0.1:8767"
DEFAULT_TOKEN = "drill-local"
LONG_STATE = PHRASE * 20  # ~220 rendered tokens — over the ANE L96 budget


def toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return json.dumps(v)
    return json.dumps(v) if isinstance(v, list) else repr(v)


def dump_toml(cfg: dict[str, dict[str, Any]]) -> str:
    return "\n\n".join(
        f"[{section}]\n"
        + "\n".join(f"{k} = {toml_value(v)}" for k, v in fields.items())
        for section, fields in cfg.items()
    ) + "\n"


def make_config(
    base: dict, variant: dict[str, Any], out: Path, mock_url: str
) -> Path:
    import copy

    cfg = copy.deepcopy(base)
    cfg["ane"]["enabled"] = variant["ane"]
    cfg["mlx"]["enabled"] = variant["mlx"]
    jev = cfg.setdefault("jev", {})
    if variant["jev"] is None:
        jev["enabled"] = False
    else:
        jev["enabled"] = True
        jev["base_url"] = mock_url if variant["jev"] == "mock" else dead_port()
        jev["api_key_env"] = "JEV_DRILL_KEY"
        # margin 1.5 > any real top-two margin — every local auto answer
        # deterministically triggers escalation, which is exactly what the
        # degraded-path probes need.
        jev["margin_threshold"] = 1.5
    obs = cfg.setdefault("observability", {})
    obs["sink"] = f"jsonl:{out.with_suffix('')}-events.jsonl"
    text = dump_toml(cfg)
    cfg_path = out.with_suffix(".toml")
    cfg_path.write_text(text)
    return cfg_path


class MockJev(http.server.BaseHTTPRequestHandler):
    hits: list[dict] = []

    def log_message(self, *_: Any) -> None:
        pass

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        MockJev.hits.append(body)
        answers: dict[str, Any] = {}
        for qid, q in (body.get("questions") or {}).items():
            crit = q.get("criteria") or {}
            labels = list(crit) if isinstance(crit, dict) else list(crit or [])
            if q.get("type") == "choice" and labels:
                sel = labels[0]
                answers[qid] = {
                    "choice": sel,
                    "probabilities": {
                        l: (0.7 if l == sel else 0.3 / max(1, len(labels) - 1))
                        for l in labels
                    },
                    "confidence": 0.7,
                }
        payload = {
            "model": body.get("model", "jev-mock"),
            "answers": answers,
            "usage": {"input_tokens": 10, "output_tokens": 10},
        }
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def start_mock() -> tuple[http.server.HTTPServer, str]:
    server = http.server.HTTPServer(("127.0.0.1", 0), MockJev)
    MockJev.hits = []
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}"


def dead_port() -> str:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return f"http://127.0.0.1:{port}"


def call(
    client: httpx.Client,
    base_url: str,
    body: dict[str, Any],
    token: str,
    timeout: float = 90.0,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        r = client.post(
            f"{base_url}/predict",
            json=body,
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout,
        )
    except httpx.HTTPError as e:
        return {"status": None, "error": f"transport:{type(e).__name__}"}
    rec: dict[str, Any] = {
        "status": r.status_code,
        "client_ms": round((time.perf_counter() - t0) * 1000, 1),
    }
    try:
        b = r.json()
    except ValueError:
        rec["error"] = r.text[:200]
        return rec
    rt = b.get("routing") or {}
    ans = (b.get("answers") or {}).get("topic") or {}
    rec.update(
        backend_answered=rt.get("backend"),
        reason=(rt.get("reason") or "")[:80],
        escalated=rt.get("escalated"),
        degraded=rt.get("degraded"),
        escalation_error=rt.get("escalation_error"),
        jev_trigger=rt.get("jev_trigger"),
        fallback=rt.get("fallback"),
        fallback_from=rt.get("fallback_from"),
        fallback_reason=rt.get("fallback_reason"),
        choice=ans.get("choice"),
        compare=b.get("compare"),
        error=(b.get("error") or {}).get("code") or "",
    )
    return rec


def body(state: str, backend: str | None = None, compare: bool = False) -> dict:
    b: dict[str, Any] = {"state": state, "questions": QUESTION,
                         "model": "multilingual"}
    if backend:
        b["backend"] = backend
    if compare:
        b["compare"] = True
    return b


def check(name: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"probe": name, "ok": bool(ok), "detail": detail}


def drill(
    client: httpx.Client,
    base_url: str,
    token: str,
    variant: dict[str, Any],
    health: dict[str, Any],
) -> list[dict[str, Any]]:
    """One probe battery + expected outcomes for a backend configuration."""
    available = {
        k
        for k, v in (health.get("backends") or {}).items()
        if isinstance(v, dict) and v.get("available")
    }
    # Health availability is key-presence for Jev — a dead endpoint still
    # reports available until dispatch fails, which is what we're testing.
    jev_live = "jev" in available and variant["jev"] != "down"
    jev_configured = variant["jev"] is not None
    mlx_live = "mlx" in available
    ane_live = "ane" in available
    out: list[dict[str, Any]] = []

    out.append(check(
        "health.available",
        available == {k for k, on in
                      {"ane": variant["ane"], "mlx": variant["mlx"],
                       "jev": jev_configured}.items() if on},
        f"available={sorted(available)}",
    ))

    # Explicit pins — the DESIGN §146 contract: absent backend => 409.
    for bname, live in (("ane", ane_live), ("mlx", mlx_live), ("jev", jev_live)):
        if bname == "jev" and variant["jev"] == "down":
            continue  # dead-remote explicit asserted separately below
        rec = call(client, base_url, body(PHRASE, bname), token)
        if live and bname == "ane":
            # Explicit ane may ride the one permitted ANE→MLX retry on
            # capacity/shape/not-ready — legal only when honestly marked.
            ok = rec["status"] == 200 and (
                rec["backend_answered"] == "ane"
                or (rec["backend_answered"] == "mlx"
                    and rec["fallback"] is True
                    and rec["fallback_from"] == "ane"
                    and rec.get("fallback_reason"))
            )
            out.append(check(
                "explicit.ane",
                ok,
                f"{rec['status']} answered={rec['backend_answered']} "
                f"fallback={rec['fallback']} from={rec['fallback_from']} "
                f"why={rec.get('fallback_reason')}",
            ))
        elif live:
            out.append(check(
                f"explicit.{bname}",
                rec["status"] == 200 and rec["backend_answered"] == bname
                and rec.get("choice"),
                f"{rec['status']} answered={rec['backend_answered']}",
            ))
        else:
            out.append(check(
                f"explicit.{bname}",
                rec["status"] == 409 and rec["error"] == "backend_unavailable",
                f"{rec['status']} code={rec['error']}",
            ))

    # Forced-margin auto: outcome depends on whether Jev is configured/live.
    rec = call(client, base_url, body(PHRASE), token)
    if not (mlx_live or ane_live or jev_live):
        exp = rec["status"] == 503
        detail = f"{rec['status']} code={rec['error']}"
    elif jev_live and not (mlx_live or ane_live):
        # Only Jev remains: auto routes straight to it — the escalation
        # gate needs a local primary answer, so `escalated` stays false.
        exp = rec["status"] == 200 and rec["backend_answered"] == "jev" and rec.get("choice")
        detail = (f"{rec['status']} answered={rec['backend_answered']} "
                  f"(direct jev passthrough, esc={rec['escalated']})")
    elif jev_live:
        exp = (rec["status"] == 200 and rec["backend_answered"] == "jev"
               and rec["escalated"] is True and rec.get("choice"))
        detail = (f"{rec['status']} answered={rec['backend_answered']} "
                  f"esc={rec['escalated']} trig={rec.get('jev_trigger')}")
    elif jev_configured:  # configured but dead: retain local + mark honestly
        exp = (rec["status"] == 200 and rec["backend_answered"] in ("ane", "mlx")
               and rec["degraded"] is True and rec.get("escalation_error")
               and rec.get("choice"))
        detail = (f"{rec['status']} answered={rec['backend_answered']} "
                  f"degraded={rec['degraded']} err={rec.get('escalation_error')}")
    else:  # Jev unconfigured: gate skipped — clean local answer, no pretend escalation
        exp = (rec["status"] == 200 and rec["backend_answered"] in ("ane", "mlx")
               and not rec["escalated"] and rec["degraded"] is not True
               and rec.get("choice"))
        detail = (f"{rec['status']} answered={rec['backend_answered']} "
                  f"esc={rec['escalated']} degraded={rec['degraded']}")
    out.append(check("auto.short.forced_margin", exp, detail))

    rec = call(client, base_url, body(LONG_STATE), token)
    if not (mlx_live or ane_live or jev_live):
        exp, detail = rec["status"] == 503, f"{rec['status']} code={rec['error']}"
    elif jev_live:
        exp = rec["status"] == 200 and rec["backend_answered"] == "jev"
        detail = f"{rec['status']} answered={rec['backend_answered']}"
    elif mlx_live or ane_live:
        exp = rec["status"] == 200 and rec["backend_answered"] in ("ane", "mlx")
        detail = f"{rec['status']} answered={rec['backend_answered']}"
    else:
        exp, detail = False, f"{rec['status']}"
    out.append(check("auto.long", exp, detail))

    # Explicit dead remote: configured-but-unreachable Jev => upstream failure.
    if variant["jev"] == "down":
        rec = call(client, base_url, body(PHRASE, "jev"), token)
        out.append(check(
            "explicit.jev_down",
            rec["status"] in (502, 503) and rec["error"],
            f"{rec['status']} code={rec['error']}",
        ))

    # Partial compare: unavailable members carry errors; live ones answer.
    if (mlx_live or ane_live) and jev_configured:
        rec = call(client, base_url, body(PHRASE, compare=True), token)
        cmp_ = rec.get("compare") or {}
        jev_entry = (cmp_.get("jev") or {})
        ok = (
            rec["status"] == 200
            and (cmp_.get("mlx") or {}).get("answers") is not None
            and (jev_entry.get("error") is not None or jev_entry.get("answers") is not None)
        )
        out.append(check(
            "compare.partial",
            ok,
            f"status={rec['status']} compare_keys={sorted(cmp_)} "
            f"jev_err={(jev_entry.get('error') or {}) and 'set'}",
        ))

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--token", default=DEFAULT_TOKEN)
    ap.add_argument("--output", type=Path, default=Path("/tmp/t013-degradation.json"))
    ap.add_argument("--startup-timeout", type=float, default=60.0)
    ap.add_argument("--ane-timeout", type=float, default=150.0)
    args = ap.parse_args()

    import tomllib

    base = tomllib.loads(args.config.read_text())
    variants = [
        ("all3",        {"ane": True,  "mlx": True,  "jev": "mock"}),
        ("local_pair",  {"ane": True,  "mlx": True,  "jev": None}),
        ("remote_pair", {"ane": False, "mlx": True,  "jev": "mock"}),
        ("mlx_only",    {"ane": False, "mlx": True,  "jev": None}),
        ("jev_only",    {"ane": False, "mlx": False, "jev": "mock"}),
        ("jev_down",    {"ane": False, "mlx": True,  "jev": "down"}),
        ("none",        {"ane": False, "mlx": False, "jev": None}),
    ]

    mock, mock_url = start_mock()
    workdir = args.output.parent / "t013"
    workdir.mkdir(parents=True, exist_ok=True)
    os.environ["JEV_DRILL_KEY"] = "drill-only-not-a-real-key"

    report: dict[str, Any] = {"configs": {}, "mock_url": mock_url}
    try:
        for name, variant in variants:
            cfg_path = make_config(base, variant, workdir / name, mock_url)
            log = workdir / f"{name}.log"
            try:
                with Service(args.binary, cfg_path, args.base_url,
                             args.token, log) as svc:
                    health, boot_ms = svc.wait_health(args.startup_timeout)
                    if variant["ane"]:
                        # Explicit-ane probe needs residency, not warming.
                        health, _ = svc.wait_ane_ready(args.ane_timeout)
                    checks = drill(svc.client, args.base_url, args.token,
                                   variant, health)
                    report["configs"][name] = {
                        "boot_ms": boot_ms,
                        "available": sorted(
                            k
                            for k, v in (health.get("backends") or {}).items()
                            if isinstance(v, dict) and v.get("available")
                        ),
                        "checks": checks,
                        "ok": all(c["ok"] for c in checks),
                    }
            except Exception as e:  # boot refusal is itself evidence for `none`
                report["configs"][name] = {
                    "ok": name == "none",
                    "checks": [],
                    "boot_error": str(e)[-300:],
                }
    finally:
        mock.shutdown()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))

    rows = ["| config | backends up | checks | result |", "|---|---|---|---|"]
    for name, res in report["configs"].items():
        n = len(res.get("checks", []))
        passed = sum(c["ok"] for c in res.get("checks", []))
        label = "(boot refused)" if "boot_error" in res else \
            ", ".join(res.get("available") or ["none"])
        rows.append(
            f"| {name} | {label} "
            f"| {passed}/{n} | {'PASS' if res['ok'] else 'FAIL'} |"
        )
    print("\n".join(rows))
    bad = [c["probe"] for r in report["configs"].values()
           for c in r.get("checks", []) if not c["ok"]]
    if bad:
        print("failing:", *bad, sep="\n  ")
    return 0 if all(r["ok"] for r in report["configs"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
