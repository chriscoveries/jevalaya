#!/usr/bin/env python3
"""T015 consolidated latency sweep — per-backend × context-length × warm/cold
through the running service, with p50/p95/p99 and verdicts against the
DESIGN.md §Latency budgets:

- sub-100 ms end-to-end for a warm one-question request (client p95, warm);
- warm MLX execution p95 <= 20 ms, ANE p95 <= 10 ms (server-side
  ``routing.latency_ms``, short/fitting contexts only);
- router overhead (tokenize+route+dispatch, no model execution) p95 <= 5 ms,
  p99 <= 10 ms — proxied externally by ``/route`` client latency plus the
  predict client−server delta;
- cold load/preload reported separately, never folded into warm percentiles.

Jev is never exercised: the run aborts if the health snapshot exposes an
available Jev lane (same discipline as run_local_backends). Explicit ``ane``
pins over the L96 bundle ceiling are recorded as their own outcome class —
a contract rejection/fallback is honest degradation evidence, not latency.

Usage::

    python3 tools/bench/latency_sweep.py \
        --config tools/bench/jevalaya.bench.toml \
        --output tools/bench/results/t015-latency.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_local_backends import (  # noqa: E402
    DEFAULT_TARGETS,
    PHRASE,
    Service,
    assert_no_jev,
    call_predict,
    discover_contexts,
    percentile,
    request_body,
)

ROOT = HERE.parents[1]
DEFAULT_BINARY = ROOT / "target/release/jevalaya"
DEFAULT_BASE_URL = "http://127.0.0.1:8767"
DEFAULT_TOKEN = "bench-local"

# DESIGN.md §Latency budgets
BUDGET_E2E_MS = 100.0        # warm one-question request, end to end
BUDGET_MLX_P95_MS = 20.0     # warm MLX execution, server-side
BUDGET_ANE_P95_MS = 10.0     # warm ANE execution where the model meets it
BUDGET_OVERHEAD_P95_MS = 5.0
BUDGET_OVERHEAD_P99_MS = 10.0
ANE_TOKEN_CEILING = 96       # the shipped bundle's L bucket


def pstats(values: list[float]) -> dict[str, Any]:
    """p50/p95/p99 + min/max for one latency sample set."""
    if not values:
        return {"n": 0}
    ordered = sorted(float(v) for v in values)
    return {
        "n": len(ordered),
        "min_ms": round(ordered[0], 3),
        "p50_ms": percentile(ordered, 0.50),
        "p95_ms": percentile(ordered, 0.95),
        "p99_ms": percentile(ordered, 0.99),
        "max_ms": round(ordered[-1], 3),
        "mean_ms": round(sum(ordered) / len(ordered), 3),
    }


def cell(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Latency stats over the rows where the requested backend actually
    answered (fallback/rejection rows are counted, not folded in)."""
    ok = [r for r in records
          if r.get("status") == 200 and not r.get("error")]
    direct = [r for r in ok if not r.get("fallback")]
    fell_back = [r for r in ok if r.get("fallback")]
    rejected = [r for r in records if r.get("status") != 200]
    return {
        "n_calls": len(records),
        "n_answered": len(ok),
        "n_fallback": len(fell_back),
        "n_rejected": len(rejected),
        "reject_statuses": sorted({r.get("status") for r in rejected}),
        "fallback_reasons": sorted(
            {r.get("fallback_reason") or "" for r in fell_back} - {""}
        ),
        "client_ms": pstats([r["client_ms"] for r in direct]),
        "server_ms": pstats(
            [r["server_ms"] for r in direct
             if isinstance(r.get("server_ms"), (int, float))]
        ),
    }


def sweep(
    service: Service,
    base_url: str,
    token: str,
    cases: list[Any],
    backends: tuple[str, ...],
    repeats: int,
    request_timeout: float,
) -> list[dict[str, Any]]:
    """Every context × every pinned backend × `repeats`, sequential."""
    records: list[dict[str, Any]] = []
    for case in cases:
        for backend in backends:
            for i in range(repeats):
                rec = call_predict(
                    service.client, base_url, token,
                    case.state, backend,
                    f"t015-{backend}-{case.name}-{i}", request_timeout,
                )
                rec.update(backend_pin=backend, context_case=case.name,
                           token_count=case.token_count, phase="warm")
                records.append(rec)
    return records


def route_overhead(
    service: Service,
    base_url: str,
    token: str,
    cases: list[Any],
    repeats: int,
) -> list[dict[str, Any]]:
    """/route calls (auto backend): validate+tokenize+route with zero model
    execution — the closest external proxy for the §475 router-overhead
    budget."""
    records = []
    for case in cases:
        for i in range(repeats):
            t0 = time.perf_counter()
            r = service.client.post(
                f"{base_url}/route",
                json=request_body(case.state, "auto", f"t015-route-{case.name}-{i}"),
                headers={"Authorization": f"Bearer {token}"},
                timeout=15.0,
            )
            records.append({
                "context_case": case.name,
                "token_count": case.token_count,
                "status": r.status_code,
                "client_ms": round((time.perf_counter() - t0) * 1000, 3),
            })
    return records


def verdicts(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Each budget line → PASS/FAIL against measured percentiles."""
    cells = report["cells"]
    out = []

    short = [c for c in cells if c["token_count"] <= ANE_TOKEN_CEILING]
    e2e_p95 = max(
        (c["client_ms"].get("p95_ms") or math.inf)
        for c in short if c["backend_pin"] == "auto"
    ) if any(c["backend_pin"] == "auto" for c in short) else math.inf
    out.append({
        "budget": "warm one-question end-to-end p95 <= 100 ms (auto, <=96tok)",
        "measured_ms": None if math.isinf(e2e_p95) else e2e_p95,
        "ok": e2e_p95 <= BUDGET_E2E_MS,
    })

    for pin, cap in (("mlx", BUDGET_MLX_P95_MS), ("ane", BUDGET_ANE_P95_MS)):
        vals = [c["server_ms"].get("p95_ms") for c in short
                if c["backend_pin"] == pin and c["server_ms"].get("n")]
        worst = max(vals) if vals else None
        out.append({
            "budget": f"warm {pin} execution p95 <= {cap} ms (<=96tok, server)",
            "measured_ms": worst,
            "ok": worst is not None and worst <= cap,
        })

    oh = report["route_overhead"]["client_ms"]
    out.append({
        "budget": "router overhead p95 <= 5 ms / p99 <= 10 ms (/route proxy)",
        "measured_ms": [oh.get("p95_ms"), oh.get("p99_ms")],
        "ok": bool(oh.get("n")) and oh["p95_ms"] <= BUDGET_OVERHEAD_P95_MS
        and oh["p99_ms"] <= BUDGET_OVERHEAD_P99_MS,
    })
    return out


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# jevalaya consolidated latency sweep (T015)",
        "",
        "Per-backend × context-length × warm/cold through the running service. "
        "Jev disabled throughout — no paid calls.",
        "",
        "## Budget verdicts",
        "",
        "| budget | measured | verdict |",
        "|---|---:|---|",
    ]
    for v in report["verdicts"]:
        m = v["measured_ms"]
        text = "/".join(f"{x} ms" for x in m) if isinstance(m, list) else (
            f"{m} ms" if m is not None else "n/a")
        lines.append(f"| {v['budget']} | {text} | {'PASS' if v['ok'] else 'FAIL'} |")

    lines += [
        "",
        "## Warm cells — client end-to-end p50 / p95 / p99",
        "",
        "| tokens | pin | answered | client p50 | p95 | p99 | server p95 | fallbacks | rejects |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for c in sorted(report["cells"],
                    key=lambda r: (r["token_count"], r["backend_pin"])):
        cm, sm = c["client_ms"], c["server_ms"]
        lines.append(
            f"| {c['token_count']} | {c['backend_pin']} "
            f"| {','.join(c.get('backend_answered') or ['-'])} "
            f"| {cm.get('p50_ms')} | {cm.get('p95_ms')} | {cm.get('p99_ms')} "
            f"| {sm.get('p95_ms')} | {c['n_fallback']} | {c['n_rejected']} |"
        )

    cold = report["cold"]
    lines += [
        "",
        "## Cold (reported separately, never in warm percentiles)",
        "",
        f"- First MLX predict after boot: `{cold['mlx_first'].get('client_ms')} ms` client"
        f" / `{cold['mlx_first'].get('server_ms')} ms` server.",
        f"- First ANE pin (served by MLX while warming): `{cold['ane_first'].get('client_ms')} ms` client"
        f" / `{cold['ane_first'].get('server_ms')} ms` server.",
        f"- ANE resident after `{cold.get('ane_resident_ms')} ms` from process start.",
        f"- ANE **dispatch**-ready after `{report['service'].get('ane_dispatch_ready_ms')} ms` "
        "(health-ok ≠ dispatch-ready — the gap is where `not_ready` fallbacks land).",
        f"- First ANE-resident predict: `{cold.get('ane_warm_first', {}).get('client_ms')} ms` client"
        f" / `{cold.get('ane_warm_first', {}).get('server_ms')} ms` server.",
        "",
        "## Router overhead proxy (`/route`, no model execution)",
        "",
        f"client p50 `{report['route_overhead']['client_ms'].get('p50_ms')}` ms, "
        f"p95 `{report['route_overhead']['client_ms'].get('p95_ms')}` ms, "
        f"p99 `{report['route_overhead']['client_ms'].get('p99_ms')}` ms "
        f"(n={report['route_overhead']['client_ms'].get('n')})",
        "",
        "## Metadata",
        "",
        f"- `{report['service']}`",
        f"- warm repeats per cell: `{report['method']['warm_repeats']}`; "
        f"calls total: `{report['counts']['total']}`",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--token", default=DEFAULT_TOKEN)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--warm-repeats", type=int, default=30)
    ap.add_argument("--targets", type=str, default=",".join(map(str, DEFAULT_TARGETS)))
    ap.add_argument("--max-repetitions", type=int, default=100)
    ap.add_argument("--startup-timeout", type=float, default=60.0)
    ap.add_argument("--ane-timeout", type=float, default=180.0)
    ap.add_argument("--request-timeout", type=float, default=120.0)
    args = ap.parse_args()

    targets = tuple(int(t) for t in args.targets.split(",") if t.strip())
    log_dir = args.output.parent / "t015-logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # ---- cold boot: first-request latencies, then ANE residency ---------
    cold: dict[str, Any] = {}
    with Service(args.binary, args.config, args.base_url, args.token,
                 log_dir / "cold.log") as svc:
        health, _ = svc.wait_health(args.startup_timeout)
        assert_no_jev(health)
        cold["mlx_first"] = call_predict(
            svc.client, args.base_url, args.token, PHRASE, "mlx",
            "t015-cold-mlx", args.request_timeout,
        )
        cold["ane_first"] = call_predict(
            svc.client, args.base_url, args.token, PHRASE, "ane",
            "t015-cold-ane", args.request_timeout,
        )
        _, ready_ms = svc.wait_ane_ready(args.ane_timeout)
        cold["ane_resident_ms"] = ready_ms
        cold["ane_warm_first"] = call_predict(
            svc.client, args.base_url, args.token, PHRASE, "ane",
            "t015-ane-warm-first", args.request_timeout,
        )

    # ---- warm suite ------------------------------------------------------
    with Service(args.binary, args.config, args.base_url, args.token,
                 log_dir / "warm.log") as svc:
        health, boot_ms = svc.wait_health(args.startup_timeout)
        assert_no_jev(health)
        ready_health, _ = svc.wait_ane_ready(args.ane_timeout)
        # wait_ane_ready gates on the health flag, which flips long before
        # dispatch is actually served by ANE — observed: ~90 consecutive
        # post-"ready" pins still fell back not_ready→mlx. Poll an explicit
        # ANE pin until it genuinely answers as ane (or give up and let the
        # cells honestly record the fallback mass).
        dispatch_ready_ms = None
        deadline = time.perf_counter() + args.ane_timeout
        for i in range(int(args.ane_timeout // 2)):
            rec = call_predict(svc.client, args.base_url, args.token, PHRASE,
                               "ane", f"t015-prewarm-ane-{i}",
                               args.request_timeout)
            if rec.get("backend_answered") == "ane" and not rec.get("fallback"):
                dispatch_ready_ms = round(
                    (time.perf_counter() - svc.started_at) * 1000, 1)
                break
            if time.perf_counter() > deadline:
                break
            time.sleep(2.0)
        call_predict(svc.client, args.base_url, args.token, PHRASE,
                     "mlx", "t015-prewarm-mlx", args.request_timeout)
        cases = discover_contexts(svc.client, args.base_url, args.token,
                                  targets, args.max_repetitions)
        records = sweep(svc, args.base_url, args.token, cases,
                        ("ane", "mlx", "auto"), args.warm_repeats,
                        args.request_timeout)
        overhead = route_overhead(svc, args.base_url, args.token,
                                  cases, args.warm_repeats)

    cells = []
    for case in cases:
        for pin in ("ane", "mlx", "auto"):
            rows = [r for r in records
                    if r["backend_pin"] == pin and r["context_case"] == case.name]
            c = cell(rows)
            answered = {r.get("backend_answered") for r in rows
                        if r.get("backend_answered")}
            c.update(backend_pin=pin, context_case=case.name,
                     token_count=case.token_count,
                     backend_answered=sorted(answered))
            cells.append(c)

    report: dict[str, Any] = {
        "schema": "t015.latency-sweep.v1",
        "created_unix": time.time(),
        "budgets_ms": {
            "e2e_warm": BUDGET_E2E_MS, "mlx_p95": BUDGET_MLX_P95_MS,
            "ane_p95": BUDGET_ANE_P95_MS,
            "overhead_p95": BUDGET_OVERHEAD_P95_MS,
            "overhead_p99": BUDGET_OVERHEAD_P99_MS,
        },
        "service": {"binary": str(args.binary), "config": str(args.config),
                    "health_warm": ready_health,
                    "ane_dispatch_ready_ms": dispatch_ready_ms},
        "method": {
            "warm_repeats": args.warm_repeats,
            "percentile": "nearest-rank ceil(p*n)",
            "overhead_proxy": "/route client latency (validate+tokenize+route, no execution) + predict client−server delta",
            "short_contexts": f"token_count <= {ANE_TOKEN_CEILING} (ANE L96 bucket)",
        },
        "contexts": [c.public() for c in cases],
        "cold": cold,
        "cells": cells,
        "route_overhead": {
            "per_case": [
                {"context_case": case.name, "token_count": case.token_count,
                 "client_ms": pstats([r["client_ms"] for r in overhead
                                      if r["context_case"] == case.name])}
                for case in cases
            ],
            "client_ms": pstats([r["client_ms"] for r in overhead]),
        },
        "counts": {"total": len(records) + len(overhead)},
        "raw_calls": records,
    }
    report["verdicts"] = verdicts(report)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    md_path = args.output.with_suffix(".md")
    md_path.write_text(markdown(report))

    print(markdown(report).split("## Warm cells")[0])
    fails = [v for v in report["verdicts"] if not v["ok"]]
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
