#!/usr/bin/env python3
"""Run a reproducible local ANE/MLX latency suite through ``/predict``.

The runner owns the service lifecycle so a cold run always starts a fresh
process. It never sends ``backend=auto`` or ``backend=jev``: every inference
request pins either ``ane`` or ``mlx`` and ``model=multilingual``. The service
health response is checked before any request and the run aborts if Jev is
available. Outputs are a raw JSON record plus a small Markdown report.

The local benchmark config is deliberately ignored by git because it contains
machine-specific model paths. Pass it with ``--config``; see
``docs/JEVALAYA-BENCH.md`` for the setup and exact command.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "tools/bench/jevalaya.bench.toml"
DEFAULT_BINARY = ROOT / "target/release/jevalaya"
DEFAULT_BASE_URL = "http://127.0.0.1:8767"
DEFAULT_TOKEN = "bench-local"
PHRASE = "The customer asks about the order delivery and billing support. "
QUESTION = {
    "topic": {
        "type": "choice",
        "instructions": "Choose the issue category.",
        "criteria": {
            "billing": None,
            "shipping": None,
            "support": None,
            "technical": None,
        },
    }
}
DEFAULT_TARGETS = (24, 48, 72, 88, 94, 120, 256, 512, 900)


@dataclass(frozen=True)
class ContextCase:
    name: str
    target_tokens: int
    repetitions: int
    state: str
    token_count: int
    ane_eligible: bool | None

    def public(self) -> dict[str, Any]:
        return {
            "case": self.name,
            "target_tokens": self.target_tokens,
            "repetitions": self.repetitions,
            "token_count": self.token_count,
            "ane_eligible": self.ane_eligible,
        }


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1, math.ceil(p * len(ordered)) - 1))
    return round(ordered[index], 3)


def latency_stats(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "min_ms": round(min(values), 3) if values else None,
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "max_ms": round(max(values), 3) if values else None,
        "mean_ms": round(statistics.fmean(values), 3) if values else None,
    }


def request_body(state: str, backend: str, request_id: str) -> dict[str, Any]:
    return {
        "state": state,
        "questions": QUESTION,
        "backend": backend,
        "model": "multilingual",
        "request_id": request_id,
    }


class Service:
    """A fresh jevalaya process and its HTTP client."""

    def __init__(
        self,
        binary: Path,
        config: Path,
        base_url: str,
        token: str,
        log_path: Path,
    ) -> None:
        self.binary = binary
        self.config = config
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.log_path = log_path
        self.log_file = None
        self.process: subprocess.Popen[bytes] | None = None
        self.client = httpx.Client(http2=False)
        self.started_at = 0.0

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_path.open("wb")
        env = os.environ.copy()
        env["JEVALAYA_TOKEN"] = self.token
        env["HF_HUB_OFFLINE"] = "1"
        env.pop("TYPESAFE_API_KEY", None)
        self.started_at = time.perf_counter()
        self.process = subprocess.Popen(
            [str(self.binary), "serve", "--config", str(self.config)],
            cwd=str(ROOT),
            env=env,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
        )

    def health(self) -> dict[str, Any]:
        response = self.client.get(f"{self.base_url}/health", timeout=2.0)
        response.raise_for_status()
        return response.json()

    def wait_health(self, timeout: float) -> tuple[dict[str, Any], float]:
        deadline = time.perf_counter() + timeout
        last_error: Exception | None = None
        while time.perf_counter() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(self.failure_message())
            try:
                health = self.health()
                elapsed = round((time.perf_counter() - self.started_at) * 1000, 3)
                return health, elapsed
            except (httpx.HTTPError, OSError, ValueError) as error:
                last_error = error
                time.sleep(0.25)
        raise TimeoutError(f"health timeout for {self.log_path}: {last_error}")

    def wait_ane_ready(self, timeout: float) -> tuple[dict[str, Any], float]:
        deadline = time.perf_counter() + timeout
        health: dict[str, Any] = {}
        while time.perf_counter() < deadline:
            health = self.health()
            ane = (health.get("backends") or {}).get("ane") or {}
            detail = ane.get("detail") or ""
            if ane.get("available") and "warming" not in detail:
                elapsed = round((time.perf_counter() - self.started_at) * 1000, 3)
                return health, elapsed
            time.sleep(1.0)
        raise TimeoutError(f"ANE residency timeout; last health={health}")

    def failure_message(self) -> str:
        tail = ""
        try:
            tail = self.log_path.read_text(errors="replace")[-4000:]
        except OSError:
            pass
        status = self.process.returncode if self.process is not None else None
        return f"jevalaya exited with {status}; log tail:\n{tail}"

    def close(self) -> None:
        try:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=12)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
        finally:
            self.client.close()
            if self.log_file is not None:
                self.log_file.close()

    def __enter__(self) -> "Service":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def assert_no_jev(health: dict[str, Any]) -> None:
    """Reject a run where the health snapshot exposes an available Jev lane."""

    jev = (health.get("backends") or {}).get("jev")
    if isinstance(jev, dict) and jev.get("available"):
        raise RuntimeError(
            "Jev is available in the service health snapshot; refusing to run a paid-capable bench"
        )


def call_predict(
    client: httpx.Client,
    base_url: str,
    token: str,
    state: str,
    backend: str,
    request_id: str,
    timeout: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        response = client.post(
            f"{base_url.rstrip('/')}/predict",
            json=request_body(state, backend, request_id),
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout,
        )
        client_ms = round((time.perf_counter() - started) * 1000, 3)
    except httpx.HTTPError as error:
        return {
            "request_id": request_id,
            "backend_requested": backend,
            "status": None,
            "client_ms": round((time.perf_counter() - started) * 1000, 3),
            "error": f"transport:{type(error).__name__}:{error}",
        }

    result: dict[str, Any] = {
        "request_id": request_id,
        "backend_requested": backend,
        "status": response.status_code,
        "client_ms": client_ms,
        "error": "",
    }
    try:
        body = response.json()
    except ValueError:
        body = None
    if response.status_code != 200 or not isinstance(body, dict):
        result["error"] = f"http-{response.status_code}:{response.text[:300]}"
        return result

    routing = body.get("routing") or {}
    answer = (body.get("answers") or {}).get("topic") or {}
    result.update(
        {
            "backend_answered": routing.get("backend"),
            "checkpoint": routing.get("checkpoint"),
            "model_id": routing.get("model_id"),
            "reason": routing.get("reason"),
            "token_count": routing.get("token_count"),
            "ane_eligible": routing.get("ane_eligible"),
            "fallback": routing.get("fallback", False),
            "fallback_from": routing.get("fallback_from"),
            "fallback_reason": routing.get("fallback_reason"),
            "server_ms": routing.get("latency_ms"),
            "escalated": routing.get("escalated"),
            "confidence": routing.get("confidence"),
            "margin": routing.get("margin"),
            "choice": answer.get("choice"),
        }
    )
    return result


def route_probe(
    client: httpx.Client,
    base_url: str,
    token: str,
    state: str,
    request_id: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    response = client.post(
        f"{base_url.rstrip('/')}/route",
        json=request_body(state, "mlx", request_id),
        headers={"Authorization": f"Bearer {token}"},
        timeout=15.0,
    )
    client_ms = round((time.perf_counter() - started) * 1000, 3)
    try:
        body = response.json()
    except ValueError:
        body = {"error": response.text[:300]}
    return {
        "request_id": request_id,
        "client_ms": client_ms,
        "status": response.status_code,
        "token_count": body.get("token_count"),
        "backend": body.get("backend"),
        "checkpoint": body.get("checkpoint"),
        "reason": body.get("reason"),
        "ane_eligible": body.get("ane_eligible"),
        "error": "" if response.status_code == 200 else body,
    }


def discover_contexts(
    client: httpx.Client,
    base_url: str,
    token: str,
    targets: tuple[int, ...],
    max_repetitions: int,
) -> list[ContextCase]:
    candidates: list[tuple[int, str, dict[str, Any]]] = []
    for repetitions in range(1, max_repetitions + 1):
        state = PHRASE * repetitions
        probe = route_probe(client, base_url, token, state, f"t037-route-{repetitions}")
        token_count = probe.get("token_count")
        if probe.get("status") == 200 and isinstance(token_count, int):
            candidates.append((repetitions, state, probe))
    if not candidates:
        raise RuntimeError("/route returned no usable context token counts")

    cases: list[ContextCase] = []
    used: set[int] = set()
    for target in targets:
        repetitions, state, probe = min(
            candidates, key=lambda row: abs(row[2]["token_count"] - target)
        )
        if repetitions in used:
            continue
        used.add(repetitions)
        token_count = int(probe["token_count"])
        cases.append(
            ContextCase(
                name=f"ctx_{token_count}",
                target_tokens=target,
                repetitions=repetitions,
                state=state,
                token_count=token_count,
                ane_eligible=probe.get("ane_eligible"),
            )
        )
    cases.sort(key=lambda case: case.token_count)
    return cases


def annotate(record: dict[str, Any], case: ContextCase, phase: str) -> dict[str, Any]:
    record.update(
        {
            "phase": phase,
            "context_case": case.name,
            "target_tokens": case.target_tokens,
            "context_repetitions": case.repetitions,
        }
    )
    return record


def run_cold_mlx(
    binary: Path,
    config: Path,
    base_url: str,
    token: str,
    log_dir: Path,
    startup_timeout: float,
    request_timeout: float,
) -> dict[str, Any]:
    with Service(binary, config, base_url, token, log_dir / "cold_mlx.log") as service:
        health, boot_ms = service.wait_health(startup_timeout)
        assert_no_jev(health)
        cold = call_predict(
            service.client,
            base_url,
            token,
            PHRASE,
            "mlx",
            "t037-cold-mlx",
            request_timeout,
        )
        followups = [
            call_predict(
                service.client,
                base_url,
                token,
                PHRASE,
                "mlx",
                f"t037-cold-mlx-warm-{index}",
                request_timeout,
            )
            for index in range(3)
        ]
        return {
            "health": health,
            "boot_to_health_ms": boot_ms,
            "cold_first_request": cold,
            "warm_followups": followups,
            "log": str(service.log_path),
        }


def run_warm_suite(
    binary: Path,
    config: Path,
    base_url: str,
    token: str,
    log_dir: Path,
    startup_timeout: float,
    ane_timeout: float,
    request_timeout: float,
    warm_repeats: int,
    targets: tuple[int, ...],
    max_repetitions: int,
) -> dict[str, Any]:
    with Service(binary, config, base_url, token, log_dir / "warm_suite.log") as service:
        health, boot_ms = service.wait_health(startup_timeout)
        assert_no_jev(health)
        cold_ane_route = call_predict(
            service.client,
            base_url,
            token,
            PHRASE,
            "ane",
            "t037-cold-ane-route",
            request_timeout,
        )
        ready_health, ane_ready_ms = service.wait_ane_ready(ane_timeout)
        contexts = discover_contexts(
            service.client, base_url, token, targets, max_repetitions
        )
        short = [case for case in contexts if case.token_count <= 96]
        long_contexts = [case for case in contexts if case.token_count > 96]
        calls: list[dict[str, Any]] = []
        for case in short:
            for backend in ("ane", "mlx"):
                for index in range(warm_repeats):
                    calls.append(
                        annotate(
                            call_predict(
                                service.client,
                                base_url,
                                token,
                                case.state,
                                backend,
                                f"t037-{backend}-{case.name}-warm-{index}",
                                request_timeout,
                            ),
                            case,
                            "warm",
                        )
                    )
        for case in long_contexts:
            for index in range(warm_repeats):
                calls.append(
                    annotate(
                        call_predict(
                            service.client,
                            base_url,
                            token,
                            case.state,
                            "mlx",
                            f"t037-mlx-{case.name}-warm-{index}",
                            request_timeout,
                        ),
                        case,
                        "warm",
                    )
                )
        overlimit_probes = [
            annotate(
                call_predict(
                    service.client,
                    base_url,
                    token,
                    case.state,
                    "ane",
                    f"t037-ane-{case.name}-overlimit",
                    request_timeout,
                ),
                case,
                "overlimit_probe",
            )
            for case in long_contexts
        ]
        return {
            "health_at_boot": health,
            "boot_to_health_ms": boot_ms,
            "cold_ane_route": cold_ane_route,
            "ane_ready_health": ready_health,
            "ane_resident_ready_ms": ane_ready_ms,
            "contexts": [case.public() for case in contexts],
            "calls": calls,
            "overlimit_probes": overlimit_probes,
            "log": str(service.log_path),
        }


def summarize(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    good = [row for row in calls if not row.get("error") and row.get("status") == 200]
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in good:
        key = (
            str(row.get("backend_requested")),
            str(row.get("backend_answered")),
            str(row.get("context_case")),
        )
        grouped.setdefault(key, []).append(row)
    output = []
    for (requested, answered, context), rows in sorted(grouped.items()):
        server_values = [
            float(row["server_ms"])
            for row in rows
            if isinstance(row.get("server_ms"), (int, float))
        ]
        output.append(
            {
                "backend_requested": requested,
                "backend_answered": answered,
                "context_case": context,
                "token_count": rows[0].get("token_count"),
                "n": len(rows),
                "client_ms": latency_stats([float(row["client_ms"]) for row in rows]),
                "server_ms": latency_stats(server_values),
                "fallback_count": sum(bool(row.get("fallback")) for row in rows),
                "reasons": sorted({row.get("reason") or "" for row in rows}),
            }
        )
    return output


def aggregate_rows(calls: list[dict[str, Any]], predicate: Any) -> dict[str, Any]:
    rows = [row for row in calls if predicate(row) and not row.get("error")]
    server = [float(row["server_ms"]) for row in rows if isinstance(row.get("server_ms"), (int, float))]
    return {
        "n": len(rows),
        "client_ms": latency_stats([float(row["client_ms"]) for row in rows]),
        "server_ms": latency_stats(server),
    }


def markdown_report(result: dict[str, Any]) -> str:
    warm = result["warm_suite"]
    calls = warm["calls"]
    short_cases = {
        case["case"] for case in warm["contexts"] if case["token_count"] <= 96
    }
    ane_common = aggregate_rows(
        calls,
        lambda row: row.get("backend_requested") == "ane"
        and row.get("backend_answered") == "ane",
    )
    mlx_common = aggregate_rows(
        calls,
        lambda row: row.get("backend_requested") == "mlx"
        and row.get("context_case") in short_cases,
    )
    by_case = {
        (row["backend_requested"], row["backend_answered"], row["context_case"]): row
        for row in result["summary"]
    }
    lines = [
        "# jevalaya local-backend benchmark",
        "",
        "This report was generated by `tools/bench/run_local_backends.py`. "
        "Every inference request pins `model=multilingual` and `backend=ane` or `backend=mlx`; Jev is not exercised.",
        "",
        "## Headline: common short contexts",
        "",
        "| requested / answered | client p50 | client p95 | server p50 | server p95 |",
        "|---|---:|---:|---:|---:|",
        f"| ane / ane | {ane_common['client_ms']['p50_ms']} ms | {ane_common['client_ms']['p95_ms']} ms | {ane_common['server_ms']['p50_ms']} ms | {ane_common['server_ms']['p95_ms']} ms |",
        f"| mlx / mlx | {mlx_common['client_ms']['p50_ms']} ms | {mlx_common['client_ms']['p95_ms']} ms | {mlx_common['server_ms']['p50_ms']} ms | {mlx_common['server_ms']['p95_ms']} ms |",
        "",
        "## Context scaling",
        "",
        "The ANE bundle is fixed at L96. MLX rows continue beyond that ceiling.",
        "",
        "| rendered tokens | ANE client p50/p95 | MLX client p50/p95 |",
        "|---:|---:|---:|",
    ]
    for case in sorted(warm["contexts"], key=lambda value: value["token_count"]):
        ane_row = by_case.get(("ane", "ane", case["case"]))
        mlx_row = by_case.get(("mlx", "mlx", case["case"]))
        ane_text = (
            f"{ane_row['client_ms']['p50_ms']} / {ane_row['client_ms']['p95_ms']} ms"
            if ane_row
            else "over ceiling → MLX"
        )
        mlx_text = (
            f"{mlx_row['client_ms']['p50_ms']} / {mlx_row['client_ms']['p95_ms']} ms"
            if mlx_row
            else "—"
        )
        lines.append(f"| {case['token_count']} | {ane_text} | {mlx_text} |")
    cold_mlx = result["cold_mlx"]["cold_first_request"]
    cold_ane = warm["cold_ane_route"]
    lines.extend(
        [
            "",
            "## Cold behavior",
            "",
            f"- First MLX request: `{cold_mlx.get('client_ms')} ms` client / `{cold_mlx.get('server_ms')} ms` server; answered `{cold_mlx.get('backend_answered')}`.",
            f"- First ANE request: `{cold_ane.get('client_ms')} ms` client / `{cold_ane.get('server_ms')} ms` server; answered `{cold_ane.get('backend_answered')}` with `fallback={cold_ane.get('fallback')}` and reason `{cold_ane.get('fallback_reason')}`.",
            f"- ANE residency became ready after `{warm.get('ane_resident_ready_ms')} ms` from process start. The service deliberately serves MLX while ANE is warming.",
            "",
            "## Reproduction metadata",
            "",
            f"- Config: `{result['service']['config']}`",
            f"- Binary: `{result['service']['binary']}`",
            f"- Base URL: `{result['service']['base_url']}`",
            f"- Warm repeats per context/backend: `{result['method']['warm_repeats_per_context']}`",
            f"- Raw calls: `{result['counts']['calls_total']}`; errors: `{result['counts']['calls_error']}`",
            f"- Health Jev lane: `{result['service']['health_jev']}`",
            "",
        ]
    )
    return "\n".join(lines)


def parse_targets(value: str) -> tuple[int, ...]:
    try:
        targets = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("targets must be comma-separated integers") from error
    if not targets or any(target <= 0 for target in targets):
        raise argparse.ArgumentTypeError("targets must contain positive integers")
    return targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--token", default=DEFAULT_TOKEN)
    parser.add_argument("--output", type=Path, default=ROOT / "tools/bench/results/t037-local-backends.json")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--build", action="store_true", help="run cargo build --release before the benchmark")
    parser.add_argument("--warm-repeats", type=int, default=20)
    parser.add_argument("--startup-timeout", type=float, default=45.0)
    parser.add_argument("--ane-timeout", type=float, default=180.0)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--max-repetitions", type=int, default=100)
    parser.add_argument("--targets", type=parse_targets, default=DEFAULT_TARGETS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.config = args.config.expanduser().resolve()
    args.binary = args.binary.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.report = (args.report or args.output.with_suffix(".md")).expanduser().resolve()
    args.log_dir = (args.log_dir or args.output.parent).expanduser().resolve()
    if args.warm_repeats < 1:
        raise SystemExit("--warm-repeats must be positive")
    if args.max_repetitions < 1:
        raise SystemExit("--max-repetitions must be positive")
    if not args.config.is_file():
        raise SystemExit(f"missing config: {args.config}; copy the local bench config and pass --config")
    if args.build:
        env = os.environ.copy()
        env["HF_HUB_OFFLINE"] = "1"
        subprocess.run(["cargo", "build", "--release"], cwd=ROOT, env=env, check=True)
    if not args.binary.is_file():
        raise SystemExit(f"missing binary: {args.binary}; run with --build")
    if urlparse(args.base_url).scheme not in {"http", "https"}:
        raise SystemExit("--base-url must include http:// or https://")
    try:
        with httpx.Client() as probe:
            if probe.get(f"{args.base_url.rstrip('/')}/health", timeout=1.0).status_code == 200:
                raise SystemExit(f"a service already answers at {args.base_url}; stop it before this run")
    except httpx.HTTPError:
        pass

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    cold_mlx = run_cold_mlx(
        args.binary,
        args.config,
        args.base_url,
        args.token,
        args.log_dir,
        args.startup_timeout,
        args.request_timeout,
    )
    warm_suite = run_warm_suite(
        args.binary,
        args.config,
        args.base_url,
        args.token,
        args.log_dir,
        args.startup_timeout,
        args.ane_timeout,
        args.request_timeout,
        args.warm_repeats,
        args.targets,
        args.max_repetitions,
    )
    all_calls = (
        warm_suite["calls"]
        + warm_suite["overlimit_probes"]
        + [warm_suite["cold_ane_route"], cold_mlx["cold_first_request"]]
        + cold_mlx["warm_followups"]
    )
    errors = [row for row in all_calls if row.get("error")]
    answered = {row.get("backend_answered") for row in all_calls if row.get("backend_answered")}
    if "jev" in answered:
        raise RuntimeError("a request was answered by Jev; refusing to publish the run")
    health_jev = (warm_suite["health_at_boot"].get("backends") or {}).get("jev")
    result: dict[str, Any] = {
        "schema": "t037.jevalaya-local-backends.v1",
        "created_unix": time.time(),
        "service": {
            "binary": str(args.binary),
            "config": str(args.config),
            "base_url": args.base_url,
            "token": "operator-supplied local bearer",
            "health_jev": health_jev,
            "answered_backends": sorted(answered),
        },
        "method": {
            "question": QUESTION,
            "phrase": PHRASE,
            "warm_repeats_per_context": args.warm_repeats,
            "targets": list(args.targets),
            "percentile": "nearest-rank (ceil(p*n))",
            "cold_definition": "first explicit request after a fresh service process; ANE cold request may be intentional MLX fallback while residency warms",
            "warm_definition": "sequential explicit requests after ANE health stops reporting warming",
            "same_requests": "all discovered contexts <=96 tokens are sent to both ane and mlx with model=multilingual",
        },
        "cold_mlx": cold_mlx,
        "warm_suite": warm_suite,
        "summary": summarize(warm_suite["calls"]),
        "counts": {
            "calls_total": len(all_calls),
            "calls_error": len(errors),
            "warm_calls": len(warm_suite["calls"]),
            "overlimit_probes": len(warm_suite["overlimit_probes"]),
            "answered_backends": {backend: sum(row.get("backend_answered") == backend for row in all_calls) for backend in sorted(answered)},
        },
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.report.write_text(markdown_report(result), encoding="utf-8")
    print(json.dumps({
        "json": str(args.output),
        "report": str(args.report),
        "calls": result["counts"],
        "contexts": warm_suite["contexts"],
    }, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit("interrupted")
