# Repeatable local-backend benchmark

`tools/bench/run_local_backends.py` owns the service lifecycle and runs the
same fixed request suite against the existing CoreML ANE bundle and the MLX
backend. It writes a raw JSON record and a Markdown comparison report. Every
inference request pins `model=multilingual` and either `backend=ane` or
`backend=mlx`; the runner refuses to start when the health snapshot exposes an
available Jev backend and refuses to publish if any response is answered by
Jev.

The machine-specific TOML is intentionally ignored by git because it contains
absolute model and Python paths. Create `tools/bench/jevalaya.bench.toml` with
the local paths and a disabled Jev section. The service must expose the normal
`/health`, `/route`, and `/predict` endpoints, and the config should point the
`ane` backend at the old `laya-multilingual-coreml-ane` `.mlpackage` and the
`mlx` backend at the `laya-mlx` Python environment.

From the repository root, run:

```sh
PYO3_PYTHON=/absolute/path/to/laya-mlx/.venv/bin/python \
  /absolute/path/to/laya-mlx/.venv/bin/python \
  tools/bench/run_local_backends.py \
  --config tools/bench/jevalaya.bench.toml \
  --build \
  --output tools/bench/results/t037-local-backends.json
```

`--build` is optional when `target/release/jevalaya` is already current. The
default suite takes 20 sequential warm samples per backend and discovered
context. Targets are 24, 48, 72, 88, 94, 120, 256, 512, and 900 rendered
tokens; `/route` chooses the nearest reproducible repetition of the fixed
phrase. Contexts at or below the ANE L96 ceiling are sent to both backends.
Longer contexts continue through MLX and receive an explicit ANE over-limit
probe, so the JSON records the routing and fallback metadata at the boundary.

The cold section starts a fresh process for MLX and records its first request
plus three follow-ups. The warm section starts another fresh process, records
the first explicit ANE request (which may be served by MLX while ANE residency
warms), waits until `/health` stops reporting `warming`, and then runs the
context sweep. Each call records client and server latency, requested and
answering backend, model/checkpoint, token count, ANE eligibility, fallback
reason, confidence, margin, and the typed answer.

The output paths can be overridden with `--output`, `--report`, and `--log-dir`.
Use a temporary output directory for an uncommitted smoke run, for example:

```sh
.../run_local_backends.py --warm-repeats 1 --targets 32,98 \
  --output /tmp/t037-local-backends.json \
  --report /tmp/t037-local-backends.md \
  --log-dir /tmp/t037-local-backends-logs
```
