"""T024 frontier report: paired chunked-ANE experiment.

Reads results/chunks_paired.jsonl, prints accuracy per arm, backend-drift
vs context-loss decomposition, k-stratified latency, and disagreement
calibration. Writes results/chunks_paired_summary.json.
"""

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent


def acc(rows, key):
    ok = sum(1 for r in rows if r[key] == r["label"])
    return round(ok / max(1, len(rows)), 4), ok, len(rows)


def main():
    inp = HERE / "results" / "chunks_paired.jsonl"
    rows = [json.loads(l) for l in open(inp, encoding="utf-8") if l.strip()]
    docs = [r for r in rows if not r["fitting_control"]]
    print(f"n={len(rows)} k>=2 docs={len(docs)}")

    # arm correctness
    arms = {}
    for name, fn in [
        ("whole_en_mlx", lambda r: r["arms"]["whole_en_mlx"].get("predicted")),
        ("whole_multi_mlx", lambda r: r["arms"]["whole_multi_mlx"].get("predicted")),
        ("ane_majority", lambda r: r["agg_ane"]["majority"]),
        ("ane_pooled", lambda r: r["agg_ane"]["pooled"]),
        ("mlxchunk_majority", lambda r: r["agg_mlx_chunks"]["majority"]),
        ("mlxchunk_pooled", lambda r: r["agg_mlx_chunks"]["pooled"]),
    ]:
        ok = sum(1 for r in docs if fn(r) == r["label"])
        arms[name] = {"acc": round(ok / max(1, len(docs)), 4), "ok": ok, "n": len(docs)}

    # decomposition: backend drift (A-pool vs C-pool) vs aggregation loss (C-pool vs M)
    agree = lambda f, g: round(
        sum(1 for r in docs if f(r) == g(r)) / max(1, len(docs)), 4
    )
    A = lambda r: r["agg_ane"]["pooled"]
    C = lambda r: r["agg_mlx_chunks"]["pooled"]
    M = lambda r: r["arms"]["whole_multi_mlx"].get("predicted")
    E = lambda r: r["arms"]["whole_en_mlx"].get("predicted")

    # latency sums per doc per arm (server ms where present else client)
    def arm_ms(r, arm):
        if arm in ("ane_chunks", "mlx_chunks"):
            return sum((c.get("server_ms") or c.get("latency_ms") or 0) for c in r["arms"][arm])
        c = r["arms"][arm]
        return c.get("server_ms") or c.get("latency_ms") or 0

    lat = {}
    for arm in ["whole_en_mlx", "whole_multi_mlx", "ane_chunks", "mlx_chunks"]:
        xs = sorted(arm_ms(r, arm) for r in docs)
        lat[arm] = {
            "p50": round(xs[len(xs) // 2], 1),
            "p95": round(xs[int(len(xs) * 0.95)], 1),
            "n": len(xs),
        }

    # k distribution + accuracy by k for pooled arms
    by_k = {}
    for r in docs:
        d = by_k.setdefault(r["k"], {"n": 0, "ane": 0, "mlx": 0})
        d["n"] += 1
        d["ane"] += r["agg_ane"]["pooled"] == r["label"]
        d["mlx"] += r["agg_mlx_chunks"]["pooled"] == r["label"]

    # disagreement calibration: bin by vote_entropy -> pooled accuracy
    bins = {"zero": [], "low": [], "high": []}
    for r in docs:
        ve = r["agg_ane"]["vote_entropy"]
        bins["zero" if ve == 0 else ("low" if ve < 0.5 else "high")].append(r)
    calib = {
        k: {
            "n": len(v),
            "pool_acc": round(sum(1 for r in v if r["agg_ane"]["pooled"] == r["label"]) / max(1, len(v)), 4),
            "esc_rate": round(sum(1 for r in v if r["agg_ane"]["would_escalate"]) / max(1, len(v)), 4),
        }
        for k, v in bins.items()
    }

    # fallbacks: ANE-chunk calls not served by ane
    fb = sum(1 for r in docs for c in r["arms"]["ane_chunks"] if c.get("backend") != "ane")
    calls = sum(len(r["arms"]["ane_chunks"]) for r in docs)
    ties = sum(1 for r in docs if r["agg_ane"]["tie"])
    failed = sum(r["agg_ane"]["failed_chunks"] for r in docs)

    summary = {
        "n_docs": len(docs),
        "arms": arms,
        "agreement": {
            "ane_pooled_vs_mlxchunk_pooled": agree(A, C),
            "mlxchunk_pooled_vs_whole_multi": agree(C, M),
            "ane_pooled_vs_whole_multi": agree(A, M),
            "whole_multi_vs_whole_en": agree(M, E),
        },
        "latency_sums_ms": lat,
        "by_k": {str(k): {"n": v["n"], "ane_acc": round(v["ane"] / v["n"], 4),
                          "mlxchunk_acc": round(v["mlx"] / v["n"], 4)} for k, v in sorted(by_k.items())},
        "disagreement_calibration": calib,
        "ane_fallback_calls": fb,
        "ane_total_calls": calls,
        "ties": ties,
        "failed_chunks": failed,
    }

    print("## arms (accuracy)")
    for k, v in arms.items():
        print(f"- {k}: {v['acc']:.4f} ({v['ok']}/{v['n']})")
    print("## agreement (decomposition)")
    for k, v in summary["agreement"].items():
        print(f"- {k}: {v:.4f}")
    print("## latency sums p50/p95 ms")
    for k, v in lat.items():
        print(f"- {k}: {v['p50']}/{v['p95']}")
    print("## by k (ane_pooled vs mlxchunk_pooled acc)")
    for k, v in summary["by_k"].items():
        print(f"- k={k} n={v['n']}: ane {v['ane_acc']:.3f} mlxc {v['mlxchunk_acc']:.3f}")
    print("## disagreement calibration (vote_entropy bins -> pool acc / esc rate)")
    for k, v in calib.items():
        print(f"- {k}: n={v['n']} acc={v['pool_acc']:.3f} esc={v['esc_rate']:.3f}")
    print(f"## integrity: ane fallbacks {fb}/{calls}, ties {ties}, failed chunks {failed}")

    out = HERE / "results" / "chunks_paired_summary.json"
    json.dump(summary, open(out, "w"), indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
