"""Summarize a sweep results JSONL: accuracy, escalation, latency.

Usage: report.py results/agnews_mlx.jsonl [--json out.json]

Prints a markdown summary: overall + per-label accuracy, backend/checkpoint
mix, would-escalate rate and its accuracy lift if escalated requests are
excluded, latency percentiles (client + server), confidence/margin means.
"""

import argparse
import json
import statistics
import sys


def pct(rows):
    rows = [r for r in rows if r["latency_ms"] is not None]
    if not rows:
        return {}
    xs = sorted(r["latency_ms"] for r in rows)
    q = lambda p: xs[min(len(xs) - 1, int(p * len(xs)))]
    return {"n": len(xs), "p50": round(q(0.50), 1), "p95": round(q(0.95), 1), "p99": round(q(0.99), 1)}


def threshold_sweep(good):
    """Calibration curve: esc_rate/esc_acc/keep_acc over margin × conf grids.

    Rows: margin-alone gates, conf-alone gates, and OR combos — the cost
    (esc_rate ~ Jev volume) vs catch (keep_acc) trade the user decides on.
    """

    def num(key):
        return lambda r: r[key] if isinstance(r[key], (int, float)) else None

    def stats(pred):
        esc = [r for r in good if pred(r)]
        kept = [r for r in good if not pred(r)]
        return {
            "esc_rate": round(len(esc) / max(1, len(good)), 4),
            "esc_acc": round(sum(1 for r in esc if r["correct"]) / max(1, len(esc)), 4),
            "keep_acc": round(sum(1 for r in kept if r["correct"]) / max(1, len(kept)), 4),
        }

    table = []
    for m in (0.10, 0.15, 0.20, 0.30):
        table.append(
            {"gate": f"margin<{m}", **stats(lambda r, m=m: (num("margin")(r) or 1.0) < m)}
        )
    for c in (0.60, 0.70, 0.75, 0.80):
        table.append(
            {"gate": f"conf<{c}", **stats(lambda r, c=c: (num("confidence")(r) or 1.0) < c)}
        )
    for m in (0.20, 0.30):
        for c in (0.70, 0.75):
            table.append(
                {
                    "gate": f"margin<{m} OR conf<{c}",
                    **stats(
                        lambda r, m=m, c=c: (num("margin")(r) or 1.0) < m
                        or (num("confidence")(r) or 1.0) < c
                    ),
                }
            )
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.results, encoding="utf-8") if l.strip()]
    good = [r for r in rows if not r["error"]]
    bad = [r for r in rows if r["error"]]
    correct = [r for r in good if r["correct"]]
    esc = [r for r in good if r["would_escalate"]]

    by_label = {}
    for r in good:
        d = by_label.setdefault(r["label"], {"n": 0, "ok": 0})
        d["n"] += 1
        d["ok"] += r["correct"]

    backends = {}
    for r in good:
        key = f"{r['backend']}/{r['checkpoint']}"
        d = backends.setdefault(key, {"n": 0, "ok": 0})
        d["n"] += 1
        d["ok"] += r["correct"]

    confs = [r["confidence"] for r in good if isinstance(r["confidence"], (int, float))]
    margins = [r["margin"] for r in good if isinstance(r["margin"], (int, float))]
    kept = [r for r in good if not r["would_escalate"]]

    summary = {
        "n": len(rows),
        "errors": len(bad),
        "accuracy": round(len(correct) / max(1, len(good)), 4),
        "per_label": {
            k: {"n": v["n"], "acc": round(v["ok"] / v["n"], 4)} for k, v in sorted(by_label.items())
        },
        "per_backend": {
            k: {"n": v["n"], "acc": round(v["ok"] / v["n"], 4)} for k, v in sorted(backends.items())
        },
        "would_escalate_rate": round(len(esc) / max(1, len(good)), 4),
        "would_escalate_acc": round(
            sum(1 for r in esc if r["correct"]) / max(1, len(esc)), 4
        ),
        "kept_acc_if_escalated_excluded": round(
            sum(1 for r in kept if r["correct"]) / max(1, len(kept)), 4
        ),
        "latency_client_ms": pct(good),
        "latency_server_ms": pct(
            [{**r, "latency_ms": r["server_latency_ms"]} for r in good]
        ),
        "confidence_mean": round(statistics.mean(confs), 4) if confs else None,
        "margin_mean": round(statistics.mean(margins), 4) if margins else None,
        "threshold_sweep": threshold_sweep(good),
    }

    lines = [
        f"## sweep: {args.results}",
        f"- n={summary['n']} errors={summary['errors']} **accuracy={summary['accuracy']:.4f}**",
        "- per-label: "
        + ", ".join(f"{k} {v['acc']:.3f} (n={v['n']})" for k, v in summary["per_label"].items()),
        "- per-backend: "
        + ", ".join(f"{k} {v['acc']:.3f} (n={v['n']})" for k, v in summary["per_backend"].items()),
        f"- would-escalate rate={summary['would_escalate_rate']:.4f} "
        f"(acc on those={summary['would_escalate_acc']:.4f}); "
        f"kept-only acc={summary['kept_acc_if_escalated_excluded']:.4f}",
        f"- latency client ms: {summary['latency_client_ms']}",
        f"- latency server ms: {summary['latency_server_ms']}",
        f"- confidence mean={summary['confidence_mean']} margin mean={summary['margin_mean']}",
        "- threshold sweep (gate | esc_rate | esc_acc | keep_acc):",
    ]
    for row in summary["threshold_sweep"]:
        lines.append(
            f"  - {row['gate']} | {row['esc_rate']:.4f} | {row['esc_acc']:.4f} | {row['keep_acc']:.4f}"
        )
    print("\n".join(lines))
    if args.json:
        json.dump(summary, open(args.json, "w"), indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    sys.exit(main())
