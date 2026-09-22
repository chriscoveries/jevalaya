"""T024 prep: AG News >96-token docs -> naive word-window chunks for ANE L96.

Canonical geometry per DESIGN.md §Chunked inference strategy:
  P = len(build_prefix(bundle_tok, question, head_max_len))  (actual prefix)
  B = L - P - 1  with L=96, a=1  (state allowance; reject B <= 0)
Chunks: greedy word windows with encode(window) <= B, complete coverage,
original order. Each chunk verified by re-rendering
build_sequence(tok, chunk, q, max_len=96) and asserting zero truncation
(ids == prefix + chunk + sep) and intact markers.

Question: fixed 4-label AG News choice (same head as the sweeps).
Tokenizer: multilingual bundle tokenizer (the ANE gate's tokenizer).

Inputs: tools/bench/datasets/agnews_test.jsonl + results/agnews_mlx.jsonl
(english token_count selects long docs).
Output: datasets/agnews_chunks.jsonl lines:
  {id, label, k, prefix_len, B, whole_tokens_multi,
   chunks: [{text, tokens}], flags: {single_word_overrun: bool}}
Docs with k<2 are kept flagged (fitting controls); k>6 skipped for now.
"""

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "datasets"

LENGTH = 96
LABELS = ["World", "Sports", "Business", "Sci/Tech"]
INSTRUCTIONS = "Classify the news article by topic."
MIN_K = 2
MAX_K = 6


def snapshot(sub):
    root = Path(os.environ.get(
        "LAYA_SNAPSHOT_DIR",
        "/Users/chrisd/.cache/huggingface/hub/models--aac6fef--laya-multilingual-mlx/snapshots/f2b4faf51023039425946074e2cf1361d2db11d5",
    ))
    if sub:
        return root / sub
    return root


def main() -> None:
    sys.path.insert(0, os.environ.get("LAYA_MLX_SRC", "/Users/chrisd/PROJECTS/laya-mlx"))
    from laya_mlx.agent import Agent
    from laya_mlx.common import build_prefix, build_sequence, serialize_state
    from laya_mlx.tokenizer import Tokenizer

    tok = Tokenizer(snapshot("tokenizer"))
    cfg = json.loads((snapshot("") / "rl_agent_config.json").read_text())
    head_max_len = cfg["head_max_len"]
    q = Agent._to_internal(
        {"type": "choice", "instructions": INSTRUCTIONS, "criteria": LABELS}
    )
    prefix_ids, _ = build_prefix(tok, q, head_max_len)
    P = len(prefix_ids)
    B = LENGTH - P - 1
    assert B > 0, f"no state room: L={LENGTH} P={P}"
    print(f"prefix P={P} state allowance B={B} (L{LENGTH}/a1)", flush=True)

    results_path = Path(__file__).resolve().parent / "results" / "agnews_mlx.jsonl"
    counts = {}
    for line in open(results_path, encoding="utf-8"):
        r = json.loads(line)
        counts[r["id"]] = r["token_count"]

    mask = tok.mask_token
    out_path = OUT_DIR / "agnews_chunks.jsonl"
    kept = skipped_overrun = skipped_k = fitting = 0
    with open(OUT_DIR / "agnews_test.jsonl", encoding="utf-8") as fin, open(
        out_path, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            s = json.loads(line)
            if not (counts.get(s["id"]) or 0) > 96:
                continue
            text = serialize_state(s["text"]).replace(mask, " ")
            words = text.split()
            chunks, cur = [], []
            overrun = False
            for w in words:
                trial = cur + [w]
                ids = tok(" ".join(trial), add_special_tokens=False)["input_ids"]
                if len(ids) <= B:
                    cur = trial
                else:
                    if not cur:
                        overrun = True
                        break
                    chunks.append(" ".join(cur))
                    cur = [w]
                    if len(tok(" ".join(cur), add_special_tokens=False)["input_ids"]) > B:
                        overrun = True
                        break
            if overrun:
                skipped_overrun += 1
                continue
            if cur:
                chunks.append(" ".join(cur))
            # Verify: exact re-render, zero truncation, intact markers.
            verified = []
            ok = True
            for c in chunks:
                ids, markers = build_sequence(tok, c, q, LENGTH, head_max_len)
                raw = prefix_ids + tok(c, add_special_tokens=False)["input_ids"] + [tok.sep_token_id]
                if ids != raw[:LENGTH] or len(markers) != len(LABELS):
                    ok = False
                    break
                verified.append({"text": c, "tokens": len(ids)})
            if not ok:
                skipped_overrun += 1
                continue
            k = len(verified)
            if k < MIN_K:
                fitting += 1
            if k > MAX_K:
                skipped_k += 1
                continue
            whole = tok(text, add_special_tokens=False)["input_ids"]
            fout.write(
                json.dumps(
                    {
                        "id": s["id"],
                        "label": s["label"],
                        "k": k,
                        "prefix_len": P,
                        "B": B,
                        "whole_tokens_multi": len(whole),
                        "chunks": verified,
                        "fitting_control": k < MIN_K,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            kept += 1
    print(f"kept={kept} fitting_controls(included) fitting={fitting} overrun={skipped_overrun} k_over={skipped_k}")


if __name__ == "__main__":
    main()
