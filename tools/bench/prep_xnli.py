"""Prepare XNLI en/de/fr test splits -> tools/bench/datasets/xnli_{en,de,fr}.jsonl.

Source: facebook/xnli (per-language configs, 5010 test rows each).
State format: "Premise: <premise> Hypothesis: <hypothesis>".
Labels: entailment / neutral / contradiction (ClassLabel order 0/1/2).

Output lines: {"id": "xnli-<lang>-<i>", "text": ..., "label": ...}.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "datasets"
LANGS = ["en", "de", "fr"]
LABELS = ["entailment", "neutral", "contradiction"]


def main() -> None:
    from datasets import load_dataset

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for lang in LANGS:
        ds = load_dataset("facebook/xnli", lang, split="test")
        assert len(ds) == 5010, f"{lang}: unexpected size {len(ds)}"
        if list(ds.features["label"].names) != LABELS:
            raise ValueError(f"{lang}: unexpected labels {ds.features['label'].names}")
        out = OUT_DIR / f"xnli_{lang}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for i, row in enumerate(ds):
                f.write(
                    json.dumps(
                        {
                            "id": f"xnli-{lang}-{i}",
                            "text": f"Premise: {row['premise']} Hypothesis: {row['hypothesis']}",
                            "label": LABELS[row["label"]],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print(f"wrote {len(ds)} rows to {out}")


if __name__ == "__main__":
    sys.exit(main())
