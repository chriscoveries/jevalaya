"""Prepare AG News test split -> tools/bench/datasets/agnews_test.jsonl.

Source: fancyzhx/ag_news (namespace-qualified mirror of AG News; the bare
`ag_news` id is rejected by current huggingface_hub). 7600 test samples,
labels World/Sports/Business/SciTech.

Output lines: {"id": "agnews-test-<i>", "text": ..., "label": ...}.
Deterministic: dataset order is the split order; no shuffling.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "datasets"
OUT = OUT_DIR / "agnews_test.jsonl"
LABELS = ["World", "Sports", "Business", "Sci/Tech"]


def main() -> None:
    from datasets import load_dataset

    ds = load_dataset("fancyzhx/ag_news", split="test")
    assert len(ds) == 7600, f"unexpected AG News test size: {len(ds)}"
    assert list(ds.features["label"].names) == LABELS, ds.features["label"].names
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            f.write(
                json.dumps(
                    {"id": f"agnews-test-{i}", "text": row["text"], "label": LABELS[row["label"]]},
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"wrote {len(ds)} samples to {OUT}")


if __name__ == "__main__":
    sys.exit(main())
