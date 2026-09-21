"""Prepare CLINC150 test + OOS splits -> tools/bench/datasets/.

Source: the original clinc/oos-eval `data_full.json` (GitHub raw — true
intent-name strings, which the HF mirrors only carry as int ids).
Val split is skipped (test + oos_test are the sweep inputs).

Outputs:
  clinc150_test.jsonl  4500 in-scope lines {id, text, label(intent name)}
  clinc150_oos.jsonl   1000 out-of-scope lines {id, text, label: "oos"}
  clinc150_intents.json  sorted 150 intent names (the criteria vocabulary)
"""

import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "datasets"
URL = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = OUT_DIR / "data_full.json"
    if not raw_path.exists():
        print(f"downloading {URL} ...", flush=True)
        urllib.request.urlretrieve(URL, raw_path)
    data = json.loads(raw_path.read_text(encoding="utf-8"))

    # data_full.json shape: {"oos_train": [[text, intent]...], ...} where
    # in-scope splits carry intent names and oos splits carry "oos".
    names = set()
    for split in ("train", "val", "test"):
        for _, intent in data[split]:
            names.add(intent)
    intents = sorted(names)
    assert len(intents) == 150, f"expected 150 intents, got {len(intents)}"
    (OUT_DIR / "clinc150_intents.json").write_text(
        json.dumps(intents, indent=1), encoding="utf-8"
    )

    for split, fname in (("test", "clinc150_test.jsonl"), ("oos_test", "clinc150_oos.jsonl")):
        rows = data[split]
        with open(OUT_DIR / fname, "w", encoding="utf-8") as f:
            for i, (text, intent) in enumerate(rows):
                f.write(
                    json.dumps(
                        {"id": f"clinc150-{split}-{i}", "text": text, "label": intent},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print(f"wrote {len(rows)} rows to {fname}")
    print(f"intents: {len(intents)}")


if __name__ == "__main__":
    sys.exit(main())
