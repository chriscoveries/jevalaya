"""Generate fixed parity fixtures for the Rust canonical prompt renderer (T003).

Two outputs:
  1. Hermetic fixtures (fixtures/hermetic/): a small WordLevel tokenizer built
     from the union of all surface strings, plus expected ids/markers produced
     by laya_mlx.common.build_sequence + Agent._to_internal. Fully hermetic:
     Rust tests need only these files, no network, no HF cache.
  2. Real-model fixtures (fixtures/real/): expected ids/markers encoded with
     the real english + multilingual tokenizers from the HF cache. Rust tests
     using these are gated on LAYA_REAL_MODEL_DIR (local verification only).

Every string choice below is fixed and deterministic. Do not "improve" cases
without regenerating and reviewing the diff: these files are the correctness
criterion for the Rust port.
"""

import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ.get("LAYA_MLX_SRC", "/Users/chrisd/PROJECTS/laya-mlx"))

from tokenizers import Tokenizer as Backend  # noqa: E402
from tokenizers import models, pre_tokenizers  # noqa: E402

from laya_mlx.agent import Agent  # noqa: E402
from laya_mlx.common import QTYPES, build_sequence, render_options  # noqa: E402
from laya_mlx.tokenizer import Tokenizer  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "fixtures"
SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

# ---------------------------------------------------------------- fixed cases
# questions use the user-facing schema (type/instructions/criteria);
# state may be str, dict, or list (serialize_state covers all three).

CASES = [
    {
        "name": "en_choice_single",
        "state": "I was charged twice and want a refund",
        "questions": {
            "topic": {
                "type": "choice",
                "instructions": "Choose the topic",
                "criteria": ["billing", "refund", "shipping"],
            }
        },
    },
    {
        "name": "en_score",
        "state": "The package arrived late and damaged",
        "questions": {
            "level": {
                "type": "score",
                "instructions": "Rate urgency",
                "criteria": ["low", "medium", "high"],
            }
        },
    },
    {
        "name": "en_noul_defaults",
        "state": "hello world",
        "questions": {
            "yes": {"type": "noul", "instructions": "Is this a greeting?"},
        },
    },
    {
        "name": "en_noul_custom",
        "state": "delete my account now",
        "questions": {
            "risk": {
                "type": "noul",
                "instructions": "Is this high risk?",
                "criteria": {"false": "safe to proceed", "true": "requires escalation"},
            }
        },
    },
    {
        "name": "multi_question_mixed",
        "state": "The screen shows an error after payment",
        "questions": {
            "topic": {
                "type": "choice",
                "instructions": "Choose the topic",
                "criteria": ["billing", "refund", "shipping"],
            },
            "level": {
                "type": "score",
                "instructions": "Rate urgency",
                "criteria": ["low", "high"],
            },
            "yes": {"type": "noul", "instructions": "Is this true?"},
        },
    },
    {
        "name": "multilingual_zh",
        "state": "发票被重复扣款，请退款。",
        "questions": {
            "topic": {
                "type": "choice",
                "instructions": "选择主题",
                "criteria": ["账单", "退款", "物流"],
            }
        },
    },
    {
        "name": "multilingual_ar",
        "state": "تم تحصيل رسوم مضاعفة من حسابي",
        "questions": {
            "ok": {"type": "noul", "instructions": "هل هذا صحيح؟"},
        },
    },
    {
        "name": "multilingual_hi",
        "state": "मुझे धनवापसी चाहिए।",
        "questions": {
            "level": {
                "type": "score",
                "instructions": "प्राथमिकता",
                "criteria": ["कम", "उच्च"],
            }
        },
    },
    {
        "name": "typed_customer_service",
        "state": {"message": "Where is my order?", "orders": [{"id": 7}]},
        "questions": {
            "action": {
                "type": "choice",
                "instructions": "Pick the action",
                "criteria": ["reply", "escalate", "refund"],
            },
            "category": {
                "type": "choice",
                "instructions": "Pick the category",
                "criteria": ["shipping", "billing"],
            },
            "churn_risk": {
                "type": "score",
                "instructions": "Churn risk",
                "criteria": ["low", "high"],
            },
            "needs_human": {"type": "noul", "instructions": "Needs a human?"},
            "urgency": {
                "type": "score",
                "instructions": "Urgency",
                "criteria": ["low", "mid", "high"],
            },
        },
    },
    {
        "name": "mask_injection",
        "state": "[MASK] hello [MASK]",
        "questions": {
            "x": {"type": "noul", "instructions": "[MASK] true?"},
        },
    },
    {
        "name": "long_state_truncation",
        "state": "hello " * 1000,
        "questions": {
            "one": {
                "type": "choice",
                "instructions": "choose",
                "criteria": ["only"],
            }
        },
    },
    {
        "name": "structured_criteria",
        "state": {"text": "你好", "flag": False},
        "questions": {
            "topic": {
                "type": "choice",
                "instructions": "Choose",
                "criteria": {
                    "zero": 0,
                    "no": False,
                    "nested": {"value": 3},
                    "nothing": None,
                    "blank": "",
                },
            },
            "level": {
                "type": "score",
                "instructions": "Level",
                "criteria": ["low", {"tier": 2}, 5],
            },
        },
    },
    {
        "name": "nonstring_instructions",
        "state": "hello",
        "questions": {
            "t": {
                "type": "noul",
                "instructions": {"task": "verify"},
                "criteria": {"false": {"reason": "no"}, "true": {"reason": "yes"}},
            }
        },
    },
    {
        "name": "many_options_20",
        "state": "Pick one of many",
        "questions": {
            "big": {
                "type": "choice",
                "instructions": "Choose one",
                "criteria": [f"option {i}" for i in range(20)],
            }
        },
    },
    {
        "name": "many_long_options",
        "state": "Pick one of many long descriptions",
        "questions": {
            "big": {
                "type": "choice",
                "instructions": "Choose the best matching long description carefully",
                "criteria": [
                    f"alternative number {i} with a fairly long description attached"
                    for i in range(12)
                ],
            }
        },
    },
    {
        "name": "long_instructions",
        "state": "short state",
        "questions": {
            "q": {
                "type": "choice",
                "instructions": " ".join(["word"] * 120),
                "criteria": ["a", "b"],
            }
        },
    },
    {
        "name": "dict_state_unicode",
        "state": {"message": "Mein Konto wurde zweimal belastet", "n": 3},
        "questions": {
            "yes": {"type": "noul", "instructions": "Is this true?"},
        },
    },
    {
        "name": "list_state",
        "state": ["line one", {"k": "v"}, 42],
        "questions": {
            "level": {
                "type": "score",
                "instructions": "Rate it",
                "criteria": ["bad", "good"],
            }
        },
    },
]

CONFIGS = {
    "tiny": {"max_len": 128, "head_max_len": 32},
    "std": {"max_len": 512, "head_max_len": 192},
}

REAL_MODELS = {
    "english": "models--aac6fef--laya-mlx",
    "multilingual": "models--aac6fef--laya-multilingual-mlx",
}

REAL_CASES = [
    "en_choice_single",
    "en_noul_custom",
    "multilingual_zh",
    "multilingual_ar",
    "mask_injection",
    "structured_criteria",
    "nonstring_instructions",
    "many_options_20",
    "dict_state_unicode",
]


def snapshot_dir(cache_name):
    paths = sorted(glob.glob(f"/Users/chrisd/.cache/huggingface/hub/{cache_name}/snapshots/*"))
    assert paths, f"no cached snapshot for {cache_name}"
    return Path(paths[0])


def surface_strings():
    """Every exact string the encoder sees (mask scrub applied by callers)."""
    from laya_mlx.common import serialize_state

    texts = []
    for case in CASES:
        for qdef in case["questions"].values():
            q = Agent._to_internal(qdef)
            texts.append(f"{q['t']} question: {q['ins']}")
            texts.extend(" " + o for o in render_options(q))
        texts.append(serialize_state(case["state"]))
    return texts


def collect_words():
    """Union of pre-tokenizer output words over every encoded string.

    NOTE: tokenizers>=0.2x `Whitespace` isolates punctuation, so Python
    str.split() is NOT the word unit — query the real pre-tokenizer instead.
    Mask-scrubbing is applied first, exactly as build_sequence does.
    """
    probe = pre_tokenizers.Whitespace()
    words = set()
    for text in surface_strings():
        for word, _ in probe.pre_tokenize_str(text.replace("[MASK]", " ")):
            words.add(word)
    return words


def build_hermetic_tokenizer(words, path):
    vocab = {t: i for i, t in enumerate(SPECIALS)}
    for w in sorted(words):
        if w not in vocab:
            vocab[w] = len(vocab)
    tok = Backend(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "pad_token": "[PAD]",
                "cls_token": "[CLS]",
                "sep_token": "[SEP]",
                "mask_token": "[MASK]",
            }
        )
    )
    return vocab


class DictTok:
    """Minimal Tokenizer-protocol shim over the hermetic backend for build_sequence."""

    def __init__(self, backend, specials):
        self.backend = backend
        for name in ("cls_token", "sep_token", "pad_token", "mask_token"):
            setattr(self, name, specials[name])
            setattr(self, name + "_id", backend.token_to_id(specials[name]))

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self.backend.encode(text, add_special_tokens=add_special_tokens).ids}


def encode_case(tok, case, max_len, head_max_len):
    from laya_mlx.common import build_prefix, serialize_state

    items = []
    for qid, qdef in case["questions"].items():
        q = Agent._to_internal(qdef)
        ids, markers = build_sequence(tok, case["state"], q, max_len, head_max_len)
        if len(markers) != len(render_options(q)):
            raise ValueError(f"Question {qid!r} has too many options for the token budget")
        # Pre-truncation gate count: prefix + COMPLETE state + final [SEP].
        prefix_ids, _ = build_prefix(tok, q, head_max_len)
        full_state = tok(
            serialize_state(case["state"]).replace(tok.mask_token, " "),
            add_special_tokens=False,
        )["input_ids"]
        items.append(
            {
                "qid": qid,
                "qtype": QTYPES[q["t"]],
                "options": render_options(q),
                "ids": ids,
                "markers": markers,
                "raw_count": len(prefix_ids) + len(full_state) + 1,
            }
        )
    return items


def main():
    herm = OUT / "hermetic"
    (herm / "tokenizer").mkdir(parents=True, exist_ok=True)
    words = collect_words()
    vocab = build_hermetic_tokenizer(words, herm / "tokenizer")
    backend = Backend.from_file(str(herm / "tokenizer" / "tokenizer.json"))
    backend.no_padding()
    backend.no_truncation()
    tok = DictTok(
        backend,
        {"cls_token": "[CLS]", "sep_token": "[SEP]", "pad_token": "[PAD]", "mask_token": "[MASK]"},
    )
    assert tok.cls_token_id == 2 and tok.mask_token_id == 4, "special id drift"

    manifest = {"tokenizer_vocab_size": len(vocab), "configs": CONFIGS, "cases": {}}
    for cfg_name, cfg in CONFIGS.items():
        for case in CASES:
            items = encode_case(tok, case, cfg["max_len"], cfg["head_max_len"])
            key = f"{cfg_name}/{case['name']}"
            path = herm / cfg_name / f"{case['name']}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "config": cfg_name,
                        "max_len": cfg["max_len"],
                        "head_max_len": cfg["head_max_len"],
                        "case": case["name"],
                        "state": case["state"],
                        "questions": case["questions"],
                        "specials": {
                            "cls_token": "[CLS]",
                            "sep_token": "[SEP]",
                            "pad_token": "[PAD]",
                            "mask_token": "[MASK]",
                            "cls_token_id": tok.cls_token_id,
                            "sep_token_id": tok.sep_token_id,
                            "pad_token_id": tok.pad_token_id,
                            "mask_token_id": tok.mask_token_id,
                        },
                        "items": items,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            manifest["cases"][key] = {
                "input_tokens": sum(len(i["ids"]) for i in items),
                "markers": {i["qid"]: i["markers"] for i in items},
            }
    (herm / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    # ---- real-model fixtures (local verification only; needs HF cache)
    real = OUT / "real"
    real.mkdir(parents=True, exist_ok=True)
    by_name = {c["name"]: c for c in CASES}
    for model, cache in REAL_MODELS.items():
        mdir = snapshot_dir(cache)
        agent_tok = Tokenizer(mdir / "tokenizer")
        agent_cfg = json.loads((mdir / "rl_agent_config.json").read_text())
        max_len, head_max_len = agent_cfg["max_len"], agent_cfg["head_max_len"]
        tcfg = json.loads((mdir / "tokenizer" / "tokenizer_config.json").read_text())

        def tname(v):
            return v.get("content") if isinstance(v, dict) else v

        for name in REAL_CASES:
            case = by_name[name]
            items = encode_case(agent_tok, case, max_len, head_max_len)
            (real / f"{model}.{name}.json").write_text(
                json.dumps(
                    {
                        "model": model,
                        "snapshot": str(mdir),
                        "max_len": max_len,
                        "head_max_len": head_max_len,
                        "specials": {
                            "cls_token": tname(tcfg.get("cls_token")),
                            "sep_token": tname(tcfg.get("sep_token")),
                            "pad_token": tname(tcfg.get("pad_token")),
                            "mask_token": tname(tcfg.get("mask_token")),
                            "cls_token_id": agent_tok.cls_token_id,
                            "sep_token_id": agent_tok.sep_token_id,
                            "pad_token_id": agent_tok.pad_token_id,
                            "mask_token_id": agent_tok.mask_token_id,
                        },
                        "case": name,
                        "state": case["state"],
                        "questions": case["questions"],
                        "items": items,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
    n_herm = len(CASES) * len(CONFIGS)
    n_real = len(REAL_MODELS) * len(REAL_CASES)
    print(f"wrote {n_herm} hermetic + {n_real} real fixtures to {OUT}")


if __name__ == "__main__":
    main()
