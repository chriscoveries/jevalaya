#!/usr/bin/env python3
"""Offline FluidInference compiled-bucket vs laya-mlx parity probe.

This is evidence tooling, not an ANE-XL backend or an automatic promotion gate.
It performs no downloads, never uses Jev, and refuses silent prompt truncation.
See docs/ANE-XL-PARITY.md for pinned artifacts and an isolated environment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


CASES = [
    {"id": "choice_refund", "state": "I was charged twice for my order. Please refund the duplicate payment.",
     "question": {"type": "choice", "instructions": "What does the customer want?",
                  "criteria": {"refund": None, "order status": None, "technical help": None}}},
    {"id": "score_outage", "state": "The application cannot start. Every user sees an error and cannot work.",
     "question": {"type": "score", "instructions": "How severe is this issue?",
                  "criteria": ["trivial", "annoying", "unusable"]}},
    {"id": "noul_delivery", "state": "The tracking page says the parcel was delivered this morning.",
     "question": {"type": "noul", "instructions": "Has the parcel been delivered?"}},
]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def softmax(values, temperature=1.0):
    z = np.asarray(values, dtype=np.float64)
    if not z.size or not np.isfinite(z).all() or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Non-finite logits or invalid calibration temperature")
    z = z / max(1e-3, float(temperature))
    p = np.exp(z - z.max())
    return p / p.sum()


def inspect_mil(bundle: Path, length: int) -> dict:
    """Read the actual compiled program, not the model-card input description."""
    program = (bundle / "model.mil").read_text()
    signature = re.search(r"func main<[^>]+>\((.*?)\)\s*\{", program, re.S)
    if signature is None:
        raise ValueError("Compiled program has no recognizable main signature")
    found = {}
    for dtype, shape, name in re.findall(r"tensor<(\w+),\s*\[([^\]]*)\]>\s+(\w+)", signature[1]):
        found[name] = {"dtype": dtype, "shape": [int(n.strip()) for n in shape.split(",")]}
    expected = {
        "input_ids": {"dtype": "int32", "shape": [1, length]},
        "attention_mask": {"dtype": "int32", "shape": [1, length]},
        "marker_map": {"dtype": "fp32", "shape": [1, 32, length]},
        "question_type": {"dtype": "fp32", "shape": [1, 3]},
    }
    if found != expected:
        raise ValueError(f"Unexpected compiled input signature: {found!r}")
    outputs = {"logits": [1, 32], "probabilities": [1, 32], "action_probabilities": [1, 2]}
    for name, shape in outputs.items():
        pattern = rf"tensor<fp32,\s*\[\s*{shape[0]},\s*{shape[1]}\s*\]>\s+{name}\s*="
        if not re.search(pattern, program):
            raise ValueError(f"Compiled program output {name} has an unexpected dtype/shape")
    if not re.search(r"\}\s*->\s*\(logits,\s*probabilities,\s*action_probabilities\)", program):
        raise ValueError("Unexpected compiled output names/order")
    return {"inputs": found, "outputs": {k: {"dtype": "fp32", "shape": v} for k, v in outputs.items()},
            "mil_sha256": digest(bundle / "model.mil")}


def build_inputs(item, length, pad_id, mask_id):
    ids, markers, qtype = item["ids"], item["markers"], item["qtype"]
    if not ids or len(ids) > length:
        raise ValueError(f"Rendered length {len(ids)} does not fit L{length}; no truncation allowed")
    if not 1 <= len(markers) <= 32 or len(set(markers)) != len(markers):
        raise ValueError("Expected 1..32 distinct option markers")
    if qtype not in (0, 1, 2):
        raise ValueError("Unknown question type")
    if any(not 0 <= token < 2**31 for token in [pad_id, *ids]):
        raise ValueError("Token id does not fit nonnegative int32")
    if any(pos < 0 or pos >= len(ids) or ids[pos] != mask_id for pos in markers):
        raise ValueError("Option marker must point at a real MASK token")
    token_ids = np.full((1, length), pad_id, dtype=np.int32)
    token_ids[0, :len(ids)] = ids
    mask = np.zeros((1, length), dtype=np.int32)
    mask[0, :len(ids)] = 1
    marker_map = np.zeros((1, 32, length), dtype=np.float32)
    for slot, position in enumerate(markers):
        marker_map[0, slot, position] = 1.0
    question_type = np.zeros((1, 3), dtype=np.float32)
    question_type[0, qtype] = 1.0
    return {"input_ids": token_ids, "attention_mask": mask,
            "marker_map": marker_map, "question_type": question_type}


def validate_outputs(outputs):
    expected = {"logits": (1, 32), "probabilities": (1, 32), "action_probabilities": (1, 2)}
    if set(outputs) != set(expected):
        raise ValueError(f"Unexpected output names: {sorted(outputs)}")
    for name, shape in expected.items():
        value = np.asarray(outputs[name])
        if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f"Invalid {name}: shape={value.shape}, dtype={value.dtype}")
        if name != "logits" and (np.any(value < 0) or np.any(value > 1) or abs(float(value.sum()) - 1) > .002):
            raise ValueError(f"Invalid probability distribution: {name}")


def timed(call):
    before = time.perf_counter()
    result = call()
    return result, (time.perf_counter() - before) * 1000


def compare(core_options, core_action, mlx_logits, mlx_actions, temperature):
    k = len(core_options)
    reference = softmax(np.asarray(mlx_logits)[0, :k], temperature)
    action = softmax(np.asarray(mlx_actions)[0])
    return {"option_probabilities": reference.tolist(), "action_probabilities": action.tolist(),
            "option_max_abs_diff": float(np.max(np.abs(core_options - reference))),
            "action_max_abs_diff": float(np.max(np.abs(core_action - action))),
            "option_argmax_agreement": int(np.argmax(core_options)) == int(np.argmax(reference)),
            "action_argmax_agreement": int(np.argmax(core_action)) == int(np.argmax(action))}


def run(args):
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise RuntimeError("This live probe requires macOS on Apple Silicon")
    if args.laya_mlx_path:
        sys.path.insert(0, str(args.laya_mlx_path.resolve()))
    import coremltools as ct
    from tokenizers import Tokenizer
    from laya_mlx.agent import Agent, collate_items
    from laya_mlx.common import build_sequence, temp_bucket

    cfg = json.loads((args.coreml_dir / "config.json").read_text())
    source_cfg = json.loads(args.source_config.read_text())
    mlx_cfg = json.loads((args.mlx_model_dir / "rl_agent_config.json").read_text())
    if source_cfg != mlx_cfg:
        raise ValueError("Source rl_agent_config and MLX config differ; cannot assume calibration parity")
    if cfg["head_max_len"] != source_cfg["head_max_len"] or cfg["max_options"] != 32:
        raise ValueError("Model rendering budgets differ from the source checkpoint")
    bucket = next(b for b in cfg["buckets"] if b["length"] == args.bucket)
    bundle = args.coreml_dir / bucket["fp16"]
    signature = inspect_mil(bundle, args.bucket)
    if not (bundle / "weights/weight.bin").is_file():
        raise ValueError("Compiled bucket weights are missing")
    compute = {"cpu_ne": ct.ComputeUnit.CPU_AND_NE, "all": ct.ComputeUnit.ALL,
               "cpu": ct.ComputeUnit.CPU_ONLY, "cpu_gpu": ct.ComputeUnit.CPU_AND_GPU}[args.compute_units]
    load_log = []

    def log_load(backend, event, **details):
        entry = {"at": datetime.now(timezone.utc).isoformat(), "backend": backend,
                 "event": event, **details}
        load_log.append(entry)
        print(json.dumps(entry), flush=True)

    log_load("coreml", "load_started", bucket=args.bucket, compute_units=args.compute_units)
    model, coreml_load_ms = timed(lambda: ct.models.CompiledMLModel(str(bundle), compute_units=compute))
    log_load("coreml", "load_completed", elapsed_ms=coreml_load_ms)
    log_load("mlx", "load_started", dtype=args.mlx_dtype, device="gpu")
    agent, mlx_load_ms = timed(lambda: Agent(args.mlx_model_dir, dtype=args.mlx_dtype, device="gpu"))
    log_load("mlx", "load_completed", elapsed_ms=mlx_load_ms)
    if "torch" in sys.modules:
        raise RuntimeError("Probe must not import PyTorch")
    fluid_tok = Tokenizer.from_file(str(args.coreml_dir / "tokenizer.json"))
    fluid_tok.no_padding()
    fluid_tok.no_truncation()

    class FluidTokenizer:
        def __call__(self, value, add_special_tokens=False):
            return {"input_ids": fluid_tok.encode(value, add_special_tokens=add_special_tokens).ids}

    tokenizer = FluidTokenizer()
    for name in ("cls_token", "sep_token", "pad_token", "mask_token"):
        value = getattr(agent.tok, name)
        token_id = fluid_tok.token_to_id(value)
        if token_id != getattr(agent.tok, name + "_id"):
            raise ValueError(f"Tokenizer special id mismatch: {name}")
        setattr(tokenizer, name, value)
        setattr(tokenizer, name + "_id", token_id)
    cases = json.loads(args.cases.read_text()) if args.cases else CASES
    if not isinstance(cases, list) or not cases:
        raise ValueError("Cases must be a nonempty list")
    try:
        chip = subprocess.check_output(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        chip = platform.machine()
    report = {
        "purpose": "Smoke parity evidence; not an accuracy benchmark or backend promotion approval",
        "load_log": load_log,
        "platform": {"chip": chip, "macos": platform.mac_ver()[0], "python": platform.python_version()},
        "versions": {p: importlib.metadata.version(p) for p in ("coremltools", "mlx", "numpy", "tokenizers")},
        "coreml": {"snapshot": args.coreml_dir.name, "source": cfg["source"], "bucket": args.bucket,
                   "compute_units_requested": args.compute_units,
                   "placement_note": "Allowed compute units, not proof of exclusive ANE execution",
                   "signature": signature, "weight_sha256": digest(bundle / "weights/weight.bin"),
                   "load_ms": coreml_load_ms},
        "mlx": {"snapshot": args.mlx_model_dir.name, "dtype": args.mlx_dtype, "load_ms": mlx_load_ms,
                "provenance": json.loads((args.mlx_model_dir / "mlx_config.json").read_text()),
                "weight_sha256": digest(args.mlx_model_dir / "model.safetensors")},
        "calibration": {"source_config_sha256": digest(args.source_config),
                        "temperature": source_cfg["temperature"],
                        "temperature_by_options": source_cfg.get("temperature_by_options", {}),
                        "source_config_equals_mlx_config": True},
        "thresholds": {"option_max_abs_diff": args.option_tolerance, "action_max_abs_diff": args.action_tolerance},
        "renderer": {"common_sha256": digest(Path(sys.modules[build_sequence.__module__].__file__)),
                     "agent_sha256": digest(Path(sys.modules[Agent.__module__].__file__)),
                     "coreml_tokenizer_sha256": digest(args.coreml_dir / "tokenizer.json")},
        "cases": [],
    }
    for case in cases:
        items, internal = agent.prepare(case["state"], {"q": case["question"]})
        item, question = items[0], internal[0]
        # Independently render with the Fluid tokenizer; then compare token ids
        # and marker positions before either backend consumes the prompt.
        ids, markers = build_sequence(tokenizer, case["state"], question,
                                      source_cfg["max_len"], source_cfg["head_max_len"])
        if ids != item["ids"] or markers != item["markers"]:
            raise ValueError(f"Tokenizer/render parity failed for {case['id']}")
        # Also detect source-max truncation, not only overflow of this bucket.
        unbounded, _ = build_sequence(tokenizer, case["state"], question, 10**9, source_cfg["head_max_len"])
        if ids != unbounded:
            raise ValueError("Source renderer would truncate state; excluded from this probe")
        inputs = build_inputs(item, args.bucket, tokenizer.pad_token_id, tokenizer.mask_token_id)
        natural = collate_items([item], tokenizer.pad_token_id)
        padded = collate_items([item], tokenizer.pad_token_id, pad_to_multiple=args.bucket, max_length=args.bucket)
        if not np.array_equal(padded["input_ids"], inputs["input_ids"]) or not np.array_equal(padded["attention_mask"], inputs["attention_mask"]):
            raise ValueError("Padded backend inputs differ")
        core_first, first_ms = timed(lambda: model.predict(inputs))
        validate_outputs(core_first)
        core, warm_ms = timed(lambda: model.predict(inputs))
        validate_outputs(core)
        k, qt = len(markers), item["qtype"]
        temperature = source_cfg.get("temperature_by_options", {}).get(temp_bucket(qt, k), source_cfg["temperature"][qt])
        probabilities = softmax(core["logits"][0, :k], float(temperature))
        action = core["action_probabilities"][0].astype(np.float64)
        padded_output, padded_ms = timed(lambda: agent.forward(padded))
        natural_output, natural_ms = timed(lambda: agent.forward(natural))
        comparisons = {"mlx_bucket_padded": compare(probabilities, action, *padded_output, temperature),
                       "mlx_natural_length": compare(probabilities, action, *natural_output, temperature)}
        ok = all(v["option_argmax_agreement"] and v["action_argmax_agreement"]
                 and v["option_max_abs_diff"] <= args.option_tolerance
                 and v["action_max_abs_diff"] <= args.action_tolerance for v in comparisons.values())
        report["cases"].append({
            "id": case["id"], "question_type": question["t"], "token_count": len(ids),
            "marker_positions": markers, "option_count": k, "token_ids_equal": True,
            "prompt": {"request": {"state": case["state"], "questions": {"q": case["question"]}},
                       "normalized_question": question, "token_ids": ids,
                       "tokens": [fluid_tok.id_to_token(token) for token in ids],
                       "decoded_with_special_tokens": fluid_tok.decode(ids, skip_special_tokens=False)},
            "token_ids_sha256": hashlib.sha256(np.asarray(ids, dtype="<i4").tobytes()).hexdigest(),
            "temperature": temperature,
            "coreml": {"option_probabilities": probabilities.tolist(), "action_probabilities": action.tolist(),
                       "raw_option_logits": core["logits"][0, :k].tolist(),
                       "builtin_probability_vs_uncalibrated_softmax_max_diff": float(np.max(np.abs(core["probabilities"][0, :k] - softmax(core["logits"][0, :k])))),
                       "first_predict_ms": first_ms, "second_predict_ms": warm_ms},
            **comparisons, "mlx_padded_ms": padded_ms, "mlx_natural_ms": natural_ms, "passed": ok,
        })
        # Persist each completed case so interruption cannot erase collected data.
        report["complete"] = len(report["cases"]) == len(cases)
        report["passed"] = report["complete"] and all(row["passed"] for row in report["cases"])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"case": case["id"], "passed": ok, "comparisons": comparisons}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coreml-dir", type=Path, required=True)
    parser.add_argument("--mlx-model-dir", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--laya-mlx-path", type=Path)
    parser.add_argument("--bucket", type=int, choices=(128, 256, 512, 1024), default=128)
    parser.add_argument("--compute-units", choices=("cpu_ne", "all", "cpu", "cpu_gpu"), default="cpu_ne")
    parser.add_argument("--mlx-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--option-tolerance", type=float, default=.02)
    parser.add_argument("--action-tolerance", type=float, default=.02)
    parser.add_argument("--cases", type=Path, help="Optional JSON list of {id,state,question}")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(not math.isfinite(t) or t < 0 for t in (args.option_tolerance, args.action_tolerance)):
        parser.error("Tolerances must be finite and nonnegative")
    return 0 if run(args)["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
