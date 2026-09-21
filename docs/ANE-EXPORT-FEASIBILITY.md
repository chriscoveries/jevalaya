# ANE export coverage feasibility — T025

Reviewed 2026-09-22 against local `laya-coreml` commit `12b7501583c7f03a6b2e49ebe118a2c6302505b9` at `/private/tmp/laya-coreml`, installed checkpoint metadata, and the current jevalaya Rust adapter. This is a source-and-artifact-report feasibility assessment. No new models were exported, downloaded, or hardware-benchmarked in this review.

## Recommendation

Prioritize an **English L96 FP16 export**, then multilingual L192 and L256. Treat typed-decisions as a separate checkpoint validation using the English architecture, followed by a multi-question service test. Preserve current family exclusions until matching bundles pass validation and the router can select the correct family. An English artifact alone cannot change today's routing coverage.

The foreman's AG News sweep found 6,942/7,600 prompts within 96 tokens but excluded by the English checkpoint gate. That makes English L96 the highest-value next coverage experiment **for that workload**; it is not evidence that 91% of arbitrary consumer traffic will benefit. Recount with the English export's tokenizer, full prompt, and option limit before claiming coverage.

| Candidate | Assessment | Evidence and remaining work |
| --- | --- | --- |
| Multilingual L192 | Strong feasibility; prior upstream validation exists | Retained JSON reports 60/60 fitting fixture answers, 100 stable repeats, and 6,390 ANE-preferred body operations. Regenerate/package and validate this runtime. |
| Multilingual L256 | Feasible from parameterized source; not independently validated here | Same length-parameterized graph has retained L192 and L1024 successes. No L256 report was found; export, compile, parity and benchmark are required. |
| English L96 | Feasible architecture; new checkpoint-specific ANE evidence required | ModernBERT encoder and the same head structure are supported, with dimensions and RoPE read from the source. Wider/deeper than multilingual; compiler placement and latency are unknown. Packaging and family-aware routing require changes. |
| Typed-decisions L96/192/256 | Feasible architecture; separate weights/semantics validation required | Same encoder dimensions as English, different trained weights, calibration, and context/head budgets. Typical workflows contain several questions; changing the family gate does not remove the one-question gate. |

## What already exists

The current published ANE bundle is `laya-coreml-ane` **format v1**, fixed B1/L96/K32. Host embeddings and the action head accompany a Core ML body. Its installed manifest pins multilingual source revision `052592a15d198d9ad47da779604259b10b47b7aa`.

Upstream's retained [L192 validation](https://github.com/mizorewww/laya-coreml/blob/12b7501583c7f03a6b2e49ebe118a2c6302505b9/experiments/ane_engineering/validation192.json) reports 60/60 fitting answers, 100 stable repeated calls, maximum calibrated probability error 0.002925, and short-question p50/p95 8.18/8.58 ms. Its [L1024 validation](https://github.com/mizorewww/laya-coreml/blob/12b7501583c7f03a6b2e49ebe118a2c6302505b9/experiments/ane_engineering/validation1024.json) reports 63/63 and 100 stable repeats. Both plans report 6,390 ANE-preferred body operations. These are historical M3 Max measurements from the Python runtime, not measurements of this Rust service; L192 skips the long fixture. The checkout contains reports/manifests but not these generated model packages.

Thus 96 is an artifact selection, not a demonstrated architectural ceiling. The ~77 ms service latency supplied by the team and upstream's much shorter historical measurements need a same-artifact, same-host, release-build comparison. Time tokenization, host buffers, Core ML, output decoding and queueing separately. The Rust host constructs dense L×L masks and copies arrays through strides; these are profiling targets, not a diagnosed cause. Larger masks can increase host overhead even when ANE placement succeeds.

## The correct export pipeline

`laya-coreml convert ... --fixed --max-length 192` creates ordinary `laya-coreml` v1, **not** `laya-coreml-ane` v1. The Rust ANE adapter rejects the former. The ordinary graph uses a different I/O contract; allowing `CPU_AND_NE` also does not establish that a graph runs on ANE.

Use the source checkout's [ANE probe](https://github.com/mizorewww/laya-coreml/blob/12b7501583c7f03a6b2e49ebe118a2c6302505b9/experiments/ane_engineering/probe.py), which constructs `ConvBody`: channel-first `[B,C,1,L]`, 1×1 projections, per-head attention, fixed RoPE constants and host-prepared masks/marker selectors. Its `--source` and `--length` already vary checkpoint and length. This path creates a **research** package with `manifest.json`; a subsequent packaging step must produce the deployable v1 directory.

The pinned conversion environment is Python 3.12, coremltools 9.0, torch 2.7.0 and NumPy 2.1.3. PyTorch is an offline export dependency; inference remains PyTorch-free. The inspected `/private/tmp/cmtools` environment lacks torch, and `/private/tmp/laya-coreml/.venv` is absent. The normal wheel includes `laya_coreml`, not the research scripts: use the checkout and a dedicated conversion environment.

Start from **original upstream checkpoint weights**, not the installed `aac6fef/*-mlx` exports. `load_model()` strictly loads original PyTorch parameter names; MLX conversion renames attention projections and sequential head keys. It also does not establish equality of all original tensor precisions. Do not assume an MLX weight file is a drop-in export source.

Example commands after preparing that environment and original checkpoint directories (new, absent output directories; no Hub publishing):

```sh
python -m experiments.ane_engineering.probe --source /path/to/original/laya --kind body --length 96 --output artifacts/t025/english-l96
python -m experiments.ane_engineering.validate --source /path/to/original/laya --name laya --package artifacts/t025/english-l96/model.mlpackage --length 96 --repeats 100 --output artifacts/t025/english-l96-validation.json

python -m experiments.ane_engineering.probe --source /path/to/original/laya-multilingual --kind body --length 192 --output artifacts/t025/multilingual-l192
python -m experiments.ane_engineering.validate --source /path/to/original/laya-multilingual --name laya-multilingual --package artifacts/t025/multilingual-l192/model.mlpackage --length 192 --repeats 100 --output artifacts/t025/multilingual-l192-validation.json
```

Repeat L256 with a distinct output path. For typed-decisions use its original directory and `--name laya-typed-decisions`. Validate requested L against the original `max_len` and encoder position limit before invoking the research probe: unlike the ordinary converter, the probe only checks that length is positive. The checked `benchmarks/results/reference.json` has all three model names; ensure source hashes match rather than regenerating expected answers from the candidate.

## Checkpoint differences the exporter must preserve

These values were read from installed encoder/agent metadata; original source configs and weights remain the export authority.

| Property | Multilingual | English | Typed-decisions |
| --- | --- | --- | --- |
| Hidden width / encoder layers / heads | 768 / 22 / 12 | 1024 / 28 / 16 | 1024 / 28 / 16 |
| MLP intermediate width | 1152 | 2624 | 2624 |
| Vocabulary | 256,000 | 50,368 | 50,368 |
| Head dimension | 64 | 64 | 64 |
| Global / local RoPE theta | 160000 / 160000 | 160000 / 10000 | 160000 / 10000 |
| Agent max_len / head_max_len | 1024 / 256 | 512 / 192 | 1024 / 256 |
| Decision-head layers | 2 | 2 | 2 |

`torch_model.py` builds these widths, depths, masks, and RoPE bases from config; `ConvBody` copies the corresponding modules and constants. No multilingual-only width is embedded in the graph builder. English/typed have more encoder and head compute despite a smaller host embedding table. This supports export feasibility, not a latency or ANE-placement guarantee. Preserve each checkpoint's tokenizer, all calibration buckets, learned type vectors and action-head weights. A shared English tokenizer does not make English and typed weights interchangeable.

## Packaging changes needed, same format v1

[`scripts/prepare_hub.py`](https://github.com/mizorewww/laya-coreml/blob/12b7501583c7f03a6b2e49ebe118a2c6302505b9/scripts/prepare_hub.py) is a release recipe rather than a generic ANE packaging CLI. Its ANE branch hardcodes multilingual, `body96`/`body96-w8km`, length 96, source revision and output names. It first expects unrelated standard-release artifacts. Its generated ANE model-card claims are also hardcoded to the old L96 validation. Running it unchanged does not package new candidates correctly.

Parameterize a single-bundle packager around its verified steps: validate the research source/package hashes; copy the ML package, source encoder/agent configs and tokenizer; extract the six host tensors (embedding table, type table, two linear layers' weights/biases); check each tensor exactly against source; write actual source/revision, fixed shape, FP16 body/FP32 host-action precision, package SHA and per-file hashes. Attach candidate-specific validation evidence and generate accurate documentation. Keep `format="laya-coreml-ane"`, `format_version=1`, B1, fixed min=max=L and K32. There is no evident need for a format bump solely for length or width.

K32 remains a separate capacity limit: a 150-option question will not fit merely because L grows. Do not change `rl_agent_config.max_len` or head budgets to force a short export; the export's fixed L and the original rendering semantics are different constraints. Use separate immutable directories per checkpoint/length and never just edit an L96 manifest to claim L192.

## Rust/service integration work

The low-level Rust adapter is largely shape-driven already: `bundle.rs` accepts fixed B1/K>0 v1, `native.rs` checks `[1,W,1,L]` embeddings, L×L masks, `[1,W,1,1]` type vectors and `[1,L,1,K]` selectors, and `host.rs::ActHead` reads dimensions from tensors. Comments mention 768/772 but implementation is not limited to those widths. Check cross-tensor width agreement explicitly and validate both outputs for every candidate.

The service policy is the blocker for English/typed: `router-core/src/policy.rs` rejects typed checkpoints/workflow signatures and all non-multilingual checkpoints before ANE token counting. The prompt engine has one optional ANE tokenizer/budget and config has one ANE model. Removing an exclusion alone could send English tokens to a multilingual graph.

For the first English trial, use a separately configured single-family test instance with explicit family binding. Production extension needs a capability record keyed by **checkpoint, source revision and fixed length**, with the matching tokenizer and calibration. Select a validated bucket for the chosen checkpoint and count against its actual shape/K. Retain the prior exclusions for any family lacking a validated configured bundle. Preserve one resident ANE model until lifecycle policy is deliberately expanded; bucket/family switches then have measurable load/eviction costs. Multiple resident variants would be a new memory/lifecycle decision.

For one longer multilingual bundle, set the policy length from its validated manifest (or clamp explicit `max_tokens` to that length) instead of leaving the 96 default or permitting a larger override. Add manifest/policy consistency checks. For typed workflows, T023's per-question execution is a separate prerequisite for full-request ANE coverage; aggregate latency grows with question count. An export is not permission to split or truncate state.

## Promotion gates and sequence

1. Reproduce English L96 first. Check original source SHA, tokenizer IDs, all three output types, calibration/action probabilities, option counts and mask injection. Run original-vs-export parity and the Rust host against the Python ANE runtime on identical prepared inputs.
2. Require all fixture choices/argmax to match, bounded calibrated/action probability drift (upstream gate ≤0.02), finite outputs, unchanged token accounting and 100 stable repeats. Add non-saturated, boundary and real labeled examples; old fixture agreement alone is not accuracy evidence. Test L−1/L/L+1 and K=32/K=33 rejection.
3. Record compile/load, Core ML anticipated placement, warm p50/p95/p99 and host/body/decode breakdown in a release build. `CPU_AND_NE` permits CPU; a successful load is not sufficient ANE evidence. Compare with MLX for the same checkpoint/input/precision and measure energy if ANE efficiency is the motivation.
4. Produce a portable v1 bundle with correct provenance and repeat validation after packaging. Only then enable its family capability in an isolated service smoke and labeled coverage sweep.
5. Repeat multilingual L192/L256, choosing the smallest useful length by measured coverage and latency. Reuse the English conversion mechanism for typed-decisions but validate its own weights and workflows independently.

Larger fixed exports pad every request to their capacity; masks scale with L² (L192 versus L96: 4× mask entries; L256: about 7.11×), while several projection costs scale with L. These are scaling considerations, not runtime predictions. Do not extrapolate the current 77 ms by a single factor or assume upstream ~8 ms will carry over.

Upstream's [engineering report](https://github.com/mizorewww/laya-coreml/blob/12b7501583c7f03a6b2e49ebe118a2c6302505b9/docs/ANE_ENGINEERING.md) explains the BC1S implementation and evidence limits. Apple's [ANE Transformer guidance](https://machinelearning.apple.com/research/neural-engine-transformers) motivates the layout; its [shape documentation](https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html) distinguishes fixed/enumerated/range inputs. Neither substitutes for per-artifact placement and performance validation. Keep fixed v1 bundles for this experiment; a flexible multi-bucket graph is a separate export and runtime contract.
