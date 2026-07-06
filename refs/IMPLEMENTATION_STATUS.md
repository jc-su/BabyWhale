# Baby Whale v4 Implementation Status

Status date: 2026-05-13

This document is the working checklist for the full Baby Whale v4 plan. A feature is only `done` when it has:

- a real implementation in `baby_whale_v4/`.
- a focused unit test or smoke command.
- fail-fast validation for invalid modes.
- unsupported paths fail before use.

## Status Legend

- `done`: implemented and tested in the local educational stack.
- `prototype`: implemented enough to teach the idea, but missing scale, benchmark, or production behavior.
- `perf-gated`: implemented, but local use must pass the Mac benchmark gate.
- `missing`: not implemented yet.
- `blocked`: needs an external kernel, dataset, or design decision.

## Architecture And Runtime

| Feature | Status | Current state | Next gate |
| --- | --- | --- | --- |
| Project rename to Baby Whale v4 | done | Package, CLI, docs use `baby_whale_v4`. | Keep old names out of runtime and docs. |
| MLX backend + runtimes | done | `Backend = Literal["mlx"]`; `MLXRuntime = Literal["mlx-metal", "mlx-cuda"]`; `torch` is rejected. Runtime selection fails before model creation. | Add Linux/NVIDIA CI when available. |
| Python 3.14 + uv + Ruff + ty | done | `pyproject.toml`, `uv.lock`, local gates pass. | Keep typed tests clean. |
| Fail-fast config validation | done | `BabyWhaleV4Config` rejects invalid closed sets and cross-field invariants. | Add validation tests for every new field. |
| Checkpoint format | prototype | `.bw4` pickle payload with config hash and MLX arrays; corruption/tamper tests cover truncation, garbage, bad hash/version, missing/unknown keys, non-array state. | Add model portability. Exact RNG-*state* restore is blocked by MLX's API (round-trip of `mx.random.state` doesn't reproduce); resume is seed-deterministic. |
| Dataset ingestion | prototype | Hugging Face subset materialization writes normalized JSONL plus a manifest. Normalized JSONL can now be packed into token blocks and saved with tokenizer-hash manifests. | Add source-specific adapters and larger packed-token streaming. |
| Tokenizer training | prototype | Byte tokenizer remains for tests; educational byte-BPE training, save/load, and stable tokenizer hashes exist. Encode is now heap-based O(len·log len) (was O(len·n_merges) and hung packing long code lines), verified output-identical to the ordered-merge reference. | Add tokenizer migration checks, faster training, and optional SentencePiece comparison. |

## Model

| Feature | Status | Current state | Next gate |
| --- | --- | --- | --- |
| RMSNorm | done | MLX layer implemented and covered by forward tests. | None. |
| Partial RoPE | done | MLX rotary path implemented. | Add long-context frequency scaling tests if context grows. |
| Sliding MQA | done | Sliding causal MQA with dynamic cache. | Add latency/memory benchmarks. |
| Dynamic KV cache | done | MLX cache append/clone and cache decode parity tests. | Add offload variant. |
| Sparse MoE | done | Hash-routed bootstrap, learned top-k layers, expert-utilization metrics, and load-balance reports exist. | Add load-balancing auxiliary loss into training. |
| Shared expert | done | Shared expert path exists in MoE. | Add ablation metric. |
| HCA | prototype | Mean-pooled compressed block attention exists. | Add explicit compressed cache object and memory-reduction benchmark. |
| CSA | prototype | Overlapping compressed blocks and top-k indexer exist. | Add learned indexer training/eval and long retrieval benchmark. |
| mHC | prototype | Multi-head correction/mixing path exists. | Add Sinkhorn-normalized mixing and parity tests for `hc_mult=1`. |
| MTP | prototype | MTP heads, speculative decode, and Engine speculative routing (`generate(mode="speculative")`) exist; the per-step decode loop fails fast on speculative. | Add acceptance-rate benchmark and verifier/reject parity tests at larger sizes. |
| DeepSeek-style layer schedule | prototype | Per-layer attention schedule exists. | Add schedule search configs and tests for mixed HCA/CSA/sliding stacks. |

## Training Lifecycle

| Feature | Status | Current state | Next gate |
| --- | --- | --- | --- |
| Pretrain | prototype | MLX loop, token-weighted gradient accumulation, JSONL packed dataset input, train/eval metrics, token throughput, checkpoint save/resume, synthetic tests. A **real BPE-tokenized run** (5.56M params, 2048-vocab BPE, **15.2M tokens** of local code) descended train loss 8.12→4.46, held-out eval 4.89, ~10.5K tok/s, ~1.8 GB peak bf16; a companion byte-level run also verified optimizer+scheduler-state resume from a mid-run checkpoint. | Longer runs / larger models; per-phase context curriculum at scale. |
| Mid-train | prototype | `midtrain.py` reuses the pretrain loop over normalized JSONL mixtures with repeat weights. | Add code/math/agent source adapters and mixture reports. |
| SFT | prototype | MLX SFT loop, assistant-only chat masking, and normalized chat/tool-trace JSONL adapter exist. | Add assistant-token-only loss audits and real tool/no-tool dataset mix. |
| DPO | prototype | DPO loss, toy loop, and normalized preference JSONL adapter exist; the frozen reference's log-ratios are precomputed once (not re-run each step) — numerically identical, ~half the forwards. | Add chosen/rejected sanity reports and held-out preference eval. |
| Rejection fine-tuning | prototype | Sample/rank collection helper exists. | Add end-to-end data generation and SFT handoff. |
| GRPO | prototype | Local group sampling, reward function loop, and verifiable arithmetic/tool-use task generator exist. | Add pass-rate CLI and wire rollout records into GRPO updates. |
| On-policy distillation | done | `training/distill.py` with KL loss between frozen teacher and student logits, reward-filtered student-on-policy rollouts, `distill` CLI subcommand. Teacher weights are byte-equal before/after; KL decreases against a diverged teacher in tests. | Add multi-teacher routing for the V4 specialist consolidation step. |
| LoRA/adapters | prototype | MLX LoRA adapters can attach to selected `WhaleLinear` placements such as attention and MoE experts. | Add base-weight freezing and SFT/DPO CLI flags. |
| Muon-style optimizer | prototype | Educational Muon-style optimizer with Newton-Schulz matrix updates exists and is accepted by pretrain/midtrain. | Benchmark against AdamW/Adafactor on Mac shapes. |
| Gradient accumulation | done | MLX microbatch accumulation uses non-ignored target-token weighting and has full-batch parity tests. | Add larger BF16 memory/throughput sweeps. |

## Inference And Serving

| Feature | Status | Current state | Next gate |
| --- | --- | --- | --- |
| Prefill/decode split | done | `Engine.prefill_chunk` and `decode_step`. | Add timing stats per phase. |
| Prefix cache | done | Config/tokenizer/quant-aware keys, reuse tests. | Add cache hit-rate metrics. |
| Chunked prefill | prototype | Round-robin scheduler prevents simple decode starvation. | Add mixed-workload latency benchmark. |
| Request scheduler | done | Chunked-prefill scheduler backing the HTTP server via `inference/serving.py` (`BatchingServer`); prefills queued requests in lock-step and groups same-length/same-sampling decodes into one **batched forward per cohort** (`decode_group_batched`, greedy-parity-tested vs per-request for sliding + HCA/CSA layers). **Ragged (mixed-length) batching** via `decode_ragged_batched` + `RequestScheduler(ragged=True)` — per-row RoPE + causal/sliding key masks, greedy-parity-tested; requires an all-sliding_mqa model. | HCA/CSA/MLA **intentionally** keep the same-length cohort path (correct for every schedule); ragged's compressed/latent masks aren't worth the complexity for the educational goal. |
| Paged KV cache | done | `PagedKVPool`/`PagedKVCache` are a wired Engine storage backend (`Engine(paged_pool=...)`); the block map is shared across layers, decode is token-identical to the dense cache (parity test), and it fails fast on MLA and dim mismatch. | Add cross-request prefix block sharing and a fused gather kernel. |
| KV cache offload | done | `Engine.offload_request` / `reload_request` snapshot a request's `DynamicKVCache` to NPZ and resume decode identically (round-trip test). | Add unified-memory eviction policy and a latency/memory report. |
| Compressed cache offload | missing | Not implemented. | Add HCA/CSA cache storage and eviction policies. |
| HTTP server | prototype | `ThreadingHTTPServer` backed by a background **continuous-batching** loop (one scheduler thread owns the model; handler threads submit + wait). `/health`, `/generate`, strict `/v1/chat/completions` with incremental SSE streaming and a real `finish_reason`; `/sync_weights` runs as a loop-thread control action. **SSE-disconnect cancellation** (`RequestHandle.cancel()` → the loop drops the request, tested) and OpenAI-shaped error objects on the chat surface. | Add per-request usage accounting. |
| Tool execution runtime | prototype | Strict tool-call parser plus deterministic calculator/string/calendar registry. | Add sandboxed code/table/project-doc tools and transcript execution loop. |
| Agent skills runtime | prototype | Single-process local rollout/reward runner exists for tool-use tasks. | Wire runner outputs into GRPO updates and add multi-turn tasks. |

## Quantization And Precision

| Feature | Status | Current state | Next gate |
| --- | --- | --- | --- |
| int8 weight-only | done | Native MLX affine quantized matmul and model application tests. | Add benchmarks. |
| int4 weight-only | done | Native MLX affine quantized matmul and model application tests. | Add benchmarks. |
| MLX FP4 native | done | `mxfp4`/`nvfp4` pack, dequantize, and `quantized_matmul` tests. | Cache packed weights instead of repacking every forward. |
| `fp4-native` model path | done | `WhaleLinear` routes through MLX `quantized_matmul`. | Add packed-weight persistence and benchmark. |
| `fp4-expert` export/inference | done | Placement-aware policy maps only MoE expert linears to MLX native FP4; attention, router, MTP, and LM head stay dense. Training fails fast. | Add checkpoint export metadata and load-time validation. |
| FP4 training primitives | experimental | Raw MLX `quantized_matmul` has no weight VJP. Old full-model FP4 training config modes were removed; custom-VJP and Metal weight-gradient experiments now live only behind primitive benchmarks. | Keep as research code unless it beats dense bf16/fp32 latency and memory. |
| Memory-efficient training | prototype | Activation checkpointing, microbatch gradient accumulation, BF16 dense training, and Adafactor are implemented. The DeepSeek-aligned route is BF16 training plus `fp4-expert` export/inference. | Add larger-shape sweeps and fused optimizer/update kernels. |
| KV int8 | prototype | Per-token quantization round-trip and cache decode tests. | Add memory/latency benchmark and error report. |
| FP8 | reference-only | No public `fp8-sim` mode. DeepSeek-V4's FP4-to-FP8 compute path requires an optimized FP8 stack; Baby Whale fails fast instead of exposing a slow emulation path. | Revisit only when MLX exposes native FP8 primitives with measurable speed or memory wins. |

## Evaluation And Benchmarks

| Feature | Status | Current state | Next gate |
| --- | --- | --- | --- |
| Unit tests | done | 294 local tests pass. | Keep tests narrow and fail-fast. |
| Smoke CLI | done | Tiny forward and hybrid smoke commands. | Add pretrain/SFT smoke on local JSONL. |
| Pretraining eval | prototype | Pretrain logs train loss, validation loss, token counts, and tokens/sec. | Add perplexity reporting and held-out packed dataset command. |
| Long-context eval | prototype | `baby_whale_v4/eval/needle.py` — synthetic needle retrieval with **per-sample varying answers** (true retrieval, not the fixed-answer bigram of `SyntheticNeedleDataset`); scoring + training-sensitivity tested. | Sweep needle depth × attention schedule to chart the sliding-vs-HCA/CSA reach curve. |
| Tool-use eval | prototype | Tool reward reports JSON validity, tool existence, arg match, execution, answer match, and generated arithmetic tool tasks. | Add held-out benchmark files and exact-match CLI. |
| Agent eval | missing | No multi-turn environment. | Add small tau-bench-like local domain. |
| Quantization benchmark | prototype | `bench-fp4-training` compares dense, custom VJP, and optional Metal weight-gradient paths. `bench-inference` covers chunked prefill, prefix cache, and quantized inference; `bench-compare` runs one prompt suite across no-cache / prefix-cache / paged / quant configs. | Add layer-stack and end-to-end model latency/memory reports. |

## Multi-Modal (DeepSeek-VL2 recipe)

DeepSeek-V4 itself is text-only (multi-modal generation deferred); the canonical reference for this section is the DeepSeek-VL2 paper. All entries below are gated behind `BabyWhaleV4Config.enable_vision = False` by default — text-only behavior must remain bit-identical when the flag is off.

| Feature | Status | Current state | Next gate |
| --- | --- | --- | --- |
| Vision encoder (SigLIP-SO400M-384 MLX port) | missing | Not implemented. | Add `vision/encoder.py`; ship pretrained weights, do not train from scratch. |
| Dynamic tiling | done | `vision/tiling.py` `plan_tiles` — picks `(cols, rows)` minimizing padding with `cols*rows ≤ max_tiles`, tie-broken by aspect match, plus a global thumbnail. Pure geometry, tested. | — |
| Vision MLP connector | done | `vision/connector.py` `VisionMLPConnector` — 2-layer `WhaleLinear` MLP projecting `vision_dim → n_embd`, tested. | Wire into the model once the encoder lands. |
| Model integration | prototype | `BabyWhaleV4Model.__call__(image_features=...)` projects tile features through the connector and prepends them to the token stream (image-tokens-first VL2 layout); text-only path bit-identical when off; MoE hash-routing gets placeholder ids for image positions. Inference-only, tested. | Add the loss + KV-cache path so images work in training and streaming decode (the next VL milestone). |
| Config fields | done | `enable_vision`, `vision_encoder_kind`, `vision_tile_size`, `vision_max_tiles`, `vision_dim`, `vision_dropout` with fail-fast validation when enabled. **config_hash excludes them when disabled**, so pre-vision checkpoints still load (verified against a real checkpoint). | — |
| Stage 1 — VL alignment | missing | Not implemented. | `training/vl_align.py`: freeze LLM, train connector + vision encoder. Acceptance: connector overfits a synthetic caption set, LLM weights byte-identical. |
| Stage 2 — VL pretraining | missing | Not implemented. | `training/vl_pretrain.py`: 70% VL / 30% text mix, full unfreeze, text-token-only loss. Acceptance: text-only val PPL within 5% of midtrain baseline. |
| Stage 3 — VL SFT | missing | Not implemented. | `training/vl_sft.py`: extend chat template with `<image>` placeholder, assistant-only loss masking. |
| Rollout integration | missing | `RolloutRequest` is text-only. | Add `image_tiles: tuple[mx.array, ...] = ()` field (default empty for backward compat); thread through `Engine.new_request` and `/rollout` HTTP body. |
| Quantization placement | missing | No vision-specific placements. | Extend `LinearPlacement` with `"vision_encoder"` and `"vision_connector"`; update `fp4-expert` policy to skip vision encoder by default. |
| CLI | missing | No vision commands. | Add `vl-align`, `vl-pretrain`, `vl-sft`, `smoke-vl` subcommands. |
| Per-attention-kind parity | missing | No vision tests. | Verify tile features feed correctly through sliding / HCA / CSA / MLA. |

## Next Hard Milestones

1. **Mac pretrain run**: train on a real 5M-50M token subset and report loss, eval loss, throughput, memory, and checkpoint resume.
2. **Agent/tool SFT**: add negative no-tool examples, tool transcript execution, and SFT/DPO CLI commands.
3. **Verifiable GRPO**: add pass-rate CLI and wire local rollout records into GRPO updates.
4. **Inference benchmark expansion**: measure prefix cache, chunked prefill, native FP4, and KV int8 on the same prompt suite.
5. **Checkpoint hardening**: add exact RNG restore and more corruption tests.
6. **HCA/CSA metrics**: add compressed-cache storage and long-retrieval reports.
7. **Multi-modal Step 8 (VL2 recipe)**: vision encoder + connector + three-stage training. Roadmap entry in `BABY_WHALE_V4_ROADMAP.md` Step 8 and detailed design in `BABY_WHALE_V4_PLAN.md` Milestone 9.
