# Baby Whale v4 Implementation Roadmap

Status: started 2026-05-08  
Purpose: keep the project educational while building a small DeepSeek-V4-inspired stack.

## Principle

Do not turn the project into a giant framework in one pass. Build the top-level `baby_whale_v4/` package in small runnable slices.

Every milestone must have:

- a clear teaching goal.
- a runnable command or test.
- fail-fast validation for unsupported modes.
- a small benchmark or correctness check before the next milestone starts.

## Step 0: Reference And Naming

Done:

- keep the DeepSeek-centered plan in `refs/BABY_WHALE_V4_PLAN.md`.
- use `baby_whale_v4` as the top-level code package name.
- use `BabyWhaleV4` as the model/config class prefix.

## Step 1: Small Runnable Core

Goal: create the smallest teachable Baby Whale model that can run on the laptop.

Deliver:

- strict `BabyWhaleV4Config`.
- RMSNorm.
- partial RoPE.
- sliding-window MQA.
- dynamic KV cache.
- sparse MoE with hash-routed bootstrap layers and learned top-k layers.
- tiny `BabyWhaleV4Model`.
- CLI smoke test.
- unit tests for config, forward pass, and cache parity.

This step intentionally does not include HCA, CSA, mHC > 1, MTP, SFT, DPO, or GRPO. Those come after the MLX core is testable.

## Step 2: Data And Pretraining

Goal: make a tiny pretraining run educational and reproducible.

Deliver:

- packed token dataset and normalized JSONL packing command.
- simple local tokenizer path plus educational byte-BPE training and hash checks.
- synthetic dataset for tests.
- pretraining loop with token-weighted gradient accumulation.
- JSONL metrics with train/eval loss and token throughput.
- checkpoint save/resume.

Acceptance:

- tiny model overfits synthetic data.
- normalized JSONL can train through `pretrain` and `midtrain`.
- resume restores step, optimizer, scheduler, and config hash.

## Step 3: Long-Context Attention

Goal: add the Baby Whale version of compressed attention.

Deliver:

- HCA compressed block cache.
- CSA compressed sparse indexer.
- debug dense modes for correctness.
- long-context synthetic retrieval eval.

Acceptance:

- HCA uses less cache memory than raw KV.
- CSA improves synthetic retrieval over sliding-only.
- cache decode parity passes on small inputs.

## Step 4: mHC And MTP

Goal: introduce the more DeepSeek-V4-specific training/inference ideas.

Deliver:

- mHC with `hc_mult=2`.
- Sinkhorn-normalized stream mixing.
- MTP auxiliary head.
- speculative decode using MTP drafts.

Acceptance:

- `hc_mult=1` matches the old residual path.
- disabling MTP leaves main logits unchanged.
- speculative decode accepts only tokens verified by normal decode.

## Step 5: Post-Training

Goal: make the lifecycle real.

Deliver:

- SFT with a Baby Whale chat/tool template.
- DPO for chosen/rejected pairs.
- rejection fine-tuning from sampled completions.
- GRPO on verifiable math/code/tool-call rewards.

Acceptance:

- SFT overfits a tiny chat set.
- DPO loss matches hand-computed toy logits.
- GRPO improves a toy reward.

## Step 6: Inference Infrastructure

Goal: teach serving concepts without pretending to be vLLM.

Deliver:

- prefill/decode split.
- prefix cache.
- chunked prefill.
- cache eviction.
- request scheduler.
- local HTTP server.

Acceptance:

- prefix reuse reduces TTFT.
- chunked prefill avoids decode starvation in mixed workloads.
- all cache keys include model/tokenizer/backend/precision/config hashes.

## Step 7: Mac Optimization

Goal: make the Mac path first-class.

Deliver:

- MLX-only model, cache, training, and inference paths.
- int4/int8 weight-only inference.
- int8 KV/cache experiments.
- no FP4 emulation model mode; FP4 model paths must route through native MLX primitives.
- mandatory MLX FP4 primitive path for `mxfp4` and `nvfp4` weight packing and linear matmul.
- DeepSeek-style `fp4-expert` policy that applies native FP4 only to MoE expert linears after dense training.
- `fp4-native` fail-fast gate unless MLX `quantized_matmul` is actually used.
- primitive-only FP4 custom-VJP and Metal experiments that do not appear as model config modes.
- activation checkpointing, gradient accumulation, and Adafactor for memory-constrained BF16 training on Mac.

Acceptance:

- MLX FP4 primitives pass pack, dequantize, and quantized matmul tests in the default environment.
- `quant_mode="fp4-expert"` quantizes only MoE experts and rejects training.
- `quant_mode="fp4-native"` routes Baby Whale linear layers through MLX `quantized_matmul`.
- full-FP4 training experiments are reachable only through explicit primitive benchmarks.
- unsupported precision and kernel requests fail before model creation.

## Step 8: Vision-Language Extension (VL2-style)

Goal: extend the text-only model to images using DeepSeek-VL2's three-stage recipe. DeepSeek-V4 itself is text-only — multi-modal generation was deferred — so the canonical reference for this step is the DeepSeek-VL2 paper (arxiv 2412.10302), not V4.

The existing text-only path must not regress. The VL extension is opt-in via config; when `enable_vision` is false, all current behavior is bit-identical.

### Architecture additions

- `vision/encoder.py`: SigLIP-style ViT encoder (educational MLX port; `siglip-so400m-patch14-384` is the V2 reference).
- `vision/tiling.py`: dynamic tiling — split a high-res image into `384×384` tiles with `(m, n) ∈ argmin pad subject to mn ≤ vision_max_tiles` plus one global thumbnail.
- `vision/connector.py`: `VisionMLPConnector` (2-layer MLP) projects tile features into the LLM's `n_embd`.
- `model.py` extension: `BabyWhaleV4Model.__call__` accepts an optional `image_tiles` argument; when present, tile features are prepended to the token embedding stream before block dispatch. The existing attention/MoE path runs unchanged.

### Config additions

New fields on `BabyWhaleV4Config` (all optional, default off):

- `enable_vision: bool = False`
- `vision_encoder_kind: Literal["siglip"] = "siglip"`
- `vision_tile_size: int = 384`
- `vision_max_tiles: int = 9`
- `vision_dim: int = 1152`  *(SigLIP-SO400M output dim)*
- `vision_dropout: float = 0.0`

Validation: when `enable_vision=False`, all vision fields ignored (no encoder constructed). When `enable_vision=True`, fail-fast if `vision_dim <= 0` or `vision_max_tiles < 1`.

### Three training stages

Mirror VL2 exactly. Each stage gets its own training entry point alongside the existing `pretrain.py` / `sft.py`:

- `training/vl_align.py` — Stage 1. **LLM frozen.** Train: vision encoder + MLP connector. Loss: next-token CE on text. Data: small VL caption set (~1.2M samples scale at full size; tiny synthetic for tests). Acceptance: connector loss decreases on synthetic image-caption pairs; LLM weights unchanged after the run.
- `training/vl_pretrain.py` — Stage 2. **Everything trainable.** Data mix: 70% interleaved/captioned VL JSONL + 30% text from existing pretrain corpus. Loss: next-token CE on text tokens only (vision tokens never contribute to the loss). Acceptance: held-out VQA loss < text-only baseline; text-only validation perplexity does not degrade by more than X%.
- `training/vl_sft.py` — Stage 3. Extension of `sft.py`'s assistant-only masking; the chat template adds `<image>` placeholders that consume the tile feature stream. Acceptance: SFT overfits a tiny multi-modal QA set with both VL and text-only turns.

### Rollout / inference integration

- `RolloutRequest` extension: optional `image_tiles: tuple[mx.array, ...] = ()` field (frozen dataclass — backwards compatible default).
- `Engine` extension: `new_request(...)` accepts tiles; tile features pre-pended into the prefill stream. `decode_step` unchanged — the cache holds the combined visual + text states and decode is text-only by construction.
- `inference/server.py`: `/rollout` and `/v1/chat/completions` accept base64-encoded image bytes in the request body when the loaded config has `enable_vision=True`. When `enable_vision=False`, the field is rejected with a clear error.
- `RolloutEngine.generate_with_tools` is not changed — multi-modal turns work because the request carries tiles, and tool calls still parse from generated text.

### Quantization integration

- VL encoder + connector layers use `WhaleLinear` so the existing `int4-weight` / `int8-weight` / `fp4-native` policies apply automatically.
- A new `LinearPlacement = "vision_encoder" | "vision_connector"` literal extension lets `fp4-expert` policy choose whether to quantize vision linears (default: yes for `vision_connector`, no for `vision_encoder` since those are smaller and accuracy-sensitive).

### CLI

- `baby-whale-v4 vl-align` — Stage 1 driver
- `baby-whale-v4 vl-pretrain` — Stage 2 driver
- `baby-whale-v4 vl-sft` — Stage 3 driver
- `baby-whale-v4 smoke-vl` — tiny image + tiny prompt forward pass

### Acceptance gates

- Text-only behavior is bit-identical when `enable_vision=False` (config_hash differs only when the flag is set; an existing run's checkpoint loads correctly into a model built from its saved config).
- Stage 1 connector overfits a synthetic 4-image caption set.
- Stage 2 sustains text-only validation loss within 5% of a text-only midtrain baseline run.
- Vision tile features feed correctly through every existing attention kind (sliding, HCA, CSA, MLA) — verified by a per-kind shape parity test.
- The HTTP `/rollout` endpoint round-trips image-bearing requests (server in-thread test pattern, same as the existing rollout test).

### Explicit non-goals for Step 8 v1

- No video, no audio, no native multi-modal *generation* (decode-time image output). VL2 is understanding-only; we match that.
- No SigLIP weight training from scratch — for educational purposes, accept that the encoder ships pre-trained.
- No production-scale rollout (the encoder is non-trivial; we test on tiles ≤ 4 in CI).
