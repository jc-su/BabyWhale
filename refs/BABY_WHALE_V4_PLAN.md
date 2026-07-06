# Baby Whale v4 Modernization Plan

Status: draft plan, researched 2026-05-08  
Repository: Baby Whale v4
Target: Apple Silicon macOS first  
Compatibility stance: break APIs where useful  
Runtime stance: fail fast when a backend, precision, cache mode, or kernel is unsupported

## 1. Direction

The target is a small, readable DeepSeek-V4-inspired research system named `baby_whale_v4`:

- MoE-first, not dense-first.
- Long-context-first, not fixed short-context-first.
- cache/compression-first, not classic full KV-cache-first.
- RL/reasoning/tool-use-first, not SFT-only.
- Mac-local and educational, while still reflecting current frontier architecture and infra.

This project should not try to train or serve full DeepSeek-V4 locally. Even V4-Flash is 284B total / 13B active and official/community local paths are still heavyweight. The repo should instead implement scaled-down versions of the important ideas.

## 2. Current DeepSeek Direction

DeepSeek's recent open model line moved through these steps:

- DeepSeek-V3: MoE with Multi-head Latent Attention (MLA), DeepSeekMoE, auxiliary-loss-free load balancing, and multi-token prediction.
- DeepSeek-R1: reasoning post-training using large-scale RL; R1-Zero showed reasoning can emerge from RL without SFT, while R1 added cold-start data plus staged RL/SFT.
- DeepSeek-V3.2: efficient reasoning and agentic model with DeepSeek Sparse Attention (DSA), scalable RL, and large-scale agentic task synthesis.
- DeepSeek-V4: preview release on 2026-04-24 with V4-Pro and V4-Flash, both MoE, both 1M context, using hybrid compressed attention, mHC, Muon optimizer, mixed FP4/FP8 precision, and a post-training pipeline with SFT, GRPO, and on-policy distillation.

DeepSeek-V4 should be the conceptual north star. V3/R1/V3.2 become stepping stones, not the final architecture.

## 3. Baby Whale v4 Product Goal

Build a Mac-runnable research stack with the following end-to-end lifecycle:

```text
pack data
  -> pretrain baby base model
  -> midtrain on code/math/agent data
  -> SFT with DeepSeek-style thinking/tool templates
  -> preference optimization or rejection tuning
  -> GRPO/RL on verifiable tasks
  -> distill reasoning traces into smaller variants
  -> serve with compressed cache, prefix cache, and chunked prefill
```

The core artifact should be a tiny model family:

- `baby-whale-v4-30m`: correctness and laptop smoke tests.
- `baby-whale-v4-120m`: real experiments on 16-32 GB Macs.
- `baby-whale-v4-500m`: long-running local training on high-memory Macs.
- `baby-whale-v4-moe-1b-total`: MoE experiment with low active parameters.

## 4. Architecture Target

Use `BabyWhaleV4` as the architecture name.

### Required Components

- token embedding and tied or untied output head.
- RMSNorm.
- partial RoPE.
- shared K=V MQA backbone.
- hybrid attention schedule:
  - sliding attention for bootstrap/simple layers.
  - HCA: heavily compressed attention for global summary.
  - CSA: compressed sparse attention with an indexer for retrieval-like long-range attention.
- per-head learnable attention sinks.
- grouped low-rank output projection.
- mHC-style multi-stream residual path.
- MoE MLP in every main layer.
- first few layers use hash-routed MoE bootstrap.
- later layers use learned top-k MoE.
- sqrtsoftplus router scoring.
- shared expert in parallel with routed experts.
- clamped SwiGLU routed experts.
- optional MTP head.

### Explicit Non-Goals For V1

- no full 1M-token context in the first implementation.
- no claim of native DeepSeek-V4 parity.
- no native FP4 training claim on Mac.
- no hidden CPU/cloud substitution when MLX/MPS cannot run a kernel.
- no compatibility shim unless a test explicitly proves it is useful.

## 5. Scaled Architecture Presets

These are Mac-sized sketches, not official DeepSeek shapes.

```toml
[model]
name = "baby-whale-v4-30m"
vocab_size = 32768
n_layers = 8
d_model = 384
n_heads = 6
head_dim = 64
n_kv_heads = 1
hc_mult = 2
sliding_window = 512
context = 4096
n_routed_experts = 8
n_shared_experts = 1
experts_per_token = 2
moe_intermediate_size = 512
mtp_layers = 1
```

```toml
[model]
name = "baby-whale-v4-120m"
vocab_size = 65536
n_layers = 16
d_model = 768
n_heads = 12
head_dim = 64
n_kv_heads = 1
hc_mult = 2
sliding_window = 1024
context = 16384
n_routed_experts = 16
n_shared_experts = 1
experts_per_token = 2
moe_intermediate_size = 1024
mtp_layers = 1
```

```toml
[attention]
layer_schedule = [
  "sliding", "hca",
  "csa", "hca", "csa", "hca",
  "csa", "hca"
]
hca_compress_rate = 64
csa_compress_rate = 4
csa_index_topk = 64
partial_rotary_factor = 0.125
attention_sink = true
```

## 6. Hybrid Attention Design

### Sliding Attention

This is the correctness baseline and the local branch for all CSA/HCA layers.

Keep it simple:

- fixed local window.
- dynamic cache support.
- parity test against dense masked attention.

### HCA: Heavily Compressed Attention

HCA compresses old context aggressively and attends to pooled global state.

Baby implementation:

- group tokens into non-overlapping blocks.
- compute one compressed KV per block.
- cache compressed states per layer.
- combine local sliding KV plus compressed global KV.

Acceptance:

- HCA cache grows by compressed blocks, not raw tokens.
- no-cache and cache paths match within tolerance on tiny sequences.
- memory usage is lower than raw full KV at long context.

### CSA: Compressed Sparse Attention

CSA is the more retrieval-like branch.

Baby implementation:

- compress tokens at rate `m=4` or configurable.
- maintain overlapping compressed blocks.
- add a small learned indexer.
- select top-k compressed entries per query.
- attend over local sliding KV plus selected compressed KV.

Acceptance:

- top-k indexer selections are deterministic in eval mode.
- indexer state is stored in the cache.
- dense/debug mode can force all compressed entries for parity.

## 7. mHC Plan

DeepSeek-V4 replaces standard residual connections with manifold-constrained hyper-connections. For a baby implementation, start with a practical approximation:

- keep `hc_mult` parallel streams with shape `[B, T, hc_mult, D]`.
- mix streams before/after attention and MLP.
- implement a Sinkhorn-normalized combination matrix.
- freeze or detach Sinkhorn path in early experiments if training is unstable.

Acceptance:

- `hc_mult=1` exactly reduces to normal residual behavior.
- Sinkhorn output is approximately doubly stochastic.
- training with `hc_mult=2` is stable on synthetic data.

## 8. MoE Plan

Baby Whale v4 should be MoE-native.

### Hash-MoE Bootstrap

For the first `n_hash_layers`:

- route token ids through a frozen `token_id -> expert_id` table.
- still compute expert scores for weights.
- make the hash table part of the checkpoint.

This mimics V4's static early routing while staying simple enough to inspect.

### Learned MoE Layers

For later layers:

- top-k token-choice routing.
- sqrtsoftplus scoring.
- score-correction bias for auxiliary-loss-free-style balancing.
- shared expert always runs.
- routed experts use clamped SwiGLU.
- log per-expert token counts.

Acceptance:

- top-k routing test with hand-computed small tensors.
- no token silently drops unless capacity mode explicitly says so.
- expert imbalance metrics are emitted every train/eval step.

## 9. MTP Plan

Multi-token prediction should be implemented early because it affects training and inference experiments.

V1:

- one next-token auxiliary head.
- one extra future-token prediction head.
- configurable auxiliary loss weight.
- generation can use MTP for speculative draft only after correctness tests exist.

Acceptance:

- loss decomposes into next-token loss and MTP auxiliary loss.
- disabling MTP produces identical logits for the main head.
- speculative decode rejects incorrect MTP tokens cleanly.

## 10. Training Pipeline

### Pretraining

Command:

```bash
baby-whale-v4 train pretrain --config configs/baby_whale_v4/pretrain_30m_mlx.toml
```

Features:

- packed token data.
- document boundary tokens.
- Muon optimizer option for matrix weights.
- AdamW option for baseline.
- gradient accumulation.
- mixed precision where backend supports it.
- checkpoint lineage and config hash.
- JSONL metrics.

Fail-fast checks:

- unsupported optimizer/backend combination.
- unsupported precision.
- tokenizer mismatch.
- memory estimate exceeds configured budget.

### Mid-Training

Command:

```bash
baby-whale-v4 train midtrain --base runs/baby-whale-v4-base --config configs/baby_whale_v4/midtrain_code_math_agent.toml
```

Data mixture:

- code.
- math.
- long-context synthetic retrieval.
- tool-call traces.
- self-generated reasoning traces.

Metrics:

- aggregate val loss.
- per-domain val loss.
- retrieval/needle score.
- code/math pass rate.

### SFT

Use a DeepSeek-V4-like template with explicit thinking and tool-call modes.

Support:

- non-thinking mode.
- thinking mode with explicit reasoning content.
- tool-call grammar.
- assistant-only loss.
- optional reasoning-loss mask.

### Preference / Rejection Training

Do this before RL:

- DPO for chosen/rejected pairs.
- rejection fine-tuning from sampled completions.
- tool-call format verifier.

### GRPO / RL

Command:

```bash
baby-whale-v4 train grpo --base runs/baby-whale-v4-sft --config configs/baby_whale_v4/grpo_math_code.toml
```

Tasks:

- arithmetic.
- unit-test code tasks.
- JSON/tool-call correctness.
- retrieval from long context.

Required:

- group sampling per prompt.
- verifiable reward functions.
- KL against reference policy.
- replay logs containing prompt, samples, rewards, selected route metadata, and indexer selections.

Do not start with PPO. GRPO-style local RL is a better fit for a Mac research stack.

## 11. On-Policy Distillation

DeepSeek-V4's post-training description points toward independent domain expert cultivation followed by consolidation. Baby version:

1. Train small domain adapters or branch checkpoints:
   - math.
   - code.
   - tool-use.
   - long-context retrieval.
2. Generate on-policy samples from each.
3. Filter by verifiable reward and format checks.
4. Distill into a single consolidated checkpoint.

Acceptance:

- domain checkpoints record their source dataset and reward function.
- distilled model improves at least one held-out domain without breaking format tests.
- rejected traces are saved for debugging.

## 12. Inference Engine

The inference engine is part of the model, not a later add-on.

Core concepts:

- separate prefill and decode.
- cache object per request.
- local sliding cache.
- HCA compressed cache.
- CSA compressed/indexer cache.
- prefix cache.
- chunked prefill.
- MTP speculative decode.
- streaming output.
- tool-call parser.

Command:

```bash
baby-whale-v4 serve --from-checkpoint runs/baby-whale-v4-sft --runtime mlx-metal --port 8000
```

## 13. Cache Architecture

V4-style long-context efficiency depends on cache design. Build these cache layers:

- `SlidingKVCache`.
- `HCACache`: compressed block pool and count.
- `CSACache`: overlapping compressor state plus indexer pool.
- `PrefixCache`: content-addressed reusable prompt states.
- `OffloadCache`: explicit unified-memory/disk offload experiments.

Cache keys must include:

- model checkpoint hash.
- tokenizer hash.
- backend.
- runtime.
- precision.
- layer schedule.
- compression rates.
- cache quantization mode.

No cache reuse across incompatible configs.

## 14. MLX Runtime Strategy

### MLX Only

MLX is the only framework backend for:

- local inference.
- local training.
- quantized weights.
- LoRA/SFT.
- custom cache layouts.
- Metal-adjacent and MLX-CUDA experiments.

The concrete runtime is separate from the framework backend:

- `backend="mlx"` is the model/config framework boundary.
- `runtime="mlx-metal"` is the Mac default.
- `runtime="mlx-cuda"` is supported when the installed MLX wheel exposes CUDA.

If MLX cannot execute a requested feature on the selected runtime, the project should fail before model creation or at the narrow primitive gate with a precise error.

### Backend Reality

Full DeepSeek-V4-Flash is not a normal Mac training target. Community MLX ports exist, but their model size is far beyond a normal laptop training loop. Treat full V4 local inference as a separate compatibility experiment, not the core repo goal.

The project goal is to build `BabyWhaleV4`, not a generic DeepSeek-V4 runner.

## 15. Precision Plan

### Practical Mac Modes

- fp32 for tests.
- fp16/bf16 for training where stable.
- int8/int4 weight-only inference.
- int8 KV/cache experiments.

### DeepSeek-V4-Inspired Modes

- Native MLX FP4 for MoE expert weights.
- QAT experiments that match native packed inference paths.
- FP8 is reference-only until MLX exposes a native path that can beat dense BF16/FP32. DeepSeek-V4 dequantizes FP4 weights into FP8 compute inside an optimized FP8 framework; Baby Whale must not model that as `fp8-sim`.

### Fail-Fast Native FP4 Rule

`fp4-native` must use a real MLX FP4 execution path. It is not a label for emulated FP4 and it is not optional.

Current educational implementation:

- `baby_whale_v4.mlx_fp4` uses MLX `quantize`, `dequantize`, and `quantized_matmul` for explicit `mxfp4` and `nvfp4` weight-only experiments.
- `quant_mode="fp4-expert"` is the DeepSeek-aligned export/inference policy: only MoE expert linears resolve to native FP4, and training fails fast.
- `quant_mode="fp4-native"` switches Baby Whale linear layers to MLX `quantized_matmul`.
- raw MLX native FP4 training is blocked because MLX currently reports no gradient path through native FP4 `quantized_matmul` weights.
- full-FP4 training trials are no longer model config modes. The custom-VJP and Metal paths stay as primitive-level research benchmarks only.
- The Metal kernel currently uses an 8x8 simdgroup-matrix tile. It remains behind a performance gate because it does not consistently beat MLX's built-in matmul or bf16/fp32 memory behavior yet.
- Full-training memory controls now include activation checkpointing, microbatch gradient accumulation, and an Adafactor optimizer option. The recommended Mac training route is dense BF16, then `fp4-expert` export/inference.
- unsupported non-MLX backends fail during config validation; unavailable MLX runtimes fail during execution setup.

The project supports the following quantization modes; all of them route through
real MLX kernels — there is no quantize/dequantize emulation shim:

- `int8-weight` / `int4-weight`: `mx.quantized_matmul(mode="affine")` with packed integer weights, group_size 64.
- `fp4-expert`: DeepSeek-style MoE-expert-only native FP4 export/inference policy.
- `fp4-native`: MLX native FP4 (`mxfp4`/`nvfp4`) `quantized_matmul` for every Whale linear.

`fp4-sim` and `fp8-sim` are intentionally absent: a Python round-trip through an FP4/FP8
grid costs accuracy without buying the hardware throughput of the real `quantized_matmul`
path, and DeepSeek-V4's FP4-to-FP8 compute is a property of an optimized FP8 training stack
that MLX does not expose today.

## 16. Repository Structure

Replace the three-file teaching layout with explicit modules:

```text
baby_whale_v4/
  cli.py
  config/
  tokenizers/
  data/
  models/
    baby_whale_v4/
      config.py
      model.py
      attention.py
      hca.py
      csa.py
      mhc.py
      moe.py
      mtp.py
      cache.py
  backends.py
  training/
    pretrain.py
    midtrain.py
    sft.py
    dpo.py
    grpo.py
    distill.py
  inference/
    engine.py
    scheduler.py
    prefix_cache.py
    server.py
  evals/
  benchmarks/
```

New implementation work belongs under top-level `baby_whale_v4/`.

## 17. Milestones

### Milestone 0: Foundation

- `pyproject.toml`.
- CLI.
- typed config validation.
- JSONL metrics.
- explicit backend and precision checks.

Acceptance:

- invalid backend/attention/precision combinations fail before model creation.

### Milestone 1: BabyWhaleV4 Dense Skeleton

Even though the target is MoE, start with a dense skeleton to validate attention/cache/mHC.

- RMSNorm.
- partial RoPE.
- MQA.
- sliding attention.
- mHC with `hc_mult=1` and `hc_mult=2`.
- cache decode parity.

Acceptance:

- tiny model overfits a toy dataset.
- cache decode equals full forward on short sequences.

### Milestone 2: MoE Native

- hash-MoE bootstrap.
- learned top-k MoE.
- sqrtsoftplus routing.
- shared expert.
- clamped SwiGLU.
- expert metrics.

Acceptance:

- routing tests pass.
- no silent token drops.
- tiny MoE trains on synthetic data.

### Milestone 3: HCA

- compressed block cache.
- local + compressed attention.
- memory benchmark.

Acceptance:

- HCA cache memory is lower than raw KV at long context.
- logits match debug dense mode within tolerance on small sequences.

### Milestone 4: CSA

- overlapping compressed blocks.
- learned indexer.
- top-k compressed retrieval.
- indexer replay metadata.

Acceptance:

- deterministic eval selections.
- cache stores compressor and indexer state.
- long-context retrieval synthetic task improves over sliding-only.

### Milestone 5: MTP And Spec Decode

- MTP auxiliary loss.
- MTP draft generation.
- verifier/reject path.

Acceptance:

- MTP disabled path is identical to baseline.
- accepted speculative tokens match normal decode.

### Milestone 6: Training Lifecycle

- pretrain.
- midtrain.
- SFT.
- DPO/rejection tuning.
- GRPO.
- on-policy distillation.

Acceptance:

- toy math/code GRPO improves pass rate.
- tool-call format reward improves valid tool calls.
- distillation improves a held-out task.

### Milestone 7: Inference Infra

- prefill/decode engine.
- prefix cache.
- chunked prefill.
- compressed cache offload.
- OpenAI-compatible local server.

Acceptance:

- prefix reuse reduces TTFT.
- chunked prefill avoids decode starvation in mixed workloads.
- offload mode reports latency and memory tradeoff.

### Milestone 8: MLX Optimization

- MLX-only model path.
- MLX-only cache path.
- quantized expert weights.
- benchmark MLX dense, affine quantized, and native FP4 paths.

Acceptance:

- native FP4 uses MLX `quantized_matmul`.
- unsupported MLX kernels fail with exact feature names.

### Milestone 9: Vision-Language Extension (DeepSeek-VL2 recipe)

DeepSeek-V4 is text-only — multi-modal generation was deferred. This milestone follows the DeepSeek-VL2 paper instead, using SigLIP-SO400M-384 as the vision encoder, an MLP connector, and the existing DeepSeekMoE-aligned LLM as the language tower.

Architecture additions:

- `vision/encoder.py` — educational SigLIP-style ViT in MLX. Pretrained weights ship in; we do not train SigLIP from scratch.
- `vision/tiling.py` — dynamic tiling: split a high-res image into 384×384 tiles such that `(m·384, n·384)` minimizes padding for the input aspect ratio with `mn ≤ vision_max_tiles`, plus one global thumbnail. All tiles share the same encoder.
- `vision/connector.py` — `VisionMLPConnector`: 2-layer MLP projecting `vision_dim → n_embd`. All linears are `WhaleLinear` so quantization policies apply.
- `BabyWhaleV4Model.__call__` accepts optional `image_tiles`; tile features are prepended to the token embedding stream before block dispatch.

Three-stage training (matches VL2 paper):

- Stage 1 — VL alignment. Freeze the LLM. Train only the connector (and optionally the vision encoder's last layer). Data: ShareGPT4V-scale captions; loss: next-token CE on text. Verifies that visual features can land in the LLM's input space without disturbing the LLM.
- Stage 2 — VL pretraining. Unfreeze everything. Data: ~70% VL (interleaved, OCR, VQA, grounding), ~30% text-only from the base pretrain corpus. Loss: next-token CE on text tokens only — vision tokens never contribute to the loss. Tracks the V2 paper's split exactly.
- Stage 3 — VL SFT. Extends `sft.py`'s assistant-token masking. Chat template gets a `<image>` placeholder. Mixed multi-modal + text-only dialogues; loss masked to assistant + special tokens.

Backwards compatibility:

- All vision fields default off in `BabyWhaleV4Config`. With `enable_vision=False`, the model is bit-identical to the text-only path.
- `RolloutRequest` gains `image_tiles: tuple[mx.array, ...] = ()` (frozen-dataclass-compatible default). All current text-only rollouts unchanged.
- The HTTP rollout endpoint accepts base64 image bytes only when the loaded config has vision enabled — otherwise reject with a clear error.

Acceptance:

- text-only forward is bit-identical when `enable_vision=False`.
- Stage 1 connector loss decreases on a synthetic image-caption set; LLM weights are byte-identical before vs after.
- Stage 2 holds text-only validation perplexity within 5% of a text-only midtrain baseline.
- vision tiles feed correctly through every attention kind (sliding/HCA/CSA/MLA) — per-kind parity test.
- `bench-rollout` reports tile prefill cost separately from text prefill.

Explicit non-goals for v1:

- no audio / video / image generation.
- no SigLIP weight training from scratch — encoder ships pre-trained.
- no production-scale tile counts in CI (≤ 4 tiles in tests).

## 18. First Implementation Slice

Start here:

1. Create `pyproject.toml`, CLI, typed config.
2. Add `BabyWhaleV4Config`.
3. Implement the MLX reference:
   - RMSNorm.
   - partial RoPE.
   - MQA.
   - sliding attention.
   - mHC `hc_mult=1/2`.
   - dynamic cache.
4. Add cache parity tests.
5. Add tiny synthetic pretraining loop.
6. Add hash-MoE and learned top-k MoE.
7. Add HCA.
8. Add CSA.
9. Add native MLX FP4 quantized matmul tests and fail-fast gates.

This sequence keeps the hard parts visible and testable.

## 19. Research Sources

Primary/current sources used:

- DeepSeek official V4 preview release, 2026-04-24: https://api-docs.deepseek.com/news/news260424
- DeepSeek-V4 model card: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash
- Hugging Face Transformers DeepSeek-V4 architecture docs: https://huggingface.co/docs/transformers/model_doc/deepseek_v4
- Transformers DeepSeek-V4 config source: https://github.com/huggingface/transformers/blob/v5.8.0/src/transformers/models/deepseek_v4/configuration_deepseek_v4.py
- DeepSeek-V3 technical report: https://arxiv.org/abs/2412.19437
- DeepSeek-V3.2 release: https://api-docs.deepseek.com/news/news251201
- DeepSeek-V3.2 paper: https://arxiv.org/abs/2512.02556
- DeepSeek-R1 model card and paper citation: https://huggingface.co/deepseek-ai/DeepSeek-R1
- DeepSeek-VL2 paper (multi-modal training recipe — VL alignment / VL pretraining / SFT three-stage): https://arxiv.org/abs/2412.10302
- DeepSeek-VL2 GitHub reference: https://github.com/deepseek-ai/DeepSeek-VL2
- SigLIP encoder reference (used by DeepSeek-VL2): https://huggingface.co/google/siglip-so400m-patch14-384
- DualPath KV-cache inference paper: https://arxiv.org/abs/2602.21548
- SGLang DeepSeek-V4 launch support notes: https://www.lmsys.org/blog/2026-04-25-deepseek-v4/
- SGLang DeepSeek-V4 serving docs: https://lmsysorg.mintlify.app/cookbook/autoregressive/DeepSeek/DeepSeek-V4
- MLX documentation: https://ml-explore.github.io/mlx/build/html/index.html
- MLX `quantize`: https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html
- MLX `dequantize`: https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.dequantize.html
- MLX `quantized_matmul`: https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantized_matmul.html
- MLX community DeepSeek-V4-Flash nvfp4 port: https://huggingface.co/mlx-community/DeepSeek-V4-Flash-nvfp4
- vLLM-MLX / Apple Silicon inference paper: https://arxiv.org/abs/2601.19139
