# Baby Whale v4

`baby_whale_v4` is a small, Mac-first, DeepSeek-V4-inspired educational research project.

The goal is not to clone a production LLM framework. The goal is to build a readable model and training stack that exposes the important ideas directly:

- strict config validation and fail-fast unsupported modes.
- RMSNorm, partial RoPE, sliding MQA, and dynamic KV cache.
- sparse MoE with hash-routed bootstrap layers and learned top-k layers.
- HCA/CSA compressed attention, mHC, MTP, SFT, DPO, GRPO, inference scheduling, prefix cache, chunked prefill, quantization, and explicit FP4 gates.
- MLX-first Apple Silicon execution, including native FP4 quantized matmul experiments.

## Learn it — the course

New here? [`course/`](course/README.md) is a guided, hands-on walk through the **entire
LLM lifecycle** built on this code — architecture → pre-training → mid-training →
SFT / DPO → RL → serving → quantization → evaluation → vision. Each of the 22 modules
explains *why* a feature exists, links the real implementation, and ends on a
**measured** payoff you can reproduce. Start at
[`course/00-the-map`](course/00-the-map/README.md) and run the 5-minute end-to-end:

```bash
uv run python course/00-the-map/journey.py
```

## Layout

```text
baby_whale_v4/   # active implementation
refs/            # research plans and roadmap
tests/           # active tests
```

## Tooling

Requires Python 3.14+ (see `pyproject.toml`). Apple Silicon with MLX Metal is required.

This project uses the Astral toolchain:

- `uv` for dependency groups, locking, and command execution.
- `ruff` for linting and formatting.
- `ty` for fast type checking of project code.

Install/sync the locked environment:

```bash
uv sync --dev
```

Quality gates:

```bash
uv run ruff format baby_whale_v4 tests
uv run ruff check baby_whale_v4 tests
uv run ty check .
uv run python -m unittest discover -s tests
```

## Backend: MLX Runtimes

Baby Whale v4 is Mac-first, but MLX is the framework backend on both Apple Metal and MLX CUDA. The model config keeps `backend="mlx"`; execution chooses a concrete runtime with `--runtime mlx-metal` or `--runtime mlx-cuda`. CPU execution is not a project runtime target.

Install:

```bash
uv sync --dev
```

Use the `--backend` flag on `pretrain` and `midtrain` only as an explicit MLX assertion:

```bash
baby-whale-v4 pretrain --backend mlx --runtime mlx-metal --train-jsonl data/p.jsonl --out-dir runs/pre
baby-whale-v4 pretrain --backend mlx --runtime mlx-cuda --train-jsonl data/p.jsonl --out-dir runs/pre
```

`baby_whale_v4.device` exposes `is_metal_runtime()`, `is_cuda_runtime()`, `available_runtimes()`, `active_runtime()`, and `ensure_runtime_matches(backend, runtime)` for programmatic checks. The `BabyWhaleV4Config.backend: Backend` literal remains `"mlx"` so checkpoints are portable across MLX runtimes.

**Caveats:**
- `mx.fast.metal_kernel` paths in `baby_whale_v4/kernels/` are Apple-only and fail fast on `mlx-cuda`.
- MLX CUDA requires a CUDA-enabled MLX wheel on Linux/NVIDIA, for example `mlx[cuda12]` or `mlx[cuda13]`.

## Mac FP4 Boundary

`baby_whale_v4.mlx_fp4` exposes explicit MLX FP4 weight packing, dequantization, and quantized linear matmul helpers for `mxfp4` and `nvfp4`.

The active model, cache, inference loop, training path, and tests are MLX-first. The DeepSeek-aligned path is BF16 training, then expert-only native FP4 export/inference. `quant_mode="fp4-expert"` applies MLX native FP4 only to MoE expert linears; attention, router, MTP, and LM head layers stay dense. Training with `fp4-expert` fails fast because it is an export/inference policy.

`quant_mode="fp4-native"` switches every Baby Whale linear layer to MLX `quantized_matmul` for inference/forward experiments. Raw MLX native FP4 training is separately probed and fails fast because MLX does not expose gradients through quantized weights yet.

The old full-model FP4 training trials are no longer config modes. They remain only as primitive-level research benchmarks around `linear_mlx_fp4_train`, custom VJP, and the optional Metal weight-gradient kernel.

DeepSeek-V4's report describes FP4 QAT with FP32 master weights and FP4 weights dequantized into FP8 compute inside its optimized FP8 training framework. Baby Whale v4 does not expose an FP8 emulation mode because it would add quantize/dequantize overhead on Mac without a native MLX FP8 speed or memory win.

For memory-constrained DeepSeek-style training on Mac, keep weights dense and use BF16, activation checkpointing, gradient accumulation, and the factored optimizer:

```bash
uv run baby-whale-v4 pretrain \
  --precision bf16 \
  --quant none \
  --optimizer adafactor \
  --grad-accum 2 \
  --activation-checkpoint \
  --out-dir runs/bwv4-pretrain
```

For export/inference after training, use the expert-only FP4 policy:

```bash
uv run baby-whale-v4 serve --quant fp4-expert
```

The primitive Metal path is correctness-tested but still performance-gated. If it is slower than MLX's built-in matmul at the requested shape, it reports `passed: false` instead of silently falling back:

```bash
uv run baby-whale-v4 bench-fp4-training --weight-grad metal --max-ratio 1.0
```

For primitive-level memory checks:

```bash
uv run baby-whale-v4 bench-fp4-memory \
  --baseline bf16 \
  --cache-policy recompute \
  --optimizer adafactor \
  --fp4-master-dtype bf16 \
  --max-peak-ratio 1.05
```

## Run

With uv:

```bash
uv run python -m baby_whale_v4.cli smoke
```

The installable console command is:

```bash
uv run baby-whale-v4 smoke
```

Train a tiny byte-BPE tokenizer from normalized JSONL:

```bash
uv run baby-whale-v4 train-tokenizer \
  --input-jsonl data/pretrain.jsonl \
  --out runs/tokenizer.json \
  --vocab-size 2048
```

Pack normalized JSONL into token blocks with a tokenizer hash manifest:

```bash
uv run baby-whale-v4 pack-jsonl \
  --input-jsonl data/pretrain.jsonl \
  --tokenizer-path runs/tokenizer.json \
  --block-size 256 \
  --out runs/pretrain.tokens.npz
```

V4-style native long-context curriculum (single pretrain run that ramps sequence length across phases — no separate RoPE-rescale stage):

```bash
uv run baby-whale-v4 pretrain \
  --train-jsonl data/pretrain.jsonl \
  --tokenizer-path runs/tokenizer.json \
  --context-curriculum 384:50M,768:50M,1536:50M \
  --max-steps 100000 \
  --precision bf16 --optimizer adafactor \
  --out-dir runs/pretrain-curriculum
```

The model is built once at the curriculum's max context length; each phase re-packs the JSONL at its own block size and trains until `n_tokens` are consumed. K/M/B suffixes on counts are accepted (`80M = 80_000_000`). Metrics rows include `phase`, `phase_context_length`, `phase_tokens`, `phase_tokens_target` so phase transitions are visible in `metrics.jsonl` and via `watch-metrics`.

Run JSONL-backed pretrain or mid-train:

```bash
uv run baby-whale-v4 pretrain \
  --train-jsonl data/pretrain.jsonl \
  --eval-jsonl data/valid.jsonl \
  --tokenizer-path runs/tokenizer.json \
  --block-size 256 \
  --precision bf16 \
  --optimizer adafactor \
  --out-dir runs/pretrain-jsonl

uv run baby-whale-v4 midtrain \
  --train-jsonl data/code.jsonl \
  --train-jsonl data/math.jsonl \
  --tokenizer-path runs/tokenizer.json \
  --block-size 256 \
  --out-dir runs/midtrain
```

Chat-format SFT (assistant-only loss). One of three input sources is required: inline `--user`/`--assistant` pairs, a chat JSONL (`{"messages":[{"role","content"},...]}`), or a `CodeProblem` JSONL (auto-built into user/assistant pairs). Continues from any `.bw4` checkpoint via `--from-checkpoint`:

```bash
uv run baby-whale-v4 sft \
  --problems-jsonl runs/code/mbpp_train.jsonl \
  --tokenizer-path runs/tokenizer.json \
  --from-checkpoint runs/midtrain/final.bw4 \
  --block-size 384 \
  --batch-size 4 \
  --max-steps 1500 \
  --lr 2e-4 \
  --out-dir runs/sft
```

Serve a trained checkpoint over HTTP (`/v1/chat/completions`, `/generate`, `/rollout`, `/sync_weights`, `/health`). With `--from-checkpoint` the config (incl. vocab + context length) is loaded from the `.bw4`:

```bash
uv run baby-whale-v4 serve \
  --from-checkpoint runs/sft/final.bw4 \
  --tokenizer-path runs/tokenizer.json \
  --port 8765
```

The server runs **continuous batching**: a background loop thread owns the model and `RequestScheduler`, while `ThreadingHTTPServer` handler threads submit requests and stream results back — so concurrent requests interleave their decode steps instead of queueing behind one another. Streaming (`"stream": true`) emits one SSE chunk per decoded token (real TTFT), and `finish_reason` is `stop` on EOS or `length` on the token cap. `/sync_weights` swaps checkpoints on the loop thread so it can't race an in-flight forward.

Post-training code can now consume normalized chat/tool-trace JSONL through `sft_dataset_from_jsonl(...)` and normalized preference JSONL through `dpo_examples_from_jsonl(...)`.
Educational LoRA adapters can be attached to selected linear placements with `attach_lora_adapters(...)`, and pretrain/midtrain accept the educational `muon` optimizer.

One-shot in-process generation against any checkpoint (per-stage inspection, no HTTP server). Uses the same `Engine` as `serve` and the RL rollout path:

```bash
# Raw completion
uv run baby-whale-v4 generate \
  --from-checkpoint runs/pretrain/final.bw4 \
  --tokenizer-path runs/tokenizer.json \
  --prompt "def fibonacci(n):" --max-new-tokens 60 --mode greedy

# Chat-template (wraps as <|user|>...<|eot|><|assistant|>)
uv run baby-whale-v4 generate \
  --from-checkpoint runs/sft/final.bw4 \
  --tokenizer-path runs/tokenizer.json \
  --user "Write a function to add two numbers." --max-new-tokens 80
```

## Inference optimizations (vLLM / SGLang-inspired)

Five techniques borrowed from the vLLM and SGLang papers, scoped to what actually pays off at our tiny-model scale.

### `RadixKVCache` + `Engine.fork()` — SGLang-style branching prefix cache

A radix tree of token-ID spans replaces the hash-keyed `PrefixCache` for use cases where many requests share a prompt prefix (RL group rollouts, agent conversations with a shared system prompt). The tree splits on partial inserts, so two prompts that share `"sys:"` collapse to one shared node + two leaves.

```python
from baby_whale_v4.inference import Engine, GenerationOptions, RadixKVCache

radix = RadixKVCache(config=cfg, tokenizer_hash=tok.hash_signature(), capacity_nodes=256)
engine = Engine(model=model, config=cfg, tokenizer_hash=tok.hash_signature(), radix_cache=radix)

# fork(): prefill once, sample N continuations sharing the prompt KV.
branches = engine.fork(prompt_ids, n=8, options=GenerationOptions(max_new_tokens=64, mode="sample"))
while any(not s.finished for s in branches):
    engine.decode_step_group(branches)
```

### `mx.compile` on the sampler

Top-k, top-p, and min-p filter functions are wrapped in `mx.compile` to amortize MLX kernel dispatch — which dominates wall-clock on tiny models where compute is otherwise trivial.

### Top-p / min-p sampling

`GenerationOptions` now accepts `top_p` (nucleus) and `min_p` (Nguyen et al. 2024) alongside `top_k`. All three compose in vLLM's published order: top-k → top-p → min-p.

```python
opts = GenerationOptions(mode="sample", temperature=0.8, top_p=0.95, min_p=0.05)
```

### Request cancellation

`RequestState.cancel()` flags the state; the `RequestScheduler` honors it on the next tick — both for queued prefill and in-flight decode. Prevents wasted compute on agent timeouts and SSE client disconnects.

```python
state = sched.submit("req-1", prompt_ids, options)
# ... client disconnects ...
state.cancel()  # no further decode steps; cancelled requests drain on next tick
```

### Spec-decode acceptance rate

`model.spec_decode(...)` now returns a `SpecDecodeResult` with `tokens`, `n_drafts_proposed`, `n_drafts_accepted`, `n_verify_calls`, and an `acceptance_rate` property. ~0.5 is Medusa-typical, ~0.7-0.8 is EAGLE-2 territory — a direct educational signal of how much the MTP draft heads are helping.

### True continuous batched decode — `Engine.fork_batched()`

`fork()` (above) reuses the prompt KV across N branches but still runs N forward calls per decode step. `fork_batched()` tiles the B=1 prompt cache to **B=N** and runs **one** batched `model(inp, cache)` per step — exactly the SGLang `fork` win pattern fully realized.

```python
from baby_whale_v4.inference import generate_batched

state = engine.fork_batched(prompt_ids, n=8, options=GenerationOptions(max_new_tokens=64, mode="sample"))
generate_batched(engine.model, state)
# state.generated[i] is the i-th branch's token list
# state.captured_log_probs[i] is the i-th branch's rollout-time log-probs (for PPO/GRPO IS)
```

At our 13.7M-param scale this collapses N kernel dispatches to 1 per decode step — meaningful because MLX dispatch overhead, not compute, is the binding bottleneck. Tested: batched-greedy decode produces bit-identical tokens to N serial greedy decodes of the same prompt.

### `PagedKVCache` / `PagedKVPool` — vLLM PagedAttention storage

A global pool of fixed-size KV blocks indexed by per-request page tables — the canonical "treat KV like virtual memory" pattern from Kwon et al. (SOSP 2023).

```python
from baby_whale_v4.inference import Engine, GenerationOptions, PagedKVConfig, PagedKVPool

# Size the pool to the model (n_heads = n_kv_head, since KV is cached *before*
# the GQA/MQA expansion), then hand it to the Engine as the KV storage backend.
pool = PagedKVPool(PagedKVConfig.from_model_config(config, block_size=16, n_blocks=128))
engine = Engine(model=model, config=config, tokenizer_hash=tok.hash_signature(), paged_pool=pool)

out = engine.generate(prompt_ids, GenerationOptions(max_new_tokens=64, mode="greedy"))
# Greedy decode is token-identical to the dense DynamicKVCache path; generate()
# frees the request's blocks back to the pool on completion.
```

The low-level pool/page-table API stays exposed for direct study:

```python
from baby_whale_v4.inference import PagedKVCache

req = PagedKVCache(pool=pool)
req.append(layer_idx=0, key=k0, value=v0)        # allocates blocks as needed
keys, values = req.gather(layer_idx=0)            # contiguous [1, H, T, D]
req.free()                                        # blocks return to pool
```

**Educational scope:** the block-pool allocator, per-request page tables, and a block map shared across layers are real and now back a live Engine path (`Engine(paged_pool=...)`) whose greedy decode is verified token-identical to the dense cache. We still gather blocks to a dense `[B, H, T, D]` tensor at attention time rather than fusing into a custom Metal kernel — the wall-clock win from a fused gather doesn't materialize at our scale (no KV pressure on M2 with ctx=384), but the storage model is ready for longer contexts. MLA has no paged latent path, so the Engine fails fast if you pair `paged_pool` with an `mla` layer schedule.

### What we deliberately did **not** port

`FlashAttention 2/3`, `CUDA Graphs`, `disaggregated prefill/decode`, `beam search`, `constrained decoding (xgrammar)` — these solve problems we don't have at 13.7M params + 384 ctx on Apple Silicon. See `refs/INFERENCE_OPTIMIZATIONS.md` for the cost-benefit analysis.

## Per-stage evaluation

Each subcommand answers "is this stage GOOD?" with a number, so training progress is tracked by metric and not by eyeballing samples.

```bash
# Tokenizer: bytes/token, fertility tokens/word
uv run baby-whale-v4 eval-tokenizer \
  --tokenizer-path runs/tokenizer.json \
  --input-jsonl runs/data/heldout.jsonl

# Pretrain / midtrain: held-out perplexity + bits-per-byte
uv run baby-whale-v4 eval-bpb \
  --from-checkpoint runs/midtrain/final.bw4 \
  --tokenizer-path runs/tokenizer.json \
  --input-jsonl runs/data/heldout.jsonl \
  --block-size 256 --limit-blocks 50

# Coding pass@1 (HumanEval/MBPP-style via the sandboxed reward host)
uv run baby-whale-v4 eval-code \
  --from-checkpoint runs/sft/final.bw4 \
  --tokenizer-path runs/tokenizer.json \
  --problems-jsonl runs/data/mbpp_test.jsonl \
  --max-new-tokens 128 --chat-template

# Instruction-following strict accuracy (8 deterministic verifiers)
uv run baby-whale-v4 eval-ifeval \
  --from-checkpoint runs/sft/final.bw4 \
  --tokenizer-path runs/tokenizer.json

# DPO reward accuracy + margin on a preference JSONL
uv run baby-whale-v4 eval-dpo \
  --from-checkpoint runs/dpo/final.bw4 \
  --ref-checkpoint runs/sft/final.bw4 \
  --tokenizer-path runs/tokenizer.json \
  --input-jsonl runs/data/preferences.jsonl --beta 0.1

# GRPO/PPO/RLOO metrics health check (KL ceiling, entropy floor,
# reward stagnation, reward_std collapse, NaN/Inf)
uv run baby-whale-v4 eval-rl-health \
  --metrics-jsonl runs/grpo/grpo_metrics.jsonl \
  --max-kl 20 --min-entropy 0.1

# Engine vs. naive greedy decode parity (training/serving numerics match)
uv run baby-whale-v4 eval-parity \
  --from-checkpoint runs/sft/final.bw4 \
  --tokenizer-path runs/tokenizer.json \
  --prompt "def add(a, b):" --max-new-tokens 16
```

GRPO/PPO/RLOO trainers now emit `kl_mean`, `entropy_mean`, and `response_len_mean` alongside `reward_mean`/`reward_std` so `eval-rl-health` has the full panel to threshold-check.

## Live training metrics

`JsonlMetrics` echoes each logged row to stderr in addition to writing the JSONL — so every trainer (`pretrain`, `midtrain`, `sft`, `dpo`, `grpo`, `ppo`, `rloo`, `distill`, `train-code-agent`) prints lines like:

```text
[grpo] step=10 grpo_loss=0.234 entropy_mean=3.10 kl_mean=2.30 reward_mean=0.50 reward_std=0.30
```

For a calmer running summary instead of fast-scrolling logs:

```bash
# One-shot trajectory summary (first/last/min/max for every tracked metric)
uv run baby-whale-v4 watch-metrics --metrics-jsonl runs/grpo/grpo_metrics.jsonl

# Live dashboard — polls the JSONL and redraws (Ctrl-C to stop)
uv run baby-whale-v4 watch-metrics \
  --metrics-jsonl runs/grpo/grpo_metrics.jsonl \
  --watch --interval 2
```

The file is line-buffered, so `tail -f` on the metrics JSONL also works from another terminal.

Benchmark the local inference scheduler:

```bash
uv run baby-whale-v4 bench-inference \
  --requests 4 \
  --max-new-tokens 16 \
  --prefill-chunk 8 \
  --prefix-cache
```

Compare inference configs (no-cache / prefix-cache / paged / weight-quant) on one prompt suite:

```bash
uv run baby-whale-v4 bench-compare \
  --requests 4 \
  --max-new-tokens 16 \
  --quant int8-weight
```

DPO (preference optimization) — continues from any `.bw4` checkpoint (typically the SFT model). Take a preference JSONL produced by `prepare-code-prefs` (or built by hand) and apply a style nudge without touching capabilities:

```bash
# Build a coding preference JSONL (chosen = canonical, rejected = canonical of a different problem)
uv run baby-whale-v4 prepare-code-prefs \
  --problems-jsonl runs/data/mbpp_train.jsonl \
  --out runs/data/code_prefs.jsonl

uv run baby-whale-v4 dpo \
  --from-checkpoint runs/sft/final.bw4 \
  --tokenizer-path runs/tokenizer.json \
  --input-jsonl runs/data/code_prefs.jsonl \
  --beta 0.1 --lr 1e-5 --batch-size 2 --max-steps 200 \
  --out-dir runs/dpo
```

GRPO / PPO / RLOO also accept `--from-checkpoint` and `--tokenizer-path` so RL stages chain onto the previous one. The educational `--reward-token` mode rewards counts of a specific token id; for verifiable code rewards use `train-code-agent`:

```bash
uv run baby-whale-v4 grpo \
  --from-checkpoint runs/dpo/final.bw4 \
  --tokenizer-path runs/tokenizer.json \
  --prompt "def add(a, b):" \
  --reward-token 32 --group-size 4 --response-len 16 \
  --max-steps 50 --lr 5e-5 --beta-kl 0.01 \
  --out-dir runs/grpo
```

## RL infra (GRPO / PPO / RLOO)

The RL stack uses typed boundaries between rollout, reward, buffer, and trainer
— inspired by verl/SLIME but Mac-native. Every active mode runs through real
MLX kernels.

```text
training/    — algorithm + gradient step (grpo, ppo, rloo, dpo, sft, ...)
rl/buffer    — RolloutBuffer (sync list / async producer-consumer deque)
rl/rollout   — RolloutEngine Protocol
                ├── InProcessRolloutEngine  (wraps inference.Engine)
                └── HTTPRolloutEngine       (talks to inference/server.py /rollout)
rl/reward    — RewardHost Protocol (Local callable / HTTP verifier)
inference/   — production inference path: chunked prefill, prefix cache,
                 captured rollout-time log-probs
```

Run a quick GRPO experiment from the CLI:

```bash
uv run baby-whale-v4 grpo \
  --prompt "count sevens:" \
  --reward-token 55 \
  --group-size 8 \
  --response-len 8 \
  --max-steps 20 \
  --out-dir runs/grpo-sevens
```

Benchmark the rollout layer (verifies prefix cache reuse across a group):

```bash
uv run baby-whale-v4 bench-rollout \
  --group-size 8 \
  --max-new-tokens 16
```

Custom reward in a separate file:

```python
# my_reward.py
import mlx.core as mx
def reward_fn(sample: mx.array) -> float:
    return float(mx.sum(mx.equal(sample, 65)))  # count of 'A' bytes
```

```bash
uv run baby-whale-v4 grpo --prompt hi --reward-module my_reward.py --out-dir runs/grpo
```

## On-policy distillation

The student samples on the prompt distribution, the teacher labels each token with soft logits, and the loss is per-token reverse-KL on accepted samples. Matches the DeepSeek-V4 "specialists distill into a generalist" recipe.

```bash
uv run baby-whale-v4 distill \
  --from-checkpoint runs/sft/final.bw4 \
  --teacher-checkpoint runs/dpo/final.bw4 \
  --tokenizer-path runs/tokenizer.json \
  --problems-jsonl runs/data/mbpp_train.jsonl --limit 8 \
  --group-size 3 --response-len 24 --max-steps 200 \
  --lr 5e-5 --teacher-temperature 1.0 --student-temperature 1.0 \
  --out-dir runs/distill
```

Optional reward filtering (`--reward-token` / `--reward-module` / `--reward-threshold`) only feeds the loss with samples that pass the verifier. With no reward args, every sample contributes.

## Full DeepSeek-V4-style pipeline

Everything below runs end-to-end via CLI with `--from-checkpoint` chaining:

```text
tokenizer → pretrain → midtrain → sft → dpo → grpo/ppo/rloo → distill → serve
```

Each stage has matching evaluation:

```text
eval-tokenizer  → bytes/token, fertility
eval-bpb        → held-out perplexity + bits/byte
eval-code       → HumanEval/MBPP-style pass@1
eval-ifeval     → instruction-following strict accuracy
eval-dpo        → preference reward accuracy + margin
eval-rl-health  → KL/entropy/reward threshold check
eval-parity     → engine vs naive greedy decode match
```

## References

- Implementation roadmap: `refs/BABY_WHALE_V4_ROADMAP.md`
- Implementation status checklist: `refs/IMPLEMENTATION_STATUS.md`
- Deep research plan: `refs/BABY_WHALE_V4_PLAN.md`
- FP4 Metal kernel research: `refs/FP4_METAL_KERNEL_RESEARCH.md`
- Training dataset and agent/tool-use plan: `refs/TRAINING_DATASET_PLAN.md`
- Python type-theory style guide: `refs/PYTHON_TYPE_THEORY_STYLE.md`
