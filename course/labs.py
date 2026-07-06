"""Autograder harness for the fill-in-the-blank labs.

Each ``grade_*`` function takes the *learner's* implementation and raises
``AssertionError`` if it's wrong (or returns ``None`` on success). The reference
solutions live here too, so the course test suite can prove each grader passes
for a correct implementation — the same check the learner's script runs.

The learner never sees the reference; they run e.g.::

    python course/03-attention-mla/lab_mla.py   # PASS once you fill it in
"""

from __future__ import annotations

from collections.abc import Callable

import mlx.core as mx

# --- Module 03: MLA low-rank KV round-trip ---------------------------------
#
# The heart of MLA: instead of caching full K and V, cache a small latent
# ``c = kv @ w_down`` (dim r), and reconstruct ``kv_hat = c @ w_up`` (dim d) on
# the fly. The cache stores r numbers per token instead of d — that's the win.

MlaRoundTrip = Callable[[mx.array, mx.array, mx.array], tuple[mx.array, mx.array]]


def mla_roundtrip_reference(
    kv: mx.array, w_down: mx.array, w_up: mx.array
) -> tuple[mx.array, mx.array]:
    latent = kv @ w_down
    reconstructed = latent @ w_up
    return latent, reconstructed


def grade_mla_roundtrip(fn: MlaRoundTrip) -> None:
    mx.random.seed(0)
    seq_len, d_kv, rank = 5, 8, 3
    kv = mx.random.normal((seq_len, d_kv))
    w_down = mx.random.normal((d_kv, rank))
    w_up = mx.random.normal((rank, d_kv))

    latent, reconstructed = fn(kv, w_down, w_up)
    ref_latent, ref_reconstructed = mla_roundtrip_reference(kv, w_down, w_up)

    assert tuple(latent.shape) == (seq_len, rank), (
        f"latent should be [T, rank]={(seq_len, rank)} — the small thing you cache — "
        f"got {tuple(latent.shape)}"
    )
    assert rank < d_kv, "sanity: the latent rank must be smaller than d_kv to save memory"
    assert bool(mx.allclose(latent, ref_latent, atol=1e-5)), "latent = kv @ w_down"
    assert bool(mx.allclose(reconstructed, ref_reconstructed, atol=1e-5)), (
        "reconstructed = latent @ w_up"
    )


# --- Module 02: rotary position embedding (RoPE) ---------------------------
# Rotate feature pairs by an angle that grows with position, so a token encodes
# *where* it is and dot-products encode *relative* position. A modeling idea, not
# a systems one — this course is not only about ML systems.

RopeFn = Callable[[mx.array, mx.array, mx.array], mx.array]


def rope_reference(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotate_half = mx.concatenate([-x2, x1], axis=-1)
    return x * cos + rotate_half * sin


def grade_rope(fn: RopeFn) -> None:
    mx.random.seed(0)
    seq_len, dim = 4, 8
    x = mx.random.normal((seq_len, dim))
    positions = mx.arange(seq_len).reshape(seq_len, 1)
    inv_freq = 1.0 / (10000.0 ** (mx.arange(0, dim // 2) / (dim // 2)))
    emb = mx.concatenate([positions * inv_freq, positions * inv_freq], axis=-1)
    cos, sin = mx.cos(emb), mx.sin(emb)

    out = fn(x, cos, sin)
    assert tuple(out.shape) == (seq_len, dim), f"RoPE preserves shape, got {tuple(out.shape)}"
    assert bool(mx.allclose(out[0], x[0], atol=1e-5)), "at position 0 (angle 0) RoPE is a no-op"
    assert bool(mx.allclose(out, rope_reference(x, cos, sin), atol=1e-5)), "rotate-half formula"


# --- Module 12: the DPO loss ------------------------------------------------
# Prefer 'chosen' over 'rejected', measured relative to a frozen reference, in one
# closed-form loss — the alignment objective with no reward model and no RL loop.

DpoFn = Callable[[mx.array, mx.array, mx.array, mx.array, float], mx.array]


def dpo_loss_reference(
    pi_chosen: mx.array,
    pi_rejected: mx.array,
    ref_chosen: mx.array,
    ref_rejected: mx.array,
    beta: float,
) -> mx.array:
    logits = beta * ((pi_chosen - pi_rejected) - (ref_chosen - ref_rejected))
    return -mx.mean(mx.log(mx.sigmoid(logits)))


def grade_dpo(fn: DpoFn) -> None:
    mx.random.seed(0)
    pc, pr = mx.random.normal((6,)), mx.random.normal((6,))
    rc, rr = mx.random.normal((6,)), mx.random.normal((6,))
    out = fn(pc, pr, rc, rr, 0.1)
    assert tuple(out.shape) == (), "DPO loss is a scalar"
    assert bool(mx.allclose(out, dpo_loss_reference(pc, pr, rc, rr, 0.1), atol=1e-5)), (
        "loss = -mean(log sigmoid(beta * ((pc - pr) - (rc - rr))))"
    )
    stronger = fn(pc + 5.0, pr - 5.0, rc, rr, 0.1)
    assert float(stronger) < float(out), "loss must fall when the policy prefers 'chosen' more"


# --- Module 13: GRPO group-relative advantage ------------------------------
# No value network: score each rollout against its group by normalizing rewards,
# so 'better than the average attempt' is the RL signal.

AdvantageFn = Callable[[mx.array], mx.array]


def group_advantages_reference(rewards: mx.array) -> mx.array:
    mean = mx.mean(rewards)
    std = mx.sqrt(mx.var(rewards) + 1e-8)
    return (rewards - mean) / std


def grade_group_advantages(fn: AdvantageFn) -> None:
    rewards = mx.array([0.0, 1.0, 0.0, 1.0, 0.5, 0.0])
    adv = fn(rewards)
    assert tuple(adv.shape) == (6,), "one advantage per rollout"
    assert abs(float(mx.mean(adv))) < 1e-4, "advantages should be ~zero-mean within the group"
    assert bool(mx.allclose(adv, group_advantages_reference(rewards), atol=1e-5)), (
        "advantage = (reward - group_mean) / (group_std + eps)"
    )


# --- Module 01: RMSNorm -----------------------------------------------------
# LayerNorm re-centers (subtract mean) AND re-scales (divide by std). RMSNorm keeps
# only the re-scale — divide by the root-mean-square. Why this: transformers don't
# need the mean-centering, so dropping it is cheaper for the same result.

NormFn = Callable[[mx.array, mx.array, float], mx.array]


def rms_norm_reference(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    mean_square = mx.mean(x * x, axis=-1, keepdims=True)
    return x * mx.rsqrt(mean_square + eps) * weight


def grade_rms_norm(fn: NormFn) -> None:
    mx.random.seed(0)
    x = mx.random.normal((4, 8)) * 5.0
    weight = mx.ones((8,))
    out = fn(x, weight, 1e-5)
    assert tuple(out.shape) == (4, 8), "RMSNorm preserves shape"
    rms = mx.sqrt(mx.mean(out * out, axis=-1))
    assert bool(mx.allclose(rms, mx.ones(4), atol=1e-2)), "with weight=1, each row's RMS ~ 1"
    assert bool(mx.allclose(out, rms_norm_reference(x, weight, 1e-5), atol=1e-5)), (
        "x * rsqrt(mean(x^2) + eps) * weight"
    )


# --- Module 02: scaled dot-product attention --------------------------------
# similarity = q·kᵀ; the /√d keeps scores from saturating softmax as d grows; the mask
# forbids the future. Why this: scale THEN mask THEN softmax — order matters.

AttnFn = Callable[[mx.array, mx.array, mx.array, mx.array], mx.array]


def attention_reference(q: mx.array, k: mx.array, v: mx.array, mask: mx.array) -> mx.array:
    scores = (q @ k.T) / (q.shape[-1] ** 0.5)
    scores = mx.where(mask, scores, mx.array(-1e9))
    weights = mx.softmax(scores, axis=-1)
    return weights @ v


def grade_attention(fn: AttnFn) -> None:
    mx.random.seed(0)
    seq_len, dim = 4, 8
    q = mx.random.normal((seq_len, dim))
    k = mx.random.normal((seq_len, dim))
    v = mx.random.normal((seq_len, dim))
    rows, cols = mx.arange(seq_len)[:, None], mx.arange(seq_len)[None, :]
    mask = cols <= rows  # causal
    out = fn(q, k, v, mask)
    assert tuple(out.shape) == (seq_len, dim), "output is [T, d]"
    assert bool(mx.allclose(out, attention_reference(q, k, v, mask), atol=1e-4)), (
        "softmax(q·kᵀ / √d) · v, with masked positions set to -inf before softmax"
    )
    assert bool(mx.allclose(out[0], v[0], atol=1e-4)), "token 0 can only attend to itself"


# --- Module 05: top-k expert routing ----------------------------------------
# Score every expert, keep only the top-k, softmax OVER THOSE k. Why this: sparsity =
# capacity without paying the FLOPs of running all experts.

RouteFn = Callable[[mx.array, int], tuple[mx.array, mx.array]]


def moe_route_reference(logits: mx.array, k: int) -> tuple[mx.array, mx.array]:
    order = mx.argsort(-logits, axis=-1)[:, :k]
    top_logits = mx.take_along_axis(logits, order, axis=-1)
    return order, mx.softmax(top_logits, axis=-1)


def grade_moe_route(fn: RouteFn) -> None:
    mx.random.seed(0)
    logits = mx.random.normal((4, 6))
    idx, gates = fn(logits, 2)
    assert tuple(gates.shape) == (4, 2), "one gate per selected expert"
    assert bool(mx.allclose(mx.sum(gates, axis=-1), mx.ones(4), atol=1e-5)), "gates softmax to 1"
    got = mx.sort(mx.take_along_axis(logits, idx, axis=-1), axis=-1)
    ref_idx, _ = moe_route_reference(logits, 2)
    want = mx.sort(mx.take_along_axis(logits, ref_idx, axis=-1), axis=-1)
    assert bool(mx.allclose(got, want, atol=1e-5)), "must select the top-k experts by score"


# --- Module 16: speculative acceptance --------------------------------------
# Accept a drafted token only if it equals what the model would greedily produce; stop
# at the first mismatch. Why this: the accepted prefix IS greedy output, so speculation
# is a pure speedup with zero quality change.

AcceptFn = Callable[[mx.array, mx.array], int]


def spec_accept_reference(draft: mx.array, verify: mx.array) -> int:
    accepted = 0
    for i in range(int(draft.shape[0])):
        if int(draft[i]) != int(verify[i]):
            break
        accepted += 1
    return accepted


def grade_spec_accept(fn: AcceptFn) -> None:
    assert fn(mx.array([1, 2, 3]), mx.array([1, 2, 3])) == 3, "all match -> accept all"
    assert fn(mx.array([1, 9, 3]), mx.array([1, 2, 3])) == 1, "mismatch at 1 -> accept 1"
    assert fn(mx.array([9, 2, 3]), mx.array([1, 2, 3])) == 0, "mismatch at 0 -> accept 0"


# --- Module 09: cross-entropy with ignore-index -----------------------------
# The whole pre-training objective: make the true next token likely. The ignore-index
# skips padding/prompt positions so they don't dilute the loss. Why this: it's the
# negative log-likelihood, averaged over only the tokens that count.

CrossEntropyFn = Callable[[mx.array, mx.array, int], mx.array]


def cross_entropy_reference(logits: mx.array, targets: mx.array, ignore_index: int) -> mx.array:
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    picked = mx.take_along_axis(log_probs, targets[:, None], axis=-1)[:, 0]
    keep = mx.not_equal(targets, ignore_index).astype(log_probs.dtype)
    return -mx.sum(picked * keep) / mx.maximum(mx.sum(keep), 1.0)


def grade_cross_entropy(fn: CrossEntropyFn) -> None:
    mx.random.seed(0)
    logits = mx.random.normal((4, 10))
    targets = mx.array([3, 1, 7, -1])  # last position is ignored
    out = fn(logits, targets, -1)
    assert tuple(out.shape) == (), "cross-entropy is a scalar"
    assert bool(mx.allclose(out, cross_entropy_reference(logits, targets, -1), atol=1e-5)), (
        "-mean over kept positions of log-softmax at the target"
    )
    confident = mx.where(mx.arange(10)[None, :] == 2, 20.0, -20.0)
    assert float(fn(confident, mx.array([2]), -1)) < 0.01, "confident-correct -> ~0 loss"


# --- Module 14: KV-cache append ---------------------------------------------
# The mechanic that turns O(n²) decode into O(n): keep every past token's K (and V) and
# just append the new one, so attention never recomputes the past. Why this: concatenate
# along the time axis; nothing else changes.

AppendFn = Callable[[mx.array, mx.array], mx.array]


def kv_append_reference(k_cache: mx.array, k_new: mx.array) -> mx.array:
    return mx.concatenate([k_cache, k_new], axis=-2)


def grade_kv_append(fn: AppendFn) -> None:
    mx.random.seed(0)
    k_cache = mx.random.normal((1, 2, 3, 4))  # [B, H, T=3, D]
    k_new = mx.random.normal((1, 2, 1, 4))
    out = fn(k_cache, k_new)
    assert tuple(out.shape) == (1, 2, 4, 4), "cache grows by one along the time axis"
    assert bool(mx.allclose(out[:, :, :3, :], k_cache, atol=1e-6)), "past keys are preserved"
    assert bool(mx.allclose(out[:, :, 3:, :], k_new, atol=1e-6)), "the new key lands at the end"


# --- Module 05: the SwiGLU expert FFN ---------------------------------------
# Each MoE expert is a gated MLP. Graded against the REAL SwiGLUExpert, so you
# build the actual component, not a toy of it.


def swiglu_reference(x, w_gate, w_up, w_down, clamp):
    gate = mx.clip(w_gate(x), -clamp, clamp)
    up = mx.clip(w_up(x), -clamp, clamp)
    return w_down((gate * mx.sigmoid(gate)) * up)


def grade_swiglu(fn) -> None:
    from baby_whale_v4.layers import SwiGLUExpert

    mx.random.seed(0)
    expert = SwiGLUExpert(16, 32)
    expert.eval()
    x = mx.random.normal((4, 16))
    out = fn(x, expert.w_gate, expert.w_up, expert.w_down, expert.clamp)
    assert bool(mx.allclose(out, expert(x), atol=1e-4)), (
        "swiglu = w_down( silu(clip(w_gate·x)) * clip(w_up·x) ) — must match the real SwiGLUExpert"
    )


# --- Module 07: the MTP head ------------------------------------------------
# Predict a future token from the hidden state. Graded against the REAL MTPHead.


def mtp_head_reference(h, transform, head):
    t = transform(h)
    return head(t * mx.sigmoid(t))


def grade_mtp_head(fn) -> None:
    from baby_whale_v4.mtp import MTPHead

    mx.random.seed(0)
    mtp = MTPHead(16, 50)
    mtp.eval()
    h = mx.random.normal((2, 4, 16))
    assert bool(mx.allclose(fn(h, mtp.transform, mtp.head), mtp(h), atol=1e-4)), (
        "mtp = head(silu(transform(h))) — must match the real MTPHead"
    )


# --- Module 01: ASSEMBLE a transformer layer --------------------------------
# The composition step: wire your components into one pre-norm residual layer,
# built from the REAL RMSNorm, attention, and SwiGLU modules. hc_mult=1 form;
# Module 06 upgrades the plain residual to a learned multi-branch one.


def transformer_layer_reference(x, ln1, attn, ln2, ffn):
    h = x + attn(ln1(x))
    return h + ffn(ln2(h))


def grade_transformer_layer(fn) -> None:
    from baby_whale_v4 import BabyWhaleV4Config
    from baby_whale_v4.attention import SlidingMQAAttention
    from baby_whale_v4.layers import RMSNorm, SwiGLUExpert

    mx.random.seed(0)
    cfg = BabyWhaleV4Config.tiny(vocab_size=64, context_length=16)
    ln1, ln2 = RMSNorm(cfg.n_embd), RMSNorm(cfg.n_embd)
    attn = SlidingMQAAttention(cfg, 0)
    ffn = SwiGLUExpert(cfg.n_embd, cfg.moe_intermediate_size)
    attn.eval()
    ffn.eval()
    x = mx.random.normal((1, 8, cfg.n_embd))
    out = fn(x, ln1, attn, ln2, ffn)
    assert tuple(out.shape) == tuple(x.shape), "a layer preserves shape"
    assert bool(mx.allclose(out, transformer_layer_reference(x, ln1, attn, ln2, ffn), atol=1e-4)), (
        "pre-norm residual: h = x + attn(ln1(x)); out = h + ffn(ln2(h))"
    )
