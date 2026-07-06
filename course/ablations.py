"""Measurable payoffs — the "beat 4" of each module, as importable functions.

Kept here (not in the digit-prefixed module folders, which aren't importable) so
the test suite can green-gate the numbers and the module scripts stay thin.
"""

from __future__ import annotations

from dataclasses import dataclass

from course.systems import attention_flops, weight_memory_mb


@dataclass(frozen=True)
class KVRow:
    label: str
    bytes_per_token: int
    ratio_vs_mha: float


def mla_kv_cache_rows(
    *, n_head: int = 8, head_dim: int = 64, kv_lora_rank: int = 128, bytes_per_elem: int = 2
) -> list[KVRow]:
    """Bytes/token/layer for MHA vs MQA vs MLA at a realistic layer size.

    MHA caches K and V for every head; MQA shares one KV head; MLA caches only a
    low-rank latent. MLA lands at ~MQA size while (by reconstructing per-head)
    keeping MHA-quality attention — that's the whole pitch.
    """
    mha = n_head * head_dim * 2 * bytes_per_elem
    mqa = 1 * head_dim * 2 * bytes_per_elem
    mla = kv_lora_rank * bytes_per_elem
    return [
        KVRow("MHA (n_kv_head=8)", mha, mha / mha),
        KVRow("MQA (n_kv_head=1)", mqa, mha / mqa),
        KVRow(f"MLA (latent={kv_lora_rank})", mla, mha / mla),
    ]


def print_mla_kv_cache() -> None:
    rows = mla_kv_cache_rows()
    print("KV cache, bytes/token/layer (bf16):\n")
    for row in rows:
        print(
            f"  {row.label:22s} {row.bytes_per_token:5d}   {row.ratio_vs_mha:4.1f}x smaller than MHA"
        )
    print("\nMLA caches ~MQA-size but reconstructs per-head K/V -> MHA-quality attention.")


# --- Module 02: attention cost, full vs windowed ---------------------------
def attention_cost_rows(
    *,
    d_model: int = 512,
    window: int = 256,
    seq_lens: tuple[int, ...] = (256, 512, 1024, 2048, 4096),
) -> list[tuple[int, int, int, float]]:
    rows = []
    for n in seq_lens:
        full = attention_flops(n, d_model)
        windowed = attention_flops(n, d_model, window=window)
        rows.append((n, full, windowed, full / windowed))
    return rows


def print_attention_cost(*, window: int = 256) -> None:
    print(f"attention MACs/layer as sequence length grows (d=512, window={window}):\n")
    print(f"  {'seq_len':>8}  {'full O(n²)':>16}  {'windowed O(n)':>16}  ratio")
    for n, full, win, ratio in attention_cost_rows(window=window):
        print(f"  {n:>8}  {full:>16,}  {win:>16,}  {ratio:>4.0f}x")
    print("\nFull attention's cost explodes quadratically; the window keeps it linear.")


# --- Module 05: MoE, capacity decoupled from FLOPs -------------------------
@dataclass(frozen=True)
class MoERow:
    label: str
    params: int
    active_units: int


def moe_params_rows(
    *, d_model: int = 512, hidden: int = 2048, n_expert: int = 8, k: int = 2
) -> list[MoERow]:
    per_expert = 3 * d_model * hidden  # SwiGLU: gate, up, down
    return [
        MoERow("dense FFN", per_expert, per_expert),
        MoERow(f"MoE ({n_expert} experts, top-{k})", n_expert * per_expert, k * per_expert),
    ]


def print_moe_params(*, n_expert: int = 8, k: int = 2) -> None:
    rows = moe_params_rows(n_expert=n_expert, k=k)
    dense = rows[0]
    print(f"one FFN layer, dense vs MoE ({n_expert} experts, top-{k}):\n")
    for r in rows:
        p_mult = r.params / dense.params
        f_mult = r.active_units / dense.active_units
        print(f"  {r.label:26s} params {p_mult:>3.0f}x    active FLOPs {f_mult:>3.0f}x")
    print(
        f"\nMoE buys {n_expert}x the parameters at {k}x the compute — capacity decoupled from FLOPs."
    )


# --- Module 14: decode work with vs without a KV cache ---------------------
def kv_decode_rows(
    *, seq_lens: tuple[int, ...] = (64, 256, 1024, 4096)
) -> list[tuple[int, int, int, float]]:
    rows = []
    for n in seq_lens:
        no_cache = n * (n + 1) // 2  # reprocess the growing prefix every step
        with_cache = n  # one new token per step
        rows.append((n, no_cache, with_cache, no_cache / with_cache))
    return rows


def print_kv_decode() -> None:
    print("token-forwards to generate n tokens (all layers), with vs without a KV cache:\n")
    print(f"  {'n':>6}  {'no cache':>16}  {'with cache':>12}  saving")
    for n, no_cache, with_cache, ratio in kv_decode_rows():
        print(f"  {n:>6}  {no_cache:>16,}  {with_cache:>12,}  {ratio:>5.0f}x")
    print(
        "\nWithout a cache each step reprocesses the whole prefix -> O(n²); the cache makes it O(n)."
    )


# --- Module 16: speculative decoding, expected tokens per forward ----------
def spec_tokens_rows(
    *, k: int = 4, accept_probs: tuple[float, ...] = (0.3, 0.5, 0.7, 0.9)
) -> list[tuple[float, float]]:
    # expected emitted tokens per verify forward, drafting k with per-token accept prob p
    return [(p, (1 - p ** (k + 1)) / (1 - p)) for p in accept_probs]


def print_spec_tokens(*, k: int = 4) -> None:
    print(f"expected tokens per verify forward, drafting k={k}:\n")
    print(f"  {'accept p':>9}  tokens/forward")
    for p, expected in spec_tokens_rows(k=k):
        print(f"  {p:>9.1f}  {expected:>6.2f}   (~{expected:.1f}x fewer forwards than greedy)")
    print("\nGreedy is 1 token/forward; speculation emits several when drafts are accepted.")


# --- Module 18: quantization, bf16 vs fp4 weight memory --------------------
def quant_memory_rows(
    *, param_counts: tuple[float, ...] = (6.2e6, 7e9, 70e9)
) -> list[tuple[int, float, float, float]]:
    rows = []
    for p in param_counts:
        bf16 = weight_memory_mb(int(p), "bf16") / 1000.0  # GB
        fp4 = weight_memory_mb(int(p), "fp4") / 1000.0
        rows.append((int(p), bf16, fp4, bf16 / fp4))
    return rows


def print_quant_memory() -> None:
    print("weight memory, bf16 vs fp4:\n")
    print(f"  {'params':>16}  {'bf16 (GB)':>10}  {'fp4 (GB)':>9}  smaller")
    for params, bf16, fp4, ratio in quant_memory_rows():
        print(f"  {params:>16,}  {bf16:>10.2f}  {fp4:>9.2f}  {ratio:>3.0f}x")
    print("\n4x smaller weights = 4x less bandwidth to move them — the on-device win.")
