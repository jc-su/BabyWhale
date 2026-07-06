import math

import mlx.core as mx
import mlx.nn as nn

from baby_whale_v4.cache import KVCache
from baby_whale_v4.config import BabyWhaleV4Config
from baby_whale_v4.layers import PartialRotaryEmbedding, WhaleLinear
from baby_whale_v4.typing import AttentionKind, assert_never


def _masked_softmax(scores: mx.array, mask: mx.array) -> mx.array:
    floor = mx.full(scores.shape, -1e9, dtype=scores.dtype)
    return mx.softmax(mx.where(mask, scores, floor), axis=-1)


class _AttentionBase(nn.Module):
    """Shared QKV projection + RoPE + cache append. Subclasses implement `_attend`."""

    def __init__(self, config: BabyWhaleV4Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.head_dim
        self.sliding_window = config.sliding_window

        self.q_proj = WhaleLinear(
            config.n_embd,
            config.n_head * self.head_dim,
            bias=False,
            quant_mode=config.quant_mode,
            placement="attention",
        )
        self.k_proj = WhaleLinear(
            config.n_embd,
            config.n_kv_head * self.head_dim,
            bias=False,
            quant_mode=config.quant_mode,
            placement="attention",
        )
        self.v_proj = WhaleLinear(
            config.n_embd,
            config.n_kv_head * self.head_dim,
            bias=False,
            quant_mode=config.quant_mode,
            placement="attention",
        )
        self.o_proj = WhaleLinear(
            config.n_head * self.head_dim,
            config.n_embd,
            bias=False,
            quant_mode=config.quant_mode,
            placement="attention",
        )
        self.rope = PartialRotaryEmbedding(config.head_dim, config.rotary_dim)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def _split_heads(self, x: mx.array, n_head: int) -> mx.array:
        batch, seq_len, _ = x.shape
        return x.reshape(batch, seq_len, n_head, self.head_dim).transpose(0, 2, 1, 3)

    def _expand_kv(self, x: mx.array) -> mx.array:
        if self.n_kv_head == self.n_head:
            return x
        repeat = self.n_head // self.n_kv_head
        return mx.repeat(x, repeat, axis=1)

    def __call__(
        self,
        x: mx.array,
        cache: KVCache | None = None,
        *,
        positions: mx.array | None = None,
        key_mask: mx.array | None = None,
    ) -> mx.array:
        batch, seq_len, _ = x.shape
        if positions is None:
            past_len = 0 if cache is None else cache.sequence_length(self.layer_idx)
            positions = mx.arange(past_len, past_len + seq_len)

        q = self._split_heads(self.q_proj(x), self.n_head)
        k = self._split_heads(self.k_proj(x), self.n_kv_head)
        v = self._split_heads(self.v_proj(x), self.n_kv_head)
        q, k = self.rope(q, k, positions)

        if cache is not None:
            k, v = cache.append(self.layer_idx, k, v)

        out = self._attend(q, k, v, query_positions=positions, key_mask=key_mask)
        out = out.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.n_head * self.head_dim)
        return self.resid_dropout(self.o_proj(out))

    def _attend(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        *,
        query_positions: mx.array,
        key_mask: mx.array | None = None,
    ) -> mx.array:
        raise NotImplementedError


class SlidingMQAAttention(_AttentionBase):
    def _attend(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        *,
        query_positions: mx.array,
        key_mask: mx.array | None = None,
    ) -> mx.array:
        k_e = self._expand_kv(k)
        v_e = self._expand_kv(v)
        scores = (q @ k_e.swapaxes(-2, -1)) / math.sqrt(self.head_dim)
        if key_mask is not None:
            # Ragged batched decode supplies a precomputed [B, 1, T_q, T_k] boolean
            # mask (causal + sliding window + left-pad validity, per row).
            mask = key_mask
        else:
            key_pos = mx.arange(k_e.shape[2])[None, :]
            q_pos = query_positions[:, None]
            causal = key_pos <= q_pos
            local = key_pos >= (q_pos - self.sliding_window + 1)
            mask = (causal & local)[None, None, :, :]
        weights = self.attn_dropout(_masked_softmax(scores, mask))
        return weights @ v_e


def _block_mean_pool(x: mx.array, block_size: int) -> tuple[mx.array, int]:
    B, H, T, D = x.shape
    n_full = T // block_size
    if n_full == 0:
        return mx.zeros((B, H, 0, D), dtype=x.dtype), 0
    keep = n_full * block_size
    pooled = mx.mean(x[:, :, :keep, :].reshape(B, H, n_full, block_size, D), axis=3)
    return pooled, n_full


class HCAAttention(_AttentionBase):
    def __init__(self, config: BabyWhaleV4Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.block_size = config.hca_block_size

    def _attend(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        *,
        query_positions: mx.array,
        key_mask: mx.array | None = None,
    ) -> mx.array:
        if key_mask is not None:
            raise NotImplementedError("ragged batched decode is only supported for sliding_mqa")
        k_e = self._expand_kv(k)
        v_e = self._expand_kv(v)
        T_k = k_e.shape[2]
        comp_k, n_blocks = _block_mean_pool(k_e, self.block_size)
        comp_v, _ = _block_mean_pool(v_e, self.block_size)

        combined_k = mx.concatenate([comp_k, k_e], axis=2) if n_blocks > 0 else k_e
        combined_v = mx.concatenate([comp_v, v_e], axis=2) if n_blocks > 0 else v_e
        scores = (q @ combined_k.swapaxes(-2, -1)) / math.sqrt(self.head_dim)

        q_pos = query_positions[:, None]
        if n_blocks > 0:
            block_end = (mx.arange(n_blocks) + 1) * self.block_size
            comp_allowed = block_end[None, :] <= (q_pos - self.sliding_window + 1)
        else:
            comp_allowed = mx.zeros((query_positions.shape[0], 0), dtype=mx.bool_)

        raw_pos = mx.arange(T_k)[None, :]
        raw_allowed = (raw_pos <= q_pos) & (raw_pos >= (q_pos - self.sliding_window + 1))
        mask = mx.concatenate([comp_allowed, raw_allowed], axis=1)[None, None, :, :]
        weights = self.attn_dropout(_masked_softmax(scores, mask))
        return weights @ combined_v


def _overlap_mean_pool(x: mx.array, block_size: int, stride: int) -> tuple[mx.array, mx.array]:
    B, H, T, D = x.shape
    if block_size > T:
        return mx.zeros((B, H, 0, D), dtype=x.dtype), mx.zeros((0,), dtype=mx.int32)
    starts = list(range(0, T - block_size + 1, stride))
    if not starts:
        return mx.zeros((B, H, 0, D), dtype=x.dtype), mx.zeros((0,), dtype=mx.int32)
    blocks = mx.stack([mx.mean(x[:, :, s : s + block_size, :], axis=2) for s in starts], axis=2)
    end_pos = mx.array([s + block_size for s in starts], dtype=mx.int32)
    return blocks, end_pos


class CSAAttention(_AttentionBase):
    def __init__(self, config: BabyWhaleV4Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.block_size = config.csa_block_size
        self.stride = config.csa_block_stride
        self.topk = config.csa_index_topk
        self.indexer = WhaleLinear(
            self.head_dim,
            self.head_dim,
            bias=False,
            quant_mode=config.quant_mode,
            placement="attention",
        )
        self.dense_debug = False

    def _attend(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        *,
        query_positions: mx.array,
        key_mask: mx.array | None = None,
    ) -> mx.array:
        if key_mask is not None:
            raise NotImplementedError("ragged batched decode is only supported for sliding_mqa")
        B, H, T_q, _D = q.shape
        k_e = self._expand_kv(k)
        v_e = self._expand_kv(v)
        T_k = k_e.shape[2]

        comp_k_pre, end_pos = _overlap_mean_pool(k_e, self.block_size, self.stride)
        comp_v, _ = _overlap_mean_pool(v_e, self.block_size, self.stride)
        n_blocks = comp_k_pre.shape[2]

        q_pos = query_positions[:, None]
        raw_pos = mx.arange(T_k)[None, :]
        raw_allowed_2d = (raw_pos <= q_pos) & (raw_pos >= (q_pos - self.sliding_window + 1))

        if n_blocks == 0:
            scores_local = (q @ k_e.swapaxes(-2, -1)) / math.sqrt(self.head_dim)
            weights = self.attn_dropout(_masked_softmax(scores_local, raw_allowed_2d[None, None]))
            return weights @ v_e

        comp_k_idx = self.indexer(comp_k_pre)
        index_scores = (q @ comp_k_idx.swapaxes(-2, -1)) / math.sqrt(self.head_dim)
        comp_allowed = end_pos[None, :] <= (q_pos - self.sliding_window + 1)
        index_scores = mx.where(
            comp_allowed[None, None, :, :],
            index_scores,
            mx.full(index_scores.shape, -1e9, dtype=index_scores.dtype),
        )

        if self.dense_debug:
            sel_mask = mx.broadcast_to(comp_allowed[None, None, :, :], (B, H, T_q, n_blocks))
        else:
            k_eff = min(self.topk, n_blocks)
            top_indices = mx.argsort(index_scores, axis=-1)[..., -k_eff:]
            block_ids = mx.arange(n_blocks).reshape(1, 1, 1, 1, n_blocks)
            sel_mask = mx.any(mx.equal(top_indices[..., :, None], block_ids), axis=-2)
            sel_mask = sel_mask & comp_allowed[None, None, :, :]

        comp_scores = (q @ comp_k_pre.swapaxes(-2, -1)) / math.sqrt(self.head_dim)
        raw_scores = (q @ k_e.swapaxes(-2, -1)) / math.sqrt(self.head_dim)
        raw_allowed = mx.broadcast_to(raw_allowed_2d[None, None, :, :], (B, H, T_q, T_k))
        full_scores = mx.concatenate([comp_scores, raw_scores], axis=-1)
        full_mask = mx.concatenate([sel_mask, raw_allowed], axis=-1)
        weights = self.attn_dropout(_masked_softmax(full_scores, full_mask))
        combined_v = mx.concatenate([comp_v, v_e], axis=2)
        return weights @ combined_v


class MLAAttention(nn.Module):
    """DeepSeek-style Multi-head Latent Attention.

    Compresses the input into a single low-rank latent ``c_kv`` of dimension
    ``kv_lora_rank`` per token, then up-projects to per-head K and V at attend
    time. The KV cache stores only ``c_kv`` (shape ``[B, T, R]``) instead of
    raw K and V, which is the V2/V3 KV-cache compression idea.

    The implementation here uses a single up-projection (no rope-decoupled
    head dim split). Rotary embeddings are applied on the up-projected K with
    its full position range and on Q with the new query positions.
    """

    def __init__(self, config: BabyWhaleV4Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.sliding_window = config.sliding_window
        self.kv_lora_rank = config.kv_lora_rank

        self.q_proj = WhaleLinear(
            config.n_embd,
            config.n_head * self.head_dim,
            bias=False,
            quant_mode=config.quant_mode,
            placement="attention",
        )
        self.kv_a_proj = WhaleLinear(
            config.n_embd,
            config.kv_lora_rank,
            bias=False,
            quant_mode=config.quant_mode,
            placement="attention",
        )
        self.kv_b_proj = WhaleLinear(
            config.kv_lora_rank,
            2 * config.n_head * self.head_dim,
            bias=False,
            quant_mode=config.quant_mode,
            placement="attention",
        )
        self.o_proj = WhaleLinear(
            config.n_head * self.head_dim,
            config.n_embd,
            bias=False,
            quant_mode=config.quant_mode,
            placement="attention",
        )
        self.rope = PartialRotaryEmbedding(config.head_dim, config.rotary_dim)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def __call__(
        self,
        x: mx.array,
        cache: KVCache | None = None,
        *,
        positions: mx.array | None = None,
        key_mask: mx.array | None = None,
    ) -> mx.array:
        if key_mask is not None:
            raise NotImplementedError("ragged batched decode is only supported for sliding_mqa")
        batch, seq_len, _ = x.shape
        past_len = 0 if cache is None else cache.latent_length(self.layer_idx)
        q_positions = mx.arange(past_len, past_len + seq_len)

        q = self.q_proj(x).reshape(batch, seq_len, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        c_kv_new = self.kv_a_proj(x)  # [B, T, R]

        c_kv_full = cache.append_latent(self.layer_idx, c_kv_new) if cache is not None else c_kv_new
        T_full = c_kv_full.shape[1]

        kv = self.kv_b_proj(c_kv_full)  # [B, T_full, 2 * H * D]
        kv = kv.reshape(batch, T_full, 2, self.n_head, self.head_dim)
        k = kv[:, :, 0, :, :].transpose(0, 2, 1, 3)  # [B, H, T_full, D]
        v = kv[:, :, 1, :, :].transpose(0, 2, 1, 3)

        all_positions = mx.arange(0, T_full)
        k = self.rope.rotate_one(k, all_positions)
        q = self.rope.rotate_one(q, q_positions)

        scores = (q @ k.swapaxes(-2, -1)) / math.sqrt(self.head_dim)
        key_pos = mx.arange(T_full)[None, :]
        q_pos = q_positions[:, None]
        causal = key_pos <= q_pos
        local = key_pos >= (q_pos - self.sliding_window + 1)
        mask = (causal & local)[None, None, :, :]
        weights = self.attn_dropout(_masked_softmax(scores, mask))
        out = weights @ v
        out = out.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.n_head * self.head_dim)
        return self.resid_dropout(self.o_proj(out))


def build_attention(config: BabyWhaleV4Config, layer_idx: int) -> nn.Module:
    schedule = config.effective_layer_schedule
    kind: AttentionKind = schedule[layer_idx]
    match kind:
        case "sliding_mqa":
            return SlidingMQAAttention(config, layer_idx)
        case "hca":
            return HCAAttention(config, layer_idx)
        case "csa":
            return CSAAttention(config, layer_idx)
        case "mla":
            return MLAAttention(config, layer_idx)
        case _:
            assert_never(kind)
