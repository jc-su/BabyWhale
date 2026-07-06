from collections.abc import Iterator
from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map

from baby_whale_v4.attention import build_attention
from baby_whale_v4.cache import DynamicKVCache, KVCache
from baby_whale_v4.config import BabyWhaleV4Config
from baby_whale_v4.layers import RMSNorm, WhaleLinear
from baby_whale_v4.mhc import HyperConnect
from baby_whale_v4.moe import SparseMoE
from baby_whale_v4.mtp import MTPHead
from baby_whale_v4.vision import VisionMLPConnector


@dataclass(frozen=True)
class SpecDecodeResult:
    """Output of :meth:`BabyWhaleV4Model.spec_decode`.

    ``tokens`` is the generated sequence (prompt + completion). The other
    fields are diagnostic counters used to compute the Medusa/EAGLE
    acceptance rate: ``n_drafts_accepted / n_drafts_proposed``. A higher
    acceptance rate means fewer verification calls per emitted token.
    """

    tokens: mx.array
    n_drafts_proposed: int
    n_drafts_accepted: int
    n_verify_calls: int

    @property
    def acceptance_rate(self) -> float:
        if self.n_drafts_proposed == 0:
            return 0.0
        return self.n_drafts_accepted / self.n_drafts_proposed


@dataclass(frozen=True)
class BabyWhaleV4Output:
    logits: mx.array
    loss: mx.array | None = None
    cache: KVCache | None = None
    hidden: mx.array | None = None
    mtp_logits: list[mx.array] = field(default_factory=list)
    main_loss: mx.array | None = None
    mtp_losses: list[mx.array] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.logits, mx.array):
            raise TypeError("logits must be an MLX array")
        for name, value in (
            ("loss", self.loss),
            ("main_loss", self.main_loss),
            ("hidden", self.hidden),
        ):
            if value is not None and not isinstance(value, mx.array):
                raise TypeError(f"{name} must be an MLX array when present")
        if not all(isinstance(value, mx.array) for value in self.mtp_logits):
            raise TypeError("mtp_logits must contain only MLX arrays")
        if not all(isinstance(value, mx.array) for value in self.mtp_losses):
            raise TypeError("mtp_losses must contain only MLX arrays")


def cross_entropy_ignore(logits: mx.array, targets: mx.array, ignore_index: int = -1) -> mx.array:
    if logits.ndim < 2:
        raise ValueError("logits must have at least 2 dims")
    if targets.shape != logits.shape[:-1]:
        raise ValueError("targets shape must match logits without vocab dim")
    vocab = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocab)
    flat_targets = targets.reshape(-1)
    valid = mx.not_equal(flat_targets, ignore_index)
    safe_targets = mx.where(valid, flat_targets, mx.zeros_like(flat_targets))
    losses = nn.losses.cross_entropy(flat_logits, safe_targets, reduction="none")
    weights = valid.astype(losses.dtype)
    denom = mx.maximum(mx.sum(weights), 1.0)
    return mx.sum(losses * weights) / denom


def _dtype_for_precision(precision: str) -> mx.Dtype:
    match precision:
        case "fp32":
            return mx.float32
        case "fp16":
            return mx.float16
        case "bf16":
            return mx.bfloat16
        case _:
            raise ValueError(f"unsupported precision {precision!r}")


class BabyWhaleV4Block(nn.Module):
    def __init__(self, config: BabyWhaleV4Config, layer_idx: int, hc: HyperConnect):
        super().__init__()
        self.layer_idx = layer_idx
        self.ln_1 = RMSNorm(config.n_embd, config.rms_norm_eps)
        self.attn = build_attention(config, layer_idx)
        self.ln_2 = RMSNorm(config.n_embd, config.rms_norm_eps)
        self.moe = SparseMoE(config, layer_idx)
        self.hc = hc

    def __call__(
        self,
        h: mx.array,
        input_ids: mx.array,
        cache: KVCache | None = None,
        *,
        positions: mx.array | None = None,
        key_mask: mx.array | None = None,
    ) -> mx.array:
        x = self.hc.consume(h, layer_idx=self.layer_idx, sublayer_idx=0)
        x = self.ln_1(x)
        delta_a = self.attn(x, cache=cache, positions=positions, key_mask=key_mask)
        h = self.hc.produce(h, delta_a, layer_idx=self.layer_idx, sublayer_idx=0)

        x = self.hc.consume(h, layer_idx=self.layer_idx, sublayer_idx=1)
        x = self.ln_2(x)
        delta_m = self.moe(x, input_ids=input_ids)
        h = self.hc.produce(h, delta_m, layer_idx=self.layer_idx, sublayer_idx=1)
        return h


class BabyWhaleV4Model(nn.Module):
    def __init__(self, config: BabyWhaleV4Config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.hc = HyperConnect(config.hc_mult, config.n_layer)
        self.blocks = {
            f"layer_{i}": BabyWhaleV4Block(config, i, self.hc) for i in range(config.n_layer)
        }
        self.norm = RMSNorm(config.n_embd, config.rms_norm_eps)
        self.lm_head = WhaleLinear(
            config.n_embd,
            config.vocab_size,
            bias=False,
            quant_mode=config.quant_mode,
            placement="lm_head",
        )
        self.mtp = {
            f"head_{i}": MTPHead(config.n_embd, config.vocab_size, quant_mode=config.quant_mode)
            for i in range(config.mtp_heads)
        }
        self.vision_connector = (
            VisionMLPConnector(config.vision_dim, config.n_embd, dropout=config.vision_dropout)
            if config.enable_vision
            else None
        )
        self._apply_precision(config.precision)
        # Tie after precision cast: tree_map+update re-allocates leaves, so an
        # earlier alias to tok_emb.weight would be silently broken.
        if config.tie_weights:
            self.lm_head.inner.weight = self.tok_emb.weight

    def num_parameters(self) -> int:
        return sum(
            int(array.size)
            for _name, array in tree_flatten(self.parameters())
            if isinstance(array, mx.array)
        )

    def state_dict(self) -> dict:
        return self.parameters()

    def load_state_dict(self, state: dict) -> None:
        self.update(state)

    def _apply_precision(self, precision: str) -> None:
        dtype = _dtype_for_precision(precision)
        self.update(
            tree_map(
                lambda value: (
                    value.astype(dtype)
                    if isinstance(value, mx.array) and mx.issubdtype(value.dtype, mx.floating)
                    else value
                ),
                self.parameters(),
            )
        )

    def modules(self) -> Iterator[object]:
        yield self
        for block in self.blocks.values():
            yield block
            yield block.attn
            yield block.moe
            yield block.moe.shared_expert
            yield from block.moe.experts.values()
        yield from self.mtp.values()
        if self.vision_connector is not None:
            yield self.vision_connector

    def empty_cache(self) -> DynamicKVCache:
        return DynamicKVCache.empty(len(self.blocks))

    @property
    def device(self) -> str:
        return "mlx"

    def _total_sequence_length(self, input_ids: mx.array, cache: KVCache | None) -> int:
        past_len = 0 if cache is None else cache.max_sequence_length()
        return past_len + int(input_ids.shape[1])

    def _prepend_vision(
        self,
        x: mx.array,
        input_ids: mx.array,
        image_features: mx.array,
        targets: mx.array | None,
        cache: KVCache | None,
    ) -> tuple[mx.array, mx.array]:
        """Project encoder tile features through the connector and prepend them to
        the token-embedding stream (DeepSeek-VL2 layout: image tokens first).

        Inference-only first slice: loss/cache with images are a later milestone,
        so ``targets`` and ``cache`` must be ``None``. Vision positions get a
        placeholder id for MoE hash-routing. Text-only forwards (``image_features
        is None``) never reach here and stay bit-identical.
        """
        connector = self.vision_connector
        if connector is None:
            raise ValueError("image_features given but config.enable_vision is False")
        if targets is not None or cache is not None:
            raise ValueError("vision forward is inference-only (targets and cache must be None)")
        if image_features.ndim != 3 or image_features.shape[-1] != self.config.vision_dim:
            raise ValueError("image_features must be [batch, n_image_tokens, config.vision_dim]")
        vis = connector(image_features)  # [B, n_vis, n_embd]
        combined = mx.concatenate([vis, x], axis=1)
        pad_ids = mx.zeros((input_ids.shape[0], vis.shape[1]), dtype=input_ids.dtype)
        block_ids = mx.concatenate([pad_ids, input_ids], axis=1)
        if block_ids.shape[1] > self.config.context_length:
            raise ValueError("vision + text sequence exceeds config.context_length")
        return combined, block_ids

    def __call__(
        self,
        input_ids: mx.array,
        targets: mx.array | None = None,
        cache: KVCache | None = None,
        *,
        mtp_loss_weight: float = 0.1,
        positions: mx.array | None = None,
        key_mask: mx.array | None = None,
        image_features: mx.array | None = None,
    ) -> BabyWhaleV4Output:
        if not isinstance(input_ids, mx.array):
            raise TypeError("input_ids must be an mlx.core.array")
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq]")
        if self._total_sequence_length(input_ids, cache) > self.config.context_length:
            raise ValueError("input sequence exceeds config.context_length")
        if self.config.activation_checkpoint and cache is not None:
            raise ValueError("activation_checkpoint cannot be used with a mutable KV cache")

        x = self.drop(self.tok_emb(input_ids))
        block_ids = input_ids
        if image_features is not None:
            x, block_ids = self._prepend_vision(x, input_ids, image_features, targets, cache)
        h = self.hc.expand(x)
        for block in self.blocks.values():
            if self.config.activation_checkpoint:

                def run_block(
                    hidden: mx.array,
                    *,
                    current_block: BabyWhaleV4Block = block,
                ) -> mx.array:
                    return current_block(hidden, input_ids=block_ids, cache=None)

                h = mx.checkpoint(run_block)(h)
            else:
                h = block(
                    h, input_ids=block_ids, cache=cache, positions=positions, key_mask=key_mask
                )
        x = self.hc.reduce(h)
        x = self.norm(x)
        logits = self.lm_head(x)

        mtp_logits = [head(x) for head in self.mtp.values()]

        main_loss: mx.array | None = None
        mtp_losses: list[mx.array] = []
        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError("targets must have the same shape as input_ids")
            main_loss = cross_entropy_ignore(logits, targets)
            T = targets.shape[1]
            for k, mtp_l in enumerate(mtp_logits):
                shift = k + 1
                if T - shift <= 0:
                    mtp_losses.append(mx.array(0.0, dtype=logits.dtype))
                    continue
                pred = mtp_l[:, : T - shift, :]
                tgt = targets[:, shift:]
                mtp_losses.append(cross_entropy_ignore(pred, tgt))

        loss: mx.array | None = None
        if main_loss is not None:
            loss = main_loss + (
                mtp_loss_weight * sum(mtp_losses, mx.array(0.0, dtype=main_loss.dtype))
                if mtp_losses
                else 0.0
            )

        return BabyWhaleV4Output(
            logits=logits,
            loss=loss,
            cache=cache,
            hidden=x,
            mtp_logits=mtp_logits,
            main_loss=main_loss,
            mtp_losses=mtp_losses,
        )

    def spec_decode(self, input_ids: mx.array, max_new_tokens: int) -> "SpecDecodeResult":
        """Greedy speculative decode via MTP draft heads (Medusa/EAGLE family).

        Each MTP head is a learned projection from the *last hidden state*
        (not from the token ID) to a token distribution at position t+k+1,
        so the draft tokens are conditioned on the same rich representation
        the LM head uses — the Medusa/EAGLE recipe. We accept the first run
        of drafts that the parallel-verifier confirms; the result is
        bit-identical to plain greedy.

        Returns a ``SpecDecodeResult`` with the generated sequence, the
        number of speculative tokens proposed, and the number accepted so
        callers can compute an acceptance rate — the standard
        Medusa/EAGLE diagnostic. Acceptance rate is the educational
        signal: ~0.5 is Medusa-typical, ~0.7-0.8 is EAGLE-2 territory.
        """

        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if input_ids.shape[0] != 1:
            raise ValueError("spec_decode supports batch size 1 only in this educational impl")
        if input_ids.shape[1] + max_new_tokens > self.config.context_length:
            raise ValueError("generation would exceed config.context_length")
        m = len(self.mtp)
        seq = input_ids
        produced = 0
        n_drafts_proposed = 0
        n_drafts_accepted = 0
        n_verify_calls = 0
        while produced < max_new_tokens:
            out = self(seq)
            if out.hidden is None:
                raise RuntimeError("model did not return hidden state")
            last_h = out.hidden[:, -1, :]
            main_draft = mx.argmax(self.lm_head(last_h), axis=-1).reshape(1, 1)
            drafts = [main_draft]
            for k in range(m):
                drafts.append(mx.argmax(self.mtp[f"head_{k}"](last_h), axis=-1).reshape(1, 1))
            draft_seq = mx.concatenate(drafts, axis=1)
            if m == 0 or produced + 1 >= max_new_tokens:
                seq = mx.concatenate([seq, drafts[0]], axis=1)
                produced += 1
                continue

            extended = mx.concatenate([seq, draft_seq], axis=1)
            out2 = self(extended)
            n_verify_calls += 1
            verify_logits = out2.logits[:, seq.shape[1] : seq.shape[1] + m, :]

            accepted = [drafts[0]]
            n_drafts_proposed += m
            for k in range(m):
                verified = mx.argmax(verify_logits[:, k, :], axis=-1).reshape(1, 1)
                if bool(mx.array_equal(verified, drafts[k + 1])):
                    accepted.append(drafts[k + 1])
                    n_drafts_accepted += 1
                else:
                    accepted.append(verified)
                    break

            take = min(len(accepted), max_new_tokens - produced)
            for tok in accepted[:take]:
                seq = mx.concatenate([seq, tok], axis=1)
                produced += 1
        return SpecDecodeResult(
            tokens=seq,
            n_drafts_proposed=n_drafts_proposed,
            n_drafts_accepted=n_drafts_accepted,
            n_verify_calls=n_verify_calls,
        )
