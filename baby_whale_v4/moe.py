from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn

from baby_whale_v4.layers import SwiGLUExpert, WhaleLinear


@dataclass
class _RouterBalancerState:
    """Non-array container so the bias stays out of MLX's parameter tree."""

    values: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class MoERouteMetrics:
    total_tokens: int
    n_expert: int
    used_experts: int
    counts: tuple[int, ...]
    load_balance_loss: float
    bias: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.total_tokens <= 0:
            raise ValueError("total_tokens must be positive")
        if self.n_expert <= 0:
            raise ValueError("n_expert must be positive")
        if len(self.counts) != self.n_expert:
            raise ValueError("counts length must match n_expert")
        if len(self.bias) != self.n_expert:
            raise ValueError("bias length must match n_expert")
        if self.used_experts < 0 or self.used_experts > self.n_expert:
            raise ValueError("used_experts must be in [0, n_expert]")
        if self.load_balance_loss < 0:
            raise ValueError("load_balance_loss must be non-negative")


class SparseMoE(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_expert = config.n_expert
        self.experts_per_token = config.experts_per_token
        self.hash_routing = layer_idx < config.n_hash_layers
        self.shared_expert = SwiGLUExpert(
            config.n_embd,
            config.moe_intermediate_size,
            config.swiglu_clamp,
            quant_mode=config.quant_mode,
            placement="moe_expert",
        )
        self.experts = {
            f"expert_{i}": SwiGLUExpert(
                config.n_embd,
                config.moe_intermediate_size,
                config.swiglu_clamp,
                quant_mode=config.quant_mode,
                placement="moe_expert",
            )
            for i in range(config.n_expert)
        }
        self.router = WhaleLinear(
            config.n_embd,
            config.n_expert,
            bias=False,
            quant_mode="none",
            placement="router",
        )
        # DeepSeek-V3/V4 aux-loss-free balancing: per-expert bias that shifts
        # top-k selection only (gating weights still come from raw scores).
        # Wrapped in a dataclass so MLX's tree_flatten skips it (it walks lists
        # and dicts but treats unknown classes as opaque leaves) — the bias
        # never enters the parameter tree, so optimizer + weight decay can't
        # touch it.
        self.aux_free_bias_rate: float = float(config.aux_free_bias_rate)
        self._balancer = _RouterBalancerState(values=[0.0] * config.n_expert)

    @property
    def router_bias(self) -> tuple[float, ...]:
        return tuple(self._balancer.values)

    def _bias_array(self, dtype: mx.Dtype) -> mx.array:
        return mx.array(self._balancer.values, dtype=dtype)

    def _hash_routes(self, input_ids: mx.array) -> tuple[mx.array, mx.array]:
        routes = []
        for offset in range(self.experts_per_token):
            routes.append((input_ids + offset * 131) % self.n_expert)
        indices = mx.stack(routes, axis=-1)
        weights = mx.full(indices.shape, 1.0 / self.experts_per_token, dtype=mx.float32)
        return indices, weights

    def _learned_routes(self, x: mx.array) -> tuple[mx.array, mx.array]:
        router_logits = self.router(x)
        scores = mx.sqrt(nn.softplus(router_logits))
        if self.aux_free_bias_rate > 0.0:
            biased = scores + self._bias_array(scores.dtype)
            indices = mx.argsort(biased, axis=-1)[..., -self.experts_per_token :]
        else:
            indices = mx.argsort(scores, axis=-1)[..., -self.experts_per_token :]
        # Gating weights always use the raw scores so the bias never appears in
        # the autograd graph or in the produced expert mixture.
        values = mx.take_along_axis(scores, indices, axis=-1)
        denom = mx.maximum(mx.sum(values, axis=-1, keepdims=True), 1e-12)
        weights = values / denom
        return indices, weights

    def _maybe_update_bias(self, indices: mx.array) -> None:
        if not self.training or self.aux_free_bias_rate <= 0.0 or self.hash_routing:
            return
        flat = indices.reshape(-1).tolist()
        if not isinstance(flat, list):
            return
        counts = [0] * self.n_expert
        for value in flat:
            counts[int(value)] += 1
        if sum(counts) == 0:
            return
        mean = sum(counts) / self.n_expert
        bias = self._balancer.values
        for i in range(self.n_expert):
            if counts[i] > mean:
                bias[i] -= self.aux_free_bias_rate
            elif counts[i] < mean:
                bias[i] += self.aux_free_bias_rate

    def __call__(self, x: mx.array, input_ids: mx.array) -> mx.array:
        if self.hash_routing:
            indices, weights = self._hash_routes(input_ids)
        else:
            indices, weights = self._learned_routes(x)
            self._maybe_update_bias(indices)

        flat_x = x.reshape(-1, x.shape[-1])
        flat_indices = indices.reshape(-1, self.experts_per_token)
        flat_weights = weights.reshape(-1, self.experts_per_token).astype(x.dtype)
        flat_out = self.shared_expert(flat_x)

        for slot in range(self.experts_per_token):
            slot_indices = flat_indices[:, slot]
            slot_weights = flat_weights[:, slot][:, None]
            for expert_idx, expert in enumerate(self.experts.values()):
                selected = mx.equal(slot_indices, expert_idx).astype(x.dtype)[:, None]
                flat_out = flat_out + expert(flat_x) * slot_weights * selected

        return flat_out.reshape(x.shape)

    def route_metrics(self, x: mx.array, input_ids: mx.array) -> MoERouteMetrics:
        if self.hash_routing:
            indices, _weights = self._hash_routes(input_ids)
        else:
            indices, _weights = self._learned_routes(x)
        flat = indices.reshape(-1).tolist()
        if not isinstance(flat, list):
            raise TypeError("MoE route indices must flatten to a list")
        raw = [int(v) for v in flat]
        if not raw:
            raise ValueError("cannot compute MoE metrics for zero routed tokens")
        counts = tuple(raw.count(i) for i in range(self.n_expert))
        total = sum(counts)
        expected = total / self.n_expert
        balance = (
            sum(((count - expected) / max(expected, 1.0)) ** 2 for count in counts) / self.n_expert
        )
        return MoERouteMetrics(
            total_tokens=total,
            n_expert=self.n_expert,
            used_experts=sum(1 for count in counts if count > 0),
            counts=counts,
            load_balance_loss=balance,
            bias=tuple(self._balancer.values),
        )
