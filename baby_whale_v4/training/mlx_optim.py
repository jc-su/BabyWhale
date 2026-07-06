from dataclasses import dataclass, field
from typing import cast

import mlx.core as mx
from mlx.utils import tree_flatten, tree_map, tree_unflatten


@dataclass
class AdamWState:
    step: int = 0
    m: dict[str, mx.array] = field(default_factory=dict)
    v: dict[str, mx.array] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError("AdamWState.step must be non-negative")


@dataclass
class AdafactorState:
    step: int = 0
    row: dict[str, mx.array] = field(default_factory=dict)
    col: dict[str, mx.array] = field(default_factory=dict)
    v: dict[str, mx.array] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError("AdafactorState.step must be non-negative")


@dataclass
class MuonState:
    step: int = 0
    momentum: dict[str, mx.array] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError("MuonState.step must be non-negative")


class AdamW:
    def __init__(
        self,
        *,
        learning_rate: float,
        betas: tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.0,
        eps: float = 1e-8,
    ):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not (0 <= betas[0] < 1 and 0 <= betas[1] < 1):
            raise ValueError("betas must be in [0, 1)")
        if weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")
        self.learning_rate = learning_rate
        self.betas = betas
        self.weight_decay = weight_decay
        self.eps = eps
        self.state = AdamWState()

    def state_dict(self) -> dict[str, object]:
        return {"step": self.state.step, "m": self.state.m, "v": self.state.v}

    def load_state_dict(self, state: dict[str, object]) -> None:
        step = state["step"]
        m = state["m"]
        v = state["v"]
        if type(step) is not int:
            raise TypeError("optimizer step must be an integer")
        if not isinstance(m, dict) or not isinstance(v, dict):
            raise TypeError("optimizer m/v state must be dictionaries")
        self.state.step = step
        self.state.m = _array_state_dict(cast(dict[object, object], m), "m")
        self.state.v = _array_state_dict(cast(dict[object, object], v), "v")

    def step(self, params: dict[str, object], grads: dict[str, object]) -> dict[str, object]:
        self.state.step += 1
        b1, b2 = self.betas
        lr = self.learning_rate
        grad_by_path = dict(tree_flatten(grads))
        updates: list[tuple[str, mx.array]] = []
        for path, param in tree_flatten(params):
            if not isinstance(param, mx.array):
                continue
            grad = grad_by_path.get(path)
            if grad is None:
                updates.append((path, param))
                continue
            if not isinstance(grad, mx.array):
                raise TypeError(f"gradient {path} must be an MLX array")
            m = self.state.m.get(path, mx.zeros_like(param))
            v = self.state.v.get(path, mx.zeros_like(param))
            m = b1 * m + (1.0 - b1) * grad
            v = b2 * v + (1.0 - b2) * mx.square(grad)
            self.state.m[path] = m
            self.state.v[path] = v
            m_hat = m / (1.0 - b1**self.state.step)
            v_hat = v / (1.0 - b2**self.state.step)
            decayed = param * (1.0 - lr * self.weight_decay)
            updates.append((path, decayed - lr * m_hat / (mx.sqrt(v_hat) + self.eps)))
        return tree_unflatten(updates)


class Adafactor:
    """Factored second-moment optimizer for memory-constrained dense training."""

    def __init__(
        self,
        *,
        learning_rate: float,
        beta2: float = 0.999,
        weight_decay: float = 0.0,
        eps: float = 1e-30,
        update_clip: float = 1.0,
    ):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= beta2 < 1:
            raise ValueError("beta2 must be in [0, 1)")
        if weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")
        if eps <= 0:
            raise ValueError("eps must be positive")
        if update_clip <= 0:
            raise ValueError("update_clip must be positive")
        self.learning_rate = learning_rate
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.eps = eps
        self.update_clip = update_clip
        self.state = AdafactorState()

    def state_dict(self) -> dict[str, object]:
        return {
            "step": self.state.step,
            "row": self.state.row,
            "col": self.state.col,
            "v": self.state.v,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        step = state["step"]
        row = state["row"]
        col = state["col"]
        v = state["v"]
        if type(step) is not int:
            raise TypeError("optimizer step must be an integer")
        if not isinstance(row, dict) or not isinstance(col, dict) or not isinstance(v, dict):
            raise TypeError("optimizer row/col/v state must be dictionaries")
        self.state.step = step
        self.state.row = _array_state_dict(cast(dict[object, object], row), "row")
        self.state.col = _array_state_dict(cast(dict[object, object], col), "col")
        self.state.v = _array_state_dict(cast(dict[object, object], v), "v")

    def step(self, params: dict[str, object], grads: dict[str, object]) -> dict[str, object]:
        self.state.step += 1
        grad_by_path = dict(tree_flatten(grads))
        updates: list[tuple[str, mx.array]] = []
        for path, param in tree_flatten(params):
            if not isinstance(param, mx.array):
                continue
            grad = grad_by_path.get(path)
            if grad is None:
                updates.append((path, param))
                continue
            if not isinstance(grad, mx.array):
                raise TypeError(f"gradient {path} must be an MLX array")
            update = self._adaptive_update(path, grad.astype(mx.float32))
            update = _clip_update(update, self.update_clip)
            decayed = param.astype(mx.float32) * (1.0 - self.learning_rate * self.weight_decay)
            next_param = decayed - self.learning_rate * update
            updates.append((path, next_param.astype(param.dtype)))
        return tree_unflatten(updates)

    def _adaptive_update(self, path: str, grad: mx.array) -> mx.array:
        grad2 = mx.square(grad) + self.eps
        if grad.ndim >= 2:
            row_grad = mx.mean(grad2, axis=-1)
            col_grad = mx.mean(grad2, axis=-2)
            row = self.state.row.get(path, mx.zeros_like(row_grad))
            col = self.state.col.get(path, mx.zeros_like(col_grad))
            row = self.beta2 * row + (1.0 - self.beta2) * row_grad
            col = self.beta2 * col + (1.0 - self.beta2) * col_grad
            self.state.row[path] = row
            self.state.col[path] = col
            row_mean = mx.mean(row, axis=-1, keepdims=True)
            factored = (
                row[..., :, None]
                * col[..., None, :]
                / mx.maximum(
                    row_mean[..., None],
                    self.eps,
                )
            )
            return grad * mx.rsqrt(factored + self.eps)

        v = self.state.v.get(path, mx.zeros_like(grad))
        v = self.beta2 * v + (1.0 - self.beta2) * grad2
        self.state.v[path] = v
        return grad * mx.rsqrt(v + self.eps)


class Muon:
    """Educational Muon-style optimizer.

    2D parameters use momentum followed by a Newton-Schulz orthogonalized
    update. Non-matrix parameters use SGD with momentum, which keeps the
    optimizer fail-fast and usable for small MLX models without pretending to
    match a production Muon implementation exactly.
    """

    def __init__(
        self,
        *,
        learning_rate: float,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
    ):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= momentum < 1:
            raise ValueError("momentum must be in [0, 1)")
        if weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")
        if ns_steps <= 0:
            raise ValueError("ns_steps must be positive")
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.ns_steps = ns_steps
        self.state = MuonState()

    def state_dict(self) -> dict[str, object]:
        return {"step": self.state.step, "momentum": self.state.momentum}

    def load_state_dict(self, state: dict[str, object]) -> None:
        step = state["step"]
        momentum_state = state["momentum"]
        if type(step) is not int:
            raise TypeError("optimizer step must be an integer")
        if not isinstance(momentum_state, dict):
            raise TypeError("optimizer momentum state must be a dictionary")
        self.state.step = step
        self.state.momentum = _array_state_dict(
            cast(dict[object, object], momentum_state), "momentum"
        )

    def step(self, params: dict[str, object], grads: dict[str, object]) -> dict[str, object]:
        self.state.step += 1
        grad_by_path = dict(tree_flatten(grads))
        updates: list[tuple[str, mx.array]] = []
        for path, param in tree_flatten(params):
            if not isinstance(param, mx.array):
                continue
            grad = grad_by_path.get(path)
            if grad is None:
                updates.append((path, param))
                continue
            if not isinstance(grad, mx.array):
                raise TypeError(f"gradient {path} must be an MLX array")
            buf = self.state.momentum.get(path, mx.zeros_like(param).astype(mx.float32))
            buf = self.momentum * buf + grad.astype(mx.float32)
            self.state.momentum[path] = buf
            update = _orthogonalized_update(buf, self.ns_steps) if param.ndim == 2 else buf
            decayed = param.astype(mx.float32) * (1.0 - self.learning_rate * self.weight_decay)
            next_param = decayed - self.learning_rate * update
            updates.append((path, next_param.astype(param.dtype)))
        return tree_unflatten(updates)


def _orthogonalized_update(grad: mx.array, ns_steps: int) -> mx.array:
    if grad.ndim != 2:
        raise ValueError("orthogonalized update expects a matrix")
    x = grad.astype(mx.float32)
    if x.shape[0] > x.shape[1]:
        x = x.T
        transposed = True
    else:
        transposed = False
    x = x / (mx.sqrt(mx.sum(mx.square(x))) + 1e-7)
    for _ in range(ns_steps):
        a = x @ x.T
        x = 1.5 * x - 0.5 * (a @ x)
    out = x.T if transposed else x
    return out.astype(grad.dtype)


def _clip_update(update: mx.array, update_clip: float) -> mx.array:
    rms = mx.sqrt(mx.mean(mx.square(update.astype(mx.float32))))
    return update / mx.maximum(1.0, rms / update_clip)


def _array_state_dict(raw: dict[object, object], name: str) -> dict[str, mx.array]:
    out: dict[str, mx.array] = {}
    for key, value in raw.items():
        if not isinstance(value, mx.array):
            raise TypeError(f"optimizer {name} state for {key!r} must be an MLX array")
        out[str(key)] = value
    return out


def clip_grad_norm(grads: dict[str, object], max_norm: float) -> dict[str, object]:
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    total = mx.array(0.0, dtype=mx.float32)
    for _path, grad in tree_flatten(grads):
        if isinstance(grad, mx.array):
            total = total + mx.sum(mx.square(grad.astype(mx.float32)))
    norm = mx.sqrt(total)
    scale = mx.minimum(1.0, max_norm / (norm + 1e-6))
    return tree_map(lambda grad: grad * scale if isinstance(grad, mx.array) else grad, grads)
