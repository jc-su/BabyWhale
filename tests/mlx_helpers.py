import mlx.core as mx


def finite(x: mx.array) -> bool:
    return bool(mx.all(mx.isfinite(x)))


def close(a: mx.array, b: mx.array, *, atol: float = 1e-5, rtol: float = 1e-5) -> bool:
    return bool(mx.allclose(a, b, atol=atol, rtol=rtol))


def max_abs(a: mx.array) -> float:
    return float(mx.max(mx.abs(a)))
