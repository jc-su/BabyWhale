# FP4 Metal Kernel Research

Baby Whale v4 is allowed to break APIs, but it is not allowed to hide slow or fake FP4 paths. This note records what the current custom Metal path does, why the first version was slower, and what the next faster implementation should target.

## Current Result

The first Metal FP4 training experiment used one Metal thread per output element of:

```text
dw = dy.T @ x
```

That was correct but slow. Each output element reread the same `dy` and `x` values from global memory, while MLX's built-in matmul uses tuned tiled GEMM kernels.

The current implementation replaces that scalar kernel with an 8x8 `metal::simdgroup_matrix` kernel in:

```text
baby_whale_v4/kernels/fp4_training.py
```

This compiles through `mx.fast.metal_kernel`, so it builds only the Baby Whale custom kernel at runtime. It does not rebuild MLX.

## Research Notes

MLX custom kernels:

- `mx.fast.metal_kernel` generates the Metal function signature, compiles a Metal library for the kernel, and supports templates, thread/grid attributes, row-contiguous input handling, custom headers, atomic outputs, and custom VJP use.
- The MLX docs warn that every new kernel creates and may JIT-compile a new Metal library, so kernels should be built once and reused.
- MLX dispatches custom kernels with `dispatchThreads`, so `grid` is total thread count and `threadgroup` is the threadgroup shape.

MLX internal GEMM:

- MLX's own Metal GEMM stack uses `metal::simdgroup_matrix`, tiled block loaders, threadgroup memory, 8x8 fragments, and larger block tiles such as 32x32, 64x32, and 64x64.
- For bf16/float16, MLX chooses tile parameters by Apple GPU family and shape. The current MLX default macro often uses `bm=64`, `bn=64`, `bk=16`, `wm=1`, `wn=2` for bf16/half, with alternate `64x32` or `32x64` shapes for transposed or large-K cases.
- MLX's quantized matmul kernels use specialized qmv/qvm/qmm paths, split-K paths, and quantized block loaders. A naive replacement for one `dy.T @ x` matmul will rarely beat the full built-in stack.
- For FP4 forward, MLX already has the important native primitive: `mx.quantize` plus `mx.quantized_matmul`.

Sources:

- [MLX custom Metal kernels docs](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)
- [MLX custom Metal kernel docs source](https://github.com/ml-explore/mlx/blob/main/docs/src/dev/custom_metal_kernels.rst)
- [MLX Metal simdgroup GEMM MMA helper](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/steel/gemm/mma.h)
- [MLX Metal quantized matmul kernels](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/quantized.h)
- [MLX Metal quantized dispatch code](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/quantized.cpp)

## Why The Kernel Is Still Not Enough

The FP4 training step currently includes:

1. Quantize weights to native FP4.
2. Forward with `mx.quantized_matmul`.
3. Backward `dx` with `mx.quantized_matmul`.
4. Backward `dw` with either MLX matmul or the Baby Whale Metal kernel.
5. Bias grad and loss grad.
6. Optimizer update outside this microbenchmark.

Optimizing only step 4 cannot guarantee an end-to-end win. The project should only call the Metal path faster when the whole step is faster and lower-memory, not when one sub-kernel looks good in isolation.

The current training memory benchmark also shows why the gate must compare against bf16 rather than fp32 only:

```bash
conda run -n base uv run baby-whale-v4 bench-fp4-memory --baseline bf16 --max-peak-ratio 1.0
```

On the current local run, the FP4 training path reported `passed: false` with `peak_ratio ~= 1.26`. That means the current path is a real FP4 compute experiment, but it is not yet a memory-efficient FP4 training system.

## Faster Kernel Direction

The next implementation should not be another scalar or one-output-thread kernel. The likely winning path is:

1. Keep MLX native FP4 forward/backward-input matmul.
2. Replace the `dw` kernel with a 32x32 or 64x64 tiled simdgroup kernel, not the current 8x8 one-simdgroup tile.
3. Use multiple simdgroups per threadgroup, matching MLX's Steel GEMM style.
4. Add split-K only when the token batch dimension is large enough to need it.
5. Add a fused optimizer kernel after `dw`: apply AdamW/Lion update directly from `dw` to master weights, then produce or invalidate packed FP4 weights.
6. Cache packed FP4 weights during inference; for training, repack only after optimizer update.
7. Add separate inference-only packed-weight modules that can drop dense weights; otherwise FP4 inference keeps both dense and packed copies and is not memory optimal.

The larger win is not "custom matmul beats MLX matmul." The larger win is "avoid materializing unnecessary intermediate gradients and avoid launching separate kernels for operations that can be fused."

## Performance Gates

The gates should stay strict:

- `fp4-native`: raw MLX FP4 forward/inference only.
- custom-VJP FP4 training: primitive benchmark only, not a model config mode.
- custom Metal FP4 training: primitive benchmark only, allowed only when correctness and performance gates pass.

The current Metal kernel is useful as a research stepping stone, but it is not the default training path because it does not consistently beat MLX matmul across educational model shapes.
