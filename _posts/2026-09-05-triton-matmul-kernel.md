---
layout: post
title: "Writing a matrix multiplication kernel in Triton — the operation at the heart of every LLM"
date: 2026-09-05
categories: [inference]
tags: [triton, gpu-kernels, matmul, cuda, parallel-computing, llm-inference]
---

Last week I wrote a vector addition kernel in Triton — the simplest possible GPU program. This week: matrix multiplication. The operation that runs thousands of times every time an LLM generates a single token.

Every linear layer in a transformer is a matrix multiply. Q, K, V projections — matmul. The MLP layers — matmul. The output projection — matmul. When the PyTorch profiler showed 1460 kernel calls for 20 generated tokens, most of those were matrix multiplications. This is the kernel that matters.

## Why matmul is different from vector addition

Vector addition is embarrassingly parallel — every element is independent. Matrix multiplication has a dependency: computing one output element requires summing across an entire row and column.

For two matrices A (M×K) and B (K×N), each output element C[i,j] is the dot product of row i of A with column j of B. That dot product requires K multiply-accumulate operations.

The naive approach: for every output element, load a full row from A and a full column from B, multiply element by element, sum. For large matrices this means enormous amounts of data movement — the exact bottleneck the roofline model identifies.

## The tiling strategy

The optimization is tiling — the same idea as FlashAttention's approach to attention computation.

Instead of computing one output element at a time, compute a BLOCK_M × BLOCK_N tile of output elements at once. To compute this tile, you only need a BLOCK_M × BLOCK_K slice of A and a BLOCK_K × BLOCK_N slice of B.

Load one tile of A and one tile of B into fast on-chip memory. Multiply them. Accumulate into the output tile. Move to the next K slice. Repeat.

The key insight: the same data from A and B gets reused across multiple output elements within the tile. Instead of loading each element once per output computation, you load it once and use it BLOCK_N or BLOCK_M times. This is arithmetic intensity — more compute per memory access, which is exactly what the roofline model says you need.

## The kernel

```python
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, accumulator)
```

Each program instance is responsible for one BLOCK_M × BLOCK_N tile of the output. `pid_m` and `pid_n` identify which tile. The loop steps through K in BLOCK_K chunks, loading and accumulating.

## Connection to the full inference stack

Every concept from the past three months connects here.

The roofline model said LLM inference is memory-bandwidth-bound. This kernel shows why — and how tiling attacks that bottleneck by increasing arithmetic intensity.

The PyTorch profiler showed 1460 kernel launches. Each `aten::addmm` call in that output was launching something like this kernel. The 916ms of CPU overhead vs 43ms of GPU compute makes sense now — the kernel itself is fast, the overhead of launching it 1460 times is what costs time.

FlashAttention applies the same tiling strategy to the attention computation — keeping the attention scores in on-chip memory rather than writing them to HBM and reading them back. Same principle, more complex implementation.

## Full notebook

[github.com/anirudhs16/llm-inference-benchmarks](https://github.com/anirudhs16/llm-inference-benchmarks)

---

*Previous: [I wrote my first GPU kernel in Triton](/ai-systems/2026/08/29/triton-first-gpu-kernel/)*

*Next: LLM inference optimization dashboard — wrapping everything into one deployable tool.*

*Benchmark repo: [github.com/anirudhs16/llm-inference-benchmarks](https://github.com/anirudhs16/llm-inference-benchmarks)*
