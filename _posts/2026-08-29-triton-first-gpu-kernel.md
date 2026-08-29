---
layout: post
title: "I wrote my first GPU kernel in Triton. Here is what it taught me."
date: 2026-08-29
categories: [inference]
tags: [triton, gpu-kernels, cuda, parallel-computing, semiconductor]
---

Everything I have measured so far — TTFT, throughput, kernel call overhead — happened at the PyTorch layer. This week I went one layer deeper and wrote a GPU kernel directly.

Not in CUDA. In Triton — a Python-based language that compiles to GPU assembly. The right entry point for someone coming from application engineering who wants to understand what actually runs on the hardware.

## What a GPU kernel is

When PyTorch executes `x + y` on two tensors, it launches a CUDA kernel — a small program that runs simultaneously across thousands of GPU cores. Each core handles a different slice of the data. The addition of a million-element vector becomes a million additions happening in parallel.

The PyTorch profiler session showed 1460 kernel launches for 20 generated tokens. Each of those launches was a program like this one running on the GPU. Until now I had only seen them from the outside.

## The vector addition kernel

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
```

Each instance of this kernel handles one block of elements. `tl.program_id(axis=0)` tells this instance which block it is responsible for. `tl.arange(0, BLOCK_SIZE)` generates the indices for that block. `tl.load` pulls data from GPU memory. `tl.store` writes the result back.

The mask handles the case where the vector length is not evenly divisible by BLOCK_SIZE — the last block may be partially empty.

## What the output confirmed

```
Max difference: 0.0
Triton and PyTorch match: True
```

Same result as PyTorch's built-in addition. The kernel is correct.

## Why this connected to my background

The memory access pattern in this kernel — load a block, compute, store — is identical to what I designed at the transistor level in SRAM arrays. A memory read cycle: assert address, precharge bitlines, sense differential voltage, latch output. The abstraction level is completely different. The pattern is the same.

`tl.load` with a mask is the software equivalent of a conditional read with chip select. The block decomposition is the same tiling strategy I used to think about memory access patterns in silicon.

The hardware constraints do not change. The vocabulary changes at every layer of the stack.

## What Triton is and why it matters

CUDA requires C++ and deep knowledge of GPU architecture. Triton sits above CUDA — you write Python-like code and Triton compiles it to efficient GPU assembly automatically. It handles memory coalescing, shared memory management, and instruction scheduling.

This is the layer where FlashAttention was implemented. PagedAttention's memory management operates at this layer. The 99× speedup measured in benchmark 02 comes partly from kernels written at this level — making better use of GPU memory hierarchy than PyTorch's default implementations.

Understanding this layer does not mean writing production kernels. It means being able to read them, understand optimization decisions, and have an informed conversation about where performance comes from.

## What is next

Benchmark this kernel against PyTorch's built-in addition and measure the difference. Then move to a more complex kernel — matrix multiplication — where the memory access pattern optimization makes a measurable performance difference.

---

*Previous: [Designing an enterprise customer support AI system from scratch](/ai-systems/2026/08/15/enterprise-customer-support-ai-architecture/)*

*Next: Triton matrix multiplication — where memory access patterns actually matter for performance.*

*Full benchmark notebooks: [github.com/anirudhs16/llm-inference-benchmarks](https://github.com/anirudhs16/llm-inference-benchmarks)*
