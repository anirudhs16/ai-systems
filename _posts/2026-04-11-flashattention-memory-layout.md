---
layout: post
title: "FlashAttention: the memory layout insight that changed AI inference"
date: 2026-04-11
categories: [inference, hardware]
tags: [flashattention, gpu-memory, attention, sram, hbm, data-locality]
---

One of the most important AI inference breakthroughs of the last three years was not a better model. It was a memory layout insight — the same kind of insight I spent years applying in silicon.

## The hardware reality every LLM runs on

Your GPU has two kinds of memory. They are not equal.

**HBM (High Bandwidth Memory)** is the warehouse. A high-end GPU has 40–80 GB of it. It stores your model weights, your KV cache, your activations — everything. It is also slow to access relative to what's sitting right next to the compute cores.

**SRAM (on-chip memory / shared memory)** is the workbench right next to you. A high-end GPU has maybe 20–40 MB of it total across all streaming multiprocessors. It is orders of magnitude faster than HBM. And it is tiny.

This gap — huge slow warehouse, tiny fast workbench — is not an AI problem. It is a physics problem. It has existed since the first memory hierarchies were designed. I spent years working around it in silicon. The tools change. The constraint does not.

## What standard attention does to memory

The attention mechanism computes a relationship score between every token and every other token in your sequence. For a 4,000-token conversation that is 16 million scores.

Standard attention materialises all of those intermediate scores as a full matrix in HBM. Here is what that looks like in practice:

1. Read Q, K from HBM into SRAM → compute QK^T scores → write full score matrix back to HBM
2. Read score matrix from HBM → apply softmax → write softmax output back to HBM  
3. Read softmax output from HBM → multiply by V → write final output back to HBM

Every arrow in that list is a round trip to the warehouse. For a long sequence you are making millions of those trips. The math takes microseconds. The memory traffic takes milliseconds.

**The bottleneck is not the arithmetic. It is the trips.**

This is what hardware engineers call being *memory-bandwidth-bound*. Adding more compute cores does not help. You need less data movement.

## What FlashAttention does differently

FlashAttention ([Dao et al., 2022](https://arxiv.org/abs/2205.14135)) asks a simple question: what if we never wrote those intermediate matrices to HBM at all?

The answer is *tiling*. Instead of computing the full attention matrix at once, FlashAttention breaks both the input sequence and the computation into small blocks — tiles — that fit entirely in SRAM. It completes the full attention calculation block by block, accumulating the result on-chip, and only writes the final output back to HBM once.

Same mathematical result. A fraction of the memory traffic.

```
Standard attention:  O(N²) HBM reads/writes  (N = sequence length)
FlashAttention:      O(N) HBM reads/writes
```

For a 4K sequence, that is the difference between 16 million intermediate values moving through HBM and essentially none.

## Why this is not a new idea

This principle has a name in computer architecture: **data locality**.

Keep data close to where it is being used. Minimise the trips to slow memory. Exploit spatial and temporal locality in the memory access pattern.

It is one of the oldest ideas in hardware design. Cache hierarchies are built on it. Memory compilers optimise for it. Chip architects spend careers chasing it at 5nm.

What FlashAttention did was apply it to a computation that the ML community had assumed was fixed — because the naive formulation of attention seemed to require materialising the full matrix. The insight was that the *mathematical* requirement (compute the full attention) and the *memory* requirement (store the full intermediate matrix) are not the same thing.

You can satisfy the math without satisfying the naive memory pattern. Tiling lets you do it.

## The result

FlashAttention is now the default attention implementation in virtually every serious inference stack — vLLM, TensorRT-LLM, HuggingFace Transformers. It is not optional. It is the baseline.

It delivers:
- **2–4× speedup** on attention computation for long sequences
- **~10× reduction** in memory footprint for the attention computation
- **Longer context windows** that would be physically impossible with standard attention on the same hardware

And it was not achieved by better math, a bigger model, or more GPUs. It was achieved by respecting the memory hierarchy.

## What this tells us about AI infrastructure

The hardware constraints do not change. HBM is slow. SRAM is fast and small. Data movement costs energy and time. These facts are as true for a GPU in 2026 as they were for a memory array in 2016.

What changes is the layer of the stack where the problem lives. When I was laying out SRAM cells, the constraint was transistor capacitance and bitline length. When the FlashAttention authors sat down with an A100, the constraint was HBM bandwidth and on-chip SRAM capacity. Different tools, identical physics.

This is the thing I find most useful about having a hardware background when thinking about AI systems: the abstraction level changes, the bottleneck taxonomy does not. Every layer of the stack is still trading compute against memory movement. Every optimisation is still about keeping data close to where it is needed.

FlashAttention just made this legible at the model level.

---

*Next post: KV cache memory arithmetic — how a single long conversation can exhaust a 16 GB GPU.*

*Benchmark suite in progress: [github.com/anirudhs16/llm-inference-benchmarks](https://github.com/anirudhs16)*
