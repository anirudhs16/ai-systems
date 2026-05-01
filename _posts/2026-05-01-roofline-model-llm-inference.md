---
layout: post
title: "The roofline model: why your GPU is probably not doing what you think"
date: 2026-05-01
categories: [inference]
tags: [gpu, roofline-model, memory-bandwidth, arithmetic-intensity, llm-inference]
---

In chip design, before you tape out a block, you ask one question: is this workload compute-bound or memory-bound? The answer determines everything — which bottleneck to attack, where to spend transistors, what optimization even means.

There's a tool for answering that question. It's called the roofline model. I used it in silicon. Turns out it applies perfectly to GPU inference — and the answer has direct consequences for how you build and scale AI systems.

## What the roofline model is

The roofline model plots a workload on two axes. On one side: how much compute your hardware can do (FLOPS). On the other: how fast it can move data (memory bandwidth, bytes/sec).

Every workload has a property called **arithmetic intensity** — the ratio of compute operations to memory bytes accessed:

```
Arithmetic Intensity = FLOP / Bytes
```

A workload that does a lot of math on a small amount of data has high arithmetic intensity. One that reads a lot of data but does little math has low arithmetic intensity.

Your hardware has a fixed ratio of compute to bandwidth — called the **ridge point**. It's where the two rooflines meet.

- Workload arithmetic intensity **above** the ridge point → **compute-bound**. Adding memory bandwidth won't help. You're limited by raw FLOPS.
- Workload arithmetic intensity **below** the ridge point → **memory-bandwidth-bound**. Adding more CUDA cores won't help. You're limited by how fast data moves.

## Where LLM inference lands

An A100 80GB has roughly 312 TFLOPS (BF16) and 2 TB/s memory bandwidth. Its ridge point sits around 156 FLOP/byte.

Now look at what happens during LLM inference — specifically the **token generation phase** (autoregressive decoding):

For each new token, the model runs one forward pass through every layer. At each attention layer and every linear projection it loads the weight matrix from HBM, uses it once for a matrix-vector multiply, then moves on.

```
Example: a single linear layer in Llama 3 8B
  Weight matrix: 4096 × 4096 × 2 bytes (FP16) = 32 MB
  Operations:    2 × 4096 × 4096 = 33.5 MFLOP
  
  Arithmetic intensity = 33.5M / 32M ≈ 1 FLOP/byte
```

**1 FLOP/byte**. The A100 ridge point is 156 FLOP/byte. LLM token generation sits roughly **100× below** where it needs to be to use the GPU's compute capacity.

This is not a flaw in the model. It is a structural property of autoregressive decoding — you are generating one token at a time, using every weight once per token, moving enormous amounts of data for very little compute.

## What this means in practice

**Adding more CUDA cores does nothing for single-stream inference.** An H100 has more FLOPS than an A100. For memory-bandwidth-bound workloads, those extra FLOPS are idle. What matters is the H100's 3.35 TB/s HBM3 bandwidth versus the A100's 2 TB/s — a genuine 1.6× improvement where it counts.

**Batching is the lever, not clock speed.** When you batch multiple requests together, the weight matrices get reused across the batch. A batch size of 32 means 32 tokens computed per weight load — arithmetic intensity goes from ~1 FLOP/byte toward the ridge point. This is why throughput-optimized serving looks different from latency-optimized serving.

**Quantization helps for real reasons.** INT8 weights are half the size of FP16. Loading them requires half the memory bandwidth. For a memory-bandwidth-bound workload, that is a near-linear speedup on the bottleneck — not an approximation trick, a direct attack on the actual constraint.

**KV cache pressure compounds the problem.** As sequences grow longer, KV cache reads add more memory traffic per token without adding proportional compute. Arithmetic intensity falls further. This is why long-context inference is disproportionately expensive.

## The hardware parallel

In silicon, when you identify a memory-bandwidth-bound block, you don't throw more compute at it. You add read ports, widen the bus, restructure the memory hierarchy, or redesign the access pattern to improve locality. The same taxonomy applies here — just at a different layer of the stack.

FlashAttention improves arithmetic intensity by keeping data in SRAM instead of round-tripping to HBM. PagedAttention improves effective bandwidth utilization by reducing fragmentation. Speculative decoding restructures the compute pattern entirely to use the GPU more like a compute-bound workload during the draft phase.

Every major inference optimization of the last three years, viewed through the roofline lens, is an attempt to either raise arithmetic intensity or reduce memory traffic. The model tells you why those are the only two levers worth pulling.

## The number to remember

Most LLM inference during token generation runs at **~1 FLOP/byte arithmetic intensity**. Most GPUs hit their memory bandwidth roof around **50–150 FLOP/byte**. The gap is 2 orders of magnitude.

You are not compute-limited. You are bandwidth-limited. Every infrastructure decision that ignores this is optimizing the wrong thing.

---

*Previous: [KV cache: why a long conversation can exhaust a 16 GB GPU](/ai-systems/2026/04/18/kv-cache-memory-arithmetic/)*

*Next: vLLM and PagedAttention — running the first real benchmarks.*

*Benchmark suite: [github.com/anirudhs16/llm-inference-benchmarks](https://github.com/anirudhs16)*
