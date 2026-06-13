---
layout: post
title: "I profiled a single forward pass. 1460 GPU calls for 20 tokens."
date: 2026-06-12
categories: [inference]
tags: [pytorch, profiler, gpu, cuda, kernels, optimization]
---

Every benchmark so far has measured *that* something is slow — TTFT, tokens per second, memory usage. This week I looked inside a single forward pass to see *where* the time actually goes.

The answer surprised me, and it ties together everything from the last eight weeks.

## The setup

```python
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=20)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

Model: facebook/opt-125m, Colab T4, generating 20 tokens.

Full notebook: [github.com/anirudhs16/llm-inference-benchmarks](https://github.com/anirudhs16/llm-inference-benchmarks)

## The headline numbers

```
Self CPU time total: 2.350s
Self CUDA time total: 58.145ms
```

The GPU did 58 milliseconds of actual computation. The whole operation took 2.35 seconds. The gap, roughly 40x, is not computation. It is overhead.

## Where the overhead lives

The top row of the profiler output:

```
aten::linear   CPU total: 916ms   CUDA total: 43ms   # calls: 1460
```

1460 calls to the linear layer operation, for 20 generated tokens. Each call's actual GPU work averages 0.03ms. But the CPU side — Python packaging the instruction and sending it to the GPU driver — averages 0.6ms per call. The overhead per call is roughly 20x the computation per call.

Where does 1460 come from? Each transformer layer makes about 6 linear calls — Q, K, V projections, output projection, and two in the MLP block. With 12 layers:

```
6 calls/layer x 12 layers x 20 tokens ~= 1440 calls
```

That matches the measured 1460. Every one of those calls pays the same fixed overhead — a Python-to-GPU round trip — regardless of how small the actual computation is.

## What a kernel actually is

The profiler also shows the actual GPU programs running underneath PyTorch's operations:

```
gemv2T_kernel_val   CUDA total: 8.86ms   # calls: 20
```

This is a CUDA kernel — a small program that runs identically across thousands of GPU cores simultaneously, each core computing one piece of the output (one element of a matrix multiplication result, for instance). `aten::linear` and `aten::addmm` are PyTorch's names for "launch this kernel with these inputs." The kernel is the thing that actually runs on the hardware.

## Two fixes, two different axes

This 40x overhead gap is exactly the problem two production techniques solve, along different axes.

CUDA graphs attack the vertical axis — overhead within a single request. The sequence of operations for one forward pass is identical every time the model runs (same shapes, same order). CUDA graphs record that sequence once and replay it as a single instruction. Same 1460 GPU operations, but instead of 1460 Python-to-GPU round trips, there is one. This is what vLLM's CUDAGraph capture step does at startup — the 39 seconds observed in an earlier benchmark.

Continuous batching attacks the horizontal axis — overhead amortized across users. It does not reduce the 1440 calls per forward pass. Instead, each call processes a wider input containing multiple users' tokens stacked together. Same 1440 calls, now serving N users instead of one.

vLLM uses both. Combined, they account for a meaningful share of the 99x speedup measured in an earlier benchmark, separate from the CPU-vs-GPU and PagedAttention factors already discussed.

## Connecting everything back

This single profiling session ties together most of what the last eight weeks have been about.

The roofline model said inference is overhead-dominated, not compute-dominated. This profile shows it directly — 58ms of compute inside 2.35s of wall time.

The quantization benchmark showed INT4 was slower than FP16 on this same model. Now it is clear why. Quantization reduces the size of each operation's data, but does not reduce the number of operations. On a model where 1460 calls of fixed overhead dominate, shrinking each call's data barely moves the needle. The bottleneck was never the data size for this model — it was the call count.

The vLLM vs Ollama benchmark showed a 99x speedup. Some of that is GPU vs CPU. Some is PagedAttention. And some, visible now for the first time, is this exact overhead pattern being collapsed via CUDA graphs and batching.

## The number to remember

For a 125M parameter model generating 20 tokens: 1460 GPU calls, 58ms of actual compute, 2.35s of wall time. The ratio between compute and wall time, about 40x, is the overhead that every serving optimization in this field exists to reduce.

---

*Previous: [I built a CLI to benchmark any LLM. Here's what it outputs.](/ai-systems/2026/06/15/benchmark-cli/)*

*Next: Speculative decoding — running it for real, measuring how fewer forward passes translates to fewer of these 1460-call cycles.*

*Full benchmark notebooks: [github.com/anirudhs16/llm-inference-benchmarks](https://github.com/anirudhs16/llm-inference-benchmarks)*
