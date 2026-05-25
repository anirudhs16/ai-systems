---
layout: post
title: "vLLM vs Ollama: I ran the same benchmark on both. 99× difference."
date: 2026-05-24
categories: [inference]
tags: [vllm, ollama, benchmarks, continuous-batching, pagedattention, concurrency]
---

Last week I measured how Ollama handles concurrent requests. Per-request latency degraded 1.8× going from 1 to 4 simultaneous users. User 4 waited 24 seconds for a response that takes 9 seconds in isolation.

This week I ran the exact same benchmark on vLLM. Same prompt, same concurrency levels, different serving engine.

The numbers were not close.

## Setup

| | Ollama | vLLM |
|--|--------|------|
| Hardware | Local laptop, no GPU | Google Colab T4 (15GB VRAM) |
| Model | llama3.2 | TinyLlama 1.1B |
| Version | Latest | vLLM 0.21.0 |
| Method | `requests` + `threading` | Native batch list |

Note on models: Ollama ran llama3.2 locally, vLLM ran TinyLlama 1.1B on Colab. The models are different sizes and hardware is different, so the absolute numbers aren't directly comparable. What matters is the *shape* of the latency curve as concurrency increases — that's the architectural difference this benchmark is isolating.

Full notebooks: [github.com/anirudhs16/llm-inference-benchmarks](https://github.com/anirudhs16/llm-inference-benchmarks)

## Results

| Concurrency | Ollama per-req | vLLM per-req | Speedup |
|-------------|----------------|--------------|---------|
| 1 user      | 8.92s          | 0.09s        | 99×     |
| 2 users     | 11.58s         | 0.69s        | 17×     |
| 4 users     | 16.33s         | 0.56s        | 29×     |

Two things stand out immediately.

First: the absolute speedup. Even accounting for hardware differences, this is a different category of performance.

Second, and more interesting: vLLM's per-request latency at 4 concurrent users (0.56s) is *lower* than at 2 users (0.69s). Latency went down as concurrency increased.

That's not a mistake. That's the architecture working exactly as designed.

## Why vLLM gets faster as concurrency increases

Ollama serves requests sequentially. Request 1 gets the GPU. Request 2 waits. The queue grows with concurrency, so latency grows linearly.

vLLM uses continuous batching. It doesn't wait for one request to finish before starting the next. Instead it interleaves token generation across all active requests simultaneously. While request 1 is waiting for its next token to be computed, requests 2, 3, and 4 are generating theirs in the same forward pass.

The GPU has fixed overhead per forward pass — loading the model weights, setting up the computation. With 1 request, that overhead is paid once per token generated. With 4 concurrent requests, that same overhead is shared across 4 tokens generated in the same pass. The per-token cost drops. Per-request latency drops with it.

This is the parallelism benefit that makes GPU inference worth the cost in the first place. A single-user workload barely utilizes the GPU's parallel compute capacity. A well-batched multi-user workload uses it properly.

## Where PagedAttention fits in

Continuous batching handles the compute side. PagedAttention handles the memory side.

Each active request needs KV cache memory for its conversation history. With Ollama's approach, memory is reserved per-request upfront — a large contiguous block sized for the maximum sequence length. As concurrent users grow, GPU memory fills with reserved-but-empty allocations.

PagedAttention allocates KV cache in small fixed-size blocks (16 tokens each) on demand. Memory is only consumed proportional to tokens actually generated, not the worst-case maximum. All 4 users' KV caches live in GPU memory simultaneously, each growing incrementally as the conversation progresses.

Together: continuous batching keeps the compute pipeline full. PagedAttention keeps the memory pool efficiently utilized. The two optimizations compound.

## What Colab setup actually took

Getting vLLM running on Colab T4 required two non-obvious fixes:

The T4 has compute capability 7.5, below vLLM's V1 engine threshold. vLLM falls back to the V0 engine automatically, but the default multiprocessing method causes a `fileno` error on Colab. Fix: set `VLLM_WORKER_MULTIPROC_METHOD=spawn` before import.

The second issue: using Python `threading` to fire concurrent requests to vLLM's synchronous `LLM.generate()` causes a deadlock — the engine's internal thread pool conflicts with external threads. Fix: pass all prompts as a list directly to `llm.generate()` and let vLLM handle the batching natively. This is actually the correct approach anyway — you're telling vLLM "here are N prompts, batch them" rather than firing N separate calls.

Both errors were in the setup, not the model. The benchmark itself ran cleanly once those were resolved.

## The hardware parallel

The GPU's parallel compute units sit idle when requests are served sequentially. This is the same waste as having 1,000 workers on an assembly line but only feeding one part at a time. The workers aren't slow — the feeding mechanism is.

Continuous batching is the mechanism that feeds the parallel compute units properly. PagedAttention is the memory management layer that makes it possible to hold multiple requests' state simultaneously without running out of room.

From a semiconductor perspective: this is the same problem as DRAM bank interleaving. You have parallel hardware capable of serving multiple transactions simultaneously. The question is whether the controller above it is smart enough to issue transactions that actually exploit the parallelism. vLLM is. Ollama — as a local dev tool — doesn't need to be.

## What comes next

Benchmark 03 will sweep batch sizes explicitly — 1, 4, 8, 16, 32 — and measure both latency and throughput (tokens/second) at each point. The goal is to find where throughput plateaus and latency starts degrading unacceptably. That inflection point is the optimal operating range for a given model and hardware configuration.

---

*Previous: [I measured what happens when 4 users hit a local LLM at the same time](/ai-systems/2026/05/22/concurrent-latency-benchmark/)*

*Next: Batch size sweep — finding the throughput/latency tradeoff curve.*

*Full benchmark notebooks: [github.com/anirudhs16/llm-inference-benchmarks](https://github.com/anirudhs16/llm-inference-benchmarks)*
