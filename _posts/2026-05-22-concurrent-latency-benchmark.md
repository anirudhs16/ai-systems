---
layout: post
title: "I measured what happens when 4 users hit a local LLM at the same time"
date: 2026-05-22
categories: [inference]
tags: [benchmarks, ollama, concurrency, latency, throughput, pagedattention]
---

Theory is one thing. Running it and watching the numbers is another.

This week I set up a real benchmark — sent 1, 2, and 4 simultaneous requests to a locally running LLM and measured what happened to per-request latency. The results made the concepts from the last few weeks click in a way reading didn't.

## Setup

- Model: llama3.2 via Ollama
- Hardware: local laptop, no GPU
- Prompt: "Explain what a KV cache is in two sentences."
- Method: Python `threading` — all N requests fired simultaneously, each timed independently

Full notebook: [github.com/anirudhs16/llm-inference-benchmarks](https://github.com/anirudhs16/llm-inference-benchmarks)

## Results

| Concurrency | Wall time | Avg per-request | Max per-request |
|-------------|-----------|-----------------|-----------------|
| 1 user      | 8.92s     | 8.92s           | 8.92s           |
| 2 users     | 14.46s    | 11.58s          | 14.46s          |
| 4 users     | 24.64s    | 16.33s          | 24.64s          |

Key observation: per-request latency at 4 concurrent vs 1 is **1.8× slower**.

## What these numbers are telling us

At 1 user, the model takes 8.92 seconds to respond. That's the baseline — full access to memory and compute, no competition.

At 4 concurrent users, the last user waited 24.64 seconds for a response that takes 8.92 seconds in isolation. That's not a slow model. That's a queuing problem.

Ollama is a local development tool. It serves requests sequentially — it finishes user 1 before starting user 2. When 4 users fire simultaneously, users 2, 3, and 4 wait in line. The wall time grows, and the last user's experience degrades significantly.

**Why not exactly 4×?** Because Ollama overlaps some processing between requests and the requests don't all take identical time. The averages blend together. But the trend is clear — latency degrades linearly with sequential queuing.

## Why this happens — connecting to the roofline model

Two weeks ago I wrote about how LLM inference is memory-bandwidth-bound. The GPU spends most of its time loading model weights from HBM memory, not doing computation.

At 1 concurrent user the memory bus is the bottleneck but it's not even fully utilized — there are gaps between requests where the hardware sits idle. At 4 concurrent users with sequential serving, those gaps get filled with waiting users instead of with useful work.

A well-designed inference server would interleave all 4 users' token generation — while user 1 is waiting for its next token to be computed, users 2, 3, and 4 can be generating theirs. The memory bus stays busy, latency per user stays lower, and total throughput increases. This is called continuous batching.

## Where PagedAttention fits in

The queuing problem has two components:

**Compute queuing** — one request has to finish before the next starts. Continuous batching fixes this.

**Memory queuing** — each request needs KV cache memory reserved upfront. If memory fills up, new requests wait even before compute begins.

With 4 concurrent users at a 2,000-token context each, you need roughly 4 × 256MB = ~1GB of KV cache simultaneously. On a machine with limited RAM, Ollama's approach of reserving contiguous blocks per request makes this fragmentation problem worse as concurrency grows.

PagedAttention fixes the memory side — allocating KV cache in small 16-token pages on demand instead of large contiguous blocks upfront. More requests fit in the same memory, less waiting, lower latency per user even before you address the compute queuing.

The two fixes together — continuous batching and paged memory — are what make vLLM significantly faster than Ollama under concurrent load.

## Next benchmark

Run the same experiment on vLLM and compare the numbers directly. The hypothesis: at 4 concurrent users, vLLM's per-request average should be significantly closer to the 1-user baseline than Ollama's 1.8× degradation.

If the theory holds, the numbers will show it. If they don't, that's equally interesting.

---

*Previous: [Why enterprise AI projects fail — and it is never the model](/ai-systems/2026/05/15/enterprise-ai-failures/)*

*Next: vLLM vs Ollama — the same benchmark, a different serving engine.*

*Full benchmark notebook: [github.com/anirudhs16/llm-inference-benchmarks](https://github.com/anirudhs16/llm-inference-benchmarks)*
