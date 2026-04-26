---
layout: post
title: "KV cache: why a long conversation can exhaust a 16 GB GPU"
date: 2026-04-18
categories: [inference, memory]
tags: [kv-cache, llama3, gpu-memory, inference-engineering, context-length]
---

Last week I wrote about FlashAttention — how a memory layout insight cut attention's HBM traffic from O(N²) to O(N). This week I want to show you what happens *between* requests. The memory that does not go away when a user sends their next message.

That is the KV cache. And once you see the numbers, the entire shape of the inference engineering problem becomes clear.

## What the KV cache is

When an LLM generates your next token, it does not reprocess your entire conversation from scratch. It would be too slow. Instead, it stores a compressed representation of everything said so far — the Key and Value matrices from the attention mechanism — and reuses them for every new token it generates.

This is the KV cache. It lives on the GPU. It persists for the duration of your conversation. And it grows with every token you send or receive.

The name is literal: it is a cache of the K and V matrices that the model would otherwise recompute.

## The memory formula

```
KV cache size = 2 × num_layers × num_kv_heads × head_dim × seq_len × bytes_per_value
```

This is not an approximation. This is the exact memory footprint. Let me break it down for a real model.

**Llama 3 8B:**

| Parameter | Value | Why |
|-----------|-------|-----|
| `2` | 2 | One K matrix, one V matrix per layer |
| `num_layers` | 32 | Llama 3 8B has 32 transformer layers |
| `num_kv_heads` | 8 | Uses Grouped Query Attention — 8 KV heads, not 32 |
| `head_dim` | 128 | Each head is 128-dimensional |
| `bytes_per_value` | 2 | FP16 precision |

Plug in sequence length and the numbers are stark:

| Conversation length | KV cache size |
|--------------------|---------------|
| 2,000 tokens | ~256 MB |
| 8,000 tokens | ~1 GB |
| 32,000 tokens | ~4 GB |
| 128,000 tokens | ~16 GB |

The model weights for Llama 3 8B in FP16 consume **16 GB**. A single 128K-token conversation's KV cache matches the entire model weight footprint. On a 16 GB GPU you cannot serve that conversation at all.

## The concurrency problem

The numbers above are for a single user. Now add concurrency.

Fifty users chatting simultaneously, each mid-conversation at 4,000 tokens:

```
50 users × ~512 MB per user = ~25 GB KV cache
+ 16 GB model weights
= ~41 GB total GPU memory needed
```

An A100 80GB handles this. An RTX 4090 (24 GB consumer GPU) does not. The constraint is not the model. It is the cache.

This is why:

- **Context length is not just a model capability** — it is a memory provisioning decision
- **Concurrent users are a memory management problem** — each new user consumes GPU memory proportional to their conversation length
- **Most inference engineering right now** is about making this cache smaller, faster, and smarter

## What engineers do about it

Three main approaches, each trading something different:

**Quantization of the KV cache.** Store K and V at INT8 instead of FP16. Cuts cache size in half with minimal quality impact for most use cases. This is now standard in production deployments.

**PagedAttention (vLLM).** Instead of allocating a contiguous block of GPU memory per conversation upfront, PagedAttention allocates memory in small fixed-size pages — like virtual memory in an operating system. This eliminates fragmentation and allows the GPU to serve significantly more concurrent users from the same memory pool. The vLLM paper reports 2–4× throughput improvement on concurrent workloads.

**Sliding window / sparse attention.** Only cache the last N tokens instead of the full conversation. Llama 3's 128K context window uses a combination of techniques to make this tractable. You lose access to early conversation history but stay within memory bounds.

## Why my background makes this feel familiar

Early in my career I spent years thinking about how data moves through memory at the transistor level — designing SRAM arrays, optimising bitline capacitance, working out how many bits you can store in a given silicon area at a given process node.

The KV cache problem is the same problem at a different layer of abstraction.

You have a fixed physical resource (GPU HBM). You have a workload (concurrent conversations) that makes variable, growing demands on that resource. You have to fit as much as possible while maintaining performance guarantees. The tools are different — you are allocating CUDA memory blocks instead of placing transistors — but the constraint structure is identical.

PagedAttention is virtual memory. KV cache quantization is a precision-area tradeoff. Sliding window attention is a cache eviction policy.

The vocabulary changes at every layer of the stack. The engineering problem does not.

## What to watch

The KV cache is where a significant fraction of inference research is focused right now:

- **MLA (Multi-head Latent Attention)** in DeepSeek-V2 compresses the KV cache by projecting K and V into a lower-dimensional latent space before caching — substantially smaller cache, similar quality
- **Prefix caching** reuses KV cache across requests that share a common prefix (system prompt, document context) — major throughput wins for RAG deployments
- **KV cache offloading** moves cold cache pages to CPU RAM or NVMe, trading latency for capacity

If you are building or evaluating an inference serving system, the KV cache budget is the first constraint to size. Everything else — batch size, concurrency, context window — follows from it.

---

*Next post: GPU architecture and running my first real inference benchmarks — TTFT and throughput numbers from a T4.*

*Benchmark suite: [github.com/anirudhs16/llm-inference-benchmarks](https://github.com/anirudhs16)*
