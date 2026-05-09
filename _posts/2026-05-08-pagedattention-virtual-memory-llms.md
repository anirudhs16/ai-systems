---
layout: post
title: "PagedAttention: the OS trick that made LLM serving practical"
date: 2026-05-08
categories: [inference]
tags: [vllm, pagedattention, kv-cache, memory-management, throughput]
---

Last week I got vLLM running on a T4 GPU in Colab. First real number: 0.30s TTFT for a single request on facebook/opt-125m. The engine reserved 6.55 GiB just for KV cache before serving a single token.

That memory reservation is the problem PagedAttention was built to solve.

## The problem: KV cache memory is wasteful by default

Recall from two weeks ago — the KV cache stores a representation of every token in your conversation on the GPU, growing with every message. For a serving system handling many users simultaneously, this creates a serious memory management problem.

The naive approach allocates a contiguous block of GPU memory per conversation — large enough for the maximum possible sequence length — at the moment the request arrives. 

Three things go wrong immediately:

**Internal fragmentation.** You reserve memory for 4,096 tokens because that's your max sequence length. The actual conversation uses 312 tokens. The remaining 3,784 token slots sit empty, allocated but unused, for the entire duration of the request.

**External fragmentation.** As requests arrive and complete at different times, the GPU memory fills with gaps — blocks that were freed but are too small or wrongly shaped to be reused by new requests. Like a parking lot where every car is a different size and nobody parks efficiently.

**No sharing.** If 50 users all start with the same system prompt, each gets their own separate KV cache for those identical tokens. The same computation stored 50 times.

The result: a GPU with 40 GB of HBM might only be able to serve 10-15 concurrent users before running out of KV cache memory — not because the model is too large, but because memory is being managed poorly.

## The solution: pages, blocks, and indirection

PagedAttention ([Kwon et al., 2023](https://arxiv.org/abs/2309.05852)) borrows the core insight from OS virtual memory.

In an operating system, virtual memory solves the "not enough RAM" problem by breaking physical memory into fixed-size pages and mapping process address spaces onto them non-contiguously. A process thinks it has a large contiguous block of memory. The OS maintains a page table that translates virtual addresses to physical ones. The physical pages can be anywhere in RAM — scattered, reused, swapped to disk.

PagedAttention applies this exactly to KV cache:

- GPU memory is divided into fixed-size **blocks** (typically 16 tokens per block)
- Each sequence gets a **block table** — a mapping from logical position to physical block
- Blocks are allocated on demand as the sequence grows, not upfront
- When a request completes, its blocks are freed and immediately available for new requests

```
Without PagedAttention:
  Request arrives → reserve 4096-token contiguous block → use 312 tokens → 3784 wasted

With PagedAttention:
  Request arrives → allocate one 16-token block → fill it → allocate next block → repeat
  Only the memory actually used is ever allocated
```

The attention computation itself is modified to work with non-contiguous physical blocks via the block table lookup. From the model's perspective, the sequence is still contiguous. From the memory manager's perspective, blocks can be placed anywhere.

## What this enables

**Near-zero fragmentation.** Blocks are fixed size, allocated on demand, freed immediately on completion. The free memory pool stays clean.

**Copy-on-write prefix sharing.** If multiple requests share a common prefix — a system prompt, a document context, a RAG retrieval — their block tables can point to the same physical blocks for the shared portion. No duplication. The vLLM paper reports up to 55% reduction in memory usage for workloads with shared prefixes.

**Preemption and swapping.** Because KV cache is now managed in discrete blocks, vLLM can preempt a low-priority request mid-generation by swapping its blocks to CPU RAM, serve a higher-priority request with that GPU memory, then swap back. Without paged memory this is impossible.

## The numbers

From the vLLM paper benchmarked on A100:

| System | Throughput (tokens/sec) |
|--------|------------------------|
| FasterTransformer | ~1,800 |
| Orca | ~2,100 |
| vLLM (PagedAttention) | ~5,600 |

Roughly 3× improvement in serving throughput on the same hardware — not from a better model, not from a faster GPU, from better memory management.

In my own Colab run this week: vLLM reserved 6.55 GiB for KV cache on the 14.56 GiB T4. With naive allocation that memory would support far fewer concurrent requests before fragmentation made it unusable. With paged blocks it stays efficiently utilized across the request lifecycle.

## The hardware parallel

I spent years working on physical memory — SRAM arrays, address decoders, bitline multiplexing. The entire point of that stack is to make physical storage locations addressable through an abstraction layer. A memory compiler generates the logic that maps a logical address to a physical bitcell location. The caller never thinks about where the bits actually live.

PagedAttention is the same abstraction, two layers up. The attention mechanism is the caller. The block table is the address decoder. The physical GPU memory blocks are the bitcells. The insight — that you can decouple the logical view of memory from its physical layout — is identical at every layer of the stack.

The reason this insight keeps appearing at different layers is that it keeps solving the same problem: physical resources are finite and irregularly shaped, but callers want to think in clean logical terms. Virtual memory. Paged KV cache. Next year it will appear somewhere else under a different name.

## What to watch

**Prefix caching** extends PagedAttention to cache blocks across requests permanently — not just sharing within a batch but reusing KV cache from previous requests that started with the same prefix. Major throughput win for RAG deployments where every request starts with the same retrieved documents.

**Disaggregated prefill** separates the prefill phase (processing the input prompt) from the decode phase (generating tokens) onto different hardware, with KV cache blocks transferred between them. PagedAttention's block abstraction is what makes this transfer practical.

Both of these are active areas in production inference engineering right now.

---

*Previous: [The roofline model: why your GPU is probably not doing what you think](/ai-systems/2026/05/02/roofline-model-llm-inference/)*

*Next: Quantization — running INT8 vs FP16 benchmarks on the same model.*

*Benchmark suite: [github.com/anirudhs16/llm-inference-benchmarks](https://github.com/anirudhs16)*
