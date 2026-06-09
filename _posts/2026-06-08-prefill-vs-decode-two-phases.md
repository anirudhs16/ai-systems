---
layout: post
title: "LLM inference has two phases. They need completely different optimizations."
date: 2026-06-08
categories: [inference]
tags: [prefill, decode, ttft, throughput, latency, benchmarks, ollama]
---

I ran a simple experiment this week. I asked the same model the same topic three times — once asking for one sentence, once for a paragraph, once for a full technical explanation. I measured how long each took.

The TTFT numbers surprised me.

## The experiment

Model: llama3.2 via Ollama local. Three prompts designed to produce short, medium, and long responses.

| max_tokens | TTFT  | Total time | Decode time | Tokens generated |
|------------|-------|------------|-------------|-----------------|
| 128        | 2.93s | 6.36s      | 3.43s       | ~50             |
| 512        | 3.33s | 55.75s     | 52.42s      | ~400            |
| 2048       | 4.35s | 87.79s     | 83.44s      | ~600            |

Total time grew 13× from short to long response. TTFT grew only 1.5×.

That gap is the insight.

Full notebook: [github.com/anirudhs16/llm-inference-benchmarks](https://github.com/anirudhs16/llm-inference-benchmarks)

## What is actually happening inside the model

LLM inference is not one thing. It is two completely separate phases happening back to back.

**Phase 1 — Prefill**

When you send a message, the model reads your entire prompt at once. It processes all your tokens in parallel, builds the initial KV cache for your conversation, and prepares to generate. This happens once per request. It is fast because parallel computation on a GPU is what GPUs are designed for.

TTFT measures when this phase ends — the moment the first output token appears.

**Phase 2 — Decode**

Now the model generates your response. One token at a time. Each token requires a complete forward pass through all the model's layers. The output of one token becomes the input for the next. This cannot be parallelized — each step depends on the previous one.

This is why total time grows nearly linearly with response length. More tokens = more forward passes = more time. The decode phase is sequential by design.

## Why TTFT stayed almost flat

My TTFT went from 2.93s to 4.35s even as total time went from 6s to 88s.

That's because TTFT is driven by prompt length — how much the model has to read before generating — not by how much it will eventually write. All three experiments used similar-length prompts. The prefill cost was similar. The decode cost was completely different.

This separation is fundamental. Prefill is a parallel read operation. Decode is a sequential write operation. Different compute patterns, different bottlenecks, different optimization strategies.

## The production implication

This is where it gets practical. If you are building a system that uses LLMs, the right metric to optimize depends entirely on your use case.

**Optimize for TTFT when users are waiting.** A chat interface, a copilot, anything interactive. Users tolerate a slow full response if something appears quickly. Streaming handles the rest — tokens appear as they generate, the experience feels responsive even if the full answer takes 30 seconds.

**Optimize for throughput when nobody is watching.** Document summarization, batch analysis, overnight processing jobs. Nobody cares when the first token appears. They care how many documents get processed per hour. Pack requests tightly, maximize tokens per second, TTFT is irrelevant.

Same model. Same hardware. Completely different serving configuration. This is why production inference is not just "run the model faster" — it is understanding which phase is your bottleneck and for which workload.

## Connecting to previous benchmarks

Every benchmark I have run comes back to this prefill/decode distinction.

The vLLM vs Ollama 99× speedup — vLLM runs prefill on GPU, Ollama on CPU. Prefill is parallel-friendly so GPU wins massively.

The concurrent request degradation — Ollama's sequential serving blocks decode phases from overlapping. vLLM's continuous batching interleaves decode phases across all active requests.

The quantization result — INT4 saves memory during decode (smaller weights to load per forward pass) but adds dequantization overhead. On a small model the overhead dominated. On a large model the memory savings dominate.

Everything connects back to these two phases and which one is your constraint.

## The number to remember

Decode time grew 24× while TTFT grew 1.5× for a 12× increase in output length.

If you are profiling a slow LLM response, the first question is: slow to start, or slow to finish? The answer determines everything about where to look.

---

*Previous: [I ran FP16, INT8, and INT4 on the same model. The results surprised me.](/ai-systems/2026/06/05/quantization-fp16-int8-int4/)*

*Next: Portfolio CLI — wrapping all benchmarks into one tool with P50/P95 TTFT, tokens/sec, memory, and cost per 1K tokens.*

*Full benchmark notebooks: [github.com/anirudhs16/llm-inference-benchmarks](https://github.com/anirudhs16/llm-inference-benchmarks)*
