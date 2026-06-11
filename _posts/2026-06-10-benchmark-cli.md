---
layout: post
title: "I built a CLI to benchmark any LLM. Here's what it outputs."
date: 2026-06-10
categories: [inference]
tags: [cli, benchmarking, python, ttft, throughput, portfolio]
---

For the last few weeks I've been running one-off notebooks — copy a cell, change a number, run it, read the output. Useful for learning, but not reusable. This week I turned that into a tool.

## What it does

```bash
python benchmark.py --model llama3.2 --runs 5
```

Output:

```
==================================================
BENCHMARK RESULTS
==================================================
Model:           llama3.2
Backend:         Ollama (local)
Runs:            5
Prompt:          Explain what a KV cache is in two sentences....
--------------------------------------------------
P50 TTFT:        2.42s
P95 TTFT:        2.47s
Avg tokens/sec:  12.4
Avg total time:  8.62s
--------------------------------------------------
Cost/1K tokens:  $0.00 (local)
==================================================
```

Point it at any model running in Ollama, give it a number of runs, get back the metrics that matter for production serving decisions.

Full code: [github.com/anirudhs16/llm-inference-benchmarks](https://github.com/anirudhs16/llm-inference-benchmarks)

## Why P50 and P95, not just average

Five runs against the same model, same prompt:

```
Run 1: TTFT = 7.46s
Run 2: TTFT = 2.50s
Run 3: TTFT = 2.37s
Run 4: TTFT = 2.45s
Run 5: TTFT = 2.42s
```

The first run is nearly 3× slower than the rest. That's the model loading into memory for the first time — a cold start. Every subsequent request reuses the warm model and runs consistently around 2.4s.

An average across these five runs would be 3.4s — a number that doesn't represent either the typical case or the worst case well.

P50 (median) tells you the typical experience: 2.42s. P95 tells you what your slowest 5% of users experience: 7.46s, dominated by the cold start. In production, P95 is often the number that matters most — it's what shows up in your SLA, and it's what determines whether your worst-case latency is acceptable.

This is also why production inference servers keep models warm — paying the cold-start cost once at startup rather than on a user's first request.

## Why tokens/sec instead of just total time

Total time depends on how long the response is. A one-sentence answer and a full essay take very different amounts of time even on identical hardware — that's the prefill/decode distinction from last week's post.

Tokens per second normalizes for this. It's a hardware and configuration metric, not a per-request metric. 12.4 tokens/sec on this setup tells you something comparable across different prompts, different response lengths, even different models — assuming you're comparing apples to apples on hardware.

## Why cost per 1K tokens — even when it's $0

Right now this runs locally, so cost is $0. But the field exists for a reason: the moment you point this CLI at a hosted API (OpenAI, Anthropic, a self-hosted vLLM endpoint with a known GPU cost per hour), this becomes the number that connects technical performance to business decisions.

A model that's 20% faster but costs 3× more per token isn't automatically the right choice — it depends on the latency requirements of the use case. Having cost in the same output as latency and throughput keeps that tradeoff visible, not buried in a separate spreadsheet.

## What's under the hood

The CLI is built in four pieces:

`run_single()` — sends one streaming request, measures TTFT (time to first token) and total time, calculates tokens/sec from the decode phase.

`run_benchmark()` — runs N requests, collects results, computes P50/P95 from the TTFT distribution.

`print_results()` — formats everything into a readable summary.

`argparse` — `--model`, `--runs`, `--prompt`, `--backend` as command-line flags.

Currently supports Ollama as the backend. The structure is designed so adding a vLLM backend later is a matter of writing one new function — `run_single` already isolates all the request logic.

## What this represents

Four weeks ago I was running notebooks I half-understood, copying cells, watching numbers change without always knowing why. This week I wrote this from an empty file, one function at a time, testing each piece before moving to the next.

The CLI itself is simple — under 100 lines. But it's mine. Every line was written, run, and debugged by me. That's a different kind of artifact than a notebook full of someone else's code with my numbers plugged in.

## What's next

Add a vLLM backend so the same CLI can compare Ollama vs vLLM with one flag change. Then a `--sweep` mode that runs across multiple models or configurations and outputs a comparison table — turning this from a single-run tool into a proper comparison framework.

---

*Previous: [LLM inference has two phases. They need completely different optimizations.](/ai-systems/2026/06/08/prefill-vs-decode-two-phases/)*

*Next: PyTorch profiler — looking inside a single forward pass to see where time actually goes.*

*Full benchmark CLI and notebooks: [github.com/anirudhs16/llm-inference-benchmarks](https://github.com/anirudhs16/llm-inference-benchmarks)*
