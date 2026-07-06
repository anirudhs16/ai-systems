---
layout: post
title: "Speculative decoding: I ran it. The results were messier than expected."
date: 2026-07-05
categories: [inference]
tags: [speculative-decoding, benchmarks, forward-pass, latency, huggingface]
---

The theory is clean. A small fast model drafts tokens. A large model verifies them in one pass. Fewer forward passes, faster responses.

I ran the numbers. The results were more interesting than a clean 2x speedup.

## The setup

Two models from the same family:

- **Verifier (large):** facebook/opt-1.3b — does the actual generation
- **Drafter (small):** facebook/opt-125m — predicts what the large model would say

HuggingFace makes this a one-line change:

```python
# Without speculative decoding
model.generate(**inputs, max_new_tokens=50)

# With speculative decoding
model.generate(**inputs, max_new_tokens=50, assistant_model=draft_model)
```

That single parameter tells HuggingFace to run the draft-then-verify loop internally.

Full notebook: [github.com/anirudhs16/llm-inference-benchmarks](https://github.com/anirudhs16/llm-inference-benchmarks)

## The results

| Output length | Baseline | Speculative | Speedup |
|--------------|----------|-------------|---------|
| 20 tokens    | 2.26s    | 1.83s       | 1.24×   |
| 50 tokens    | 1.40s    | 1.52s       | 0.92×   |
| 100 tokens   | 1.75s    | 1.47s       | 1.19×   |
| 200 tokens   | 3.70s    | 2.77s       | 1.34×   |

At 50 tokens speculative decoding was actually slower. At 200 tokens it was 1.34x faster. The pattern is noisy — not the clean increasing curve the theory predicts.

## Why the numbers are messy

**Draft acceptance rate is the key variable.** Speculative decoding only helps when the small model's predictions match what the large model would have generated. If the large model rejects the drafts frequently, you pay the cost of drafting without getting the benefit of skipping forward passes.

opt-125m and opt-1.3b are different enough in quality that the acceptance rate is inconsistent. Sometimes 4 draft tokens get accepted — speedup. Sometimes 1 gets accepted — basically no benefit, with added overhead.

**The models are both small.** Speculative decoding shines when the verifier is genuinely large — 7B, 13B, 70B parameters — where each forward pass through the large model is expensive. Here the "large" model is only 1.3B. Each pass is already cheap, so the overhead of running a separate draft model doesn't pay off as decisively.

**Variance at small scales.** Total times are 1-4 seconds. CUDA warmup, memory allocation, and GPU scheduling variance can swing results by 0.3-0.5 seconds — enough to flip a marginal speedup into a marginal slowdown at 50 tokens.

## What the theory actually predicts

The mechanism is sound and the math works out clearly:

Without speculative decoding: **N tokens = N large-model forward passes**

With speculative decoding: **N tokens ≈ N/k large-model forward passes** (where k = average accepted drafts per verification pass)

If k=4 (the small model predicts correctly 4 out of every 5 tokens), a 200-token response needs ~50 large model passes instead of 200. Each large model forward pass is one complete cycle through all layers — for a 70B model on a slow GPU, this is the dominant cost. Cutting it by 4x is a real 4x speedup.

The reason this matters more at scale: with a 1.3B verifier, each forward pass takes ~50ms. With a 70B verifier, each forward pass might take 2-3 seconds. Saving 150 forward passes saves 5 minutes, not 7 seconds.

## The connection to the profiler

Last week I profiled a forward pass and found 1,460 kernel calls for 20 tokens — each call paying fixed Python-to-GPU round-trip overhead. Speculative decoding directly reduces how many of those 1,460-call cycles happen per response. Fewer large model forward passes = fewer complete kernel call sequences. The profiler data makes it concrete: every accepted batch of draft tokens is an entire 1,460-call cycle that doesn't happen.

## When to use it in production

Speculative decoding is worth the engineering complexity when:

- The verifier model is large (7B+) and each forward pass is expensive
- The drafter and verifier are from the same model family — similar training means higher draft acceptance rate
- Output sequences are long — the benefit compounds with length
- Latency is the constraint, not throughput — speculative decoding reduces latency for individual requests but adds complexity for high-concurrency batch serving

It is less useful when models are small, acceptance rates are unpredictable, or you're optimizing for throughput across many concurrent users (where continuous batching already handles the efficiency).

## What I'd test next

Run the same experiment with llama3.2 (3B) as the verifier and a matched smaller model as drafter, on longer sequences (500+ tokens). The acceptance rate should be higher with better-matched models, and the verifier is expensive enough that saved forward passes translate to meaningful time.

---

*Previous: [I profiled a single forward pass. 1460 GPU calls for 20 tokens.](/ai-systems/2026/06/19/pytorch-profiler-kernel-calls/)*

*Next: Architecture track begins — vendor comparison matrix and the first enterprise case study.*

*Full benchmark notebooks: [github.com/anirudhs16/llm-inference-benchmarks](https://github.com/anirudhs16/llm-inference-benchmarks)*
