---
layout: post
title: "I ran FP16, INT8, and INT4 on the same model. The results surprised me."
date: 2026-06-05
categories: [inference]
tags: [quantization, fp16, int8, int4, benchmarks, memory, bitsandbytes]
---

Quantization is one of the most talked-about inference optimizations. The pitch is simple: store model weights at lower precision, use less memory, go faster. I ran the numbers to see if that holds up.

The results were not what I expected.

## Setup

- Model: `facebook/opt-125m` (125M parameters)
- Hardware: Google Colab T4 GPU (15GB VRAM)
- Library: HuggingFace Transformers + bitsandbytes
- Prompt: "Explain what a KV cache is in two sentences."
- Measured: peak GPU memory and inference time at FP16, INT8, INT4

Full notebook: [github.com/anirudhs16/llm-inference-benchmarks](https://github.com/anirudhs16/llm-inference-benchmarks)

## Results

| Precision | Inference time | Peak GPU memory | vs FP16 memory | vs FP16 speed |
|-----------|---------------|-----------------|----------------|---------------|
| FP16      | 0.78s         | 258 MB          | 1.00×          | 1.00×         |
| INT8      | 2.72s         | 171 MB          | 0.66×          | 3.51× slower  |
| INT4 (NF4)| 3.47s         | 138 MB          | 0.53×          | 4.48× slower  |

INT4 used 47% less GPU memory than FP16. It was also 4.5× slower.

## What happened

The memory savings are real. INT4 stores each weight in 4 bits instead of 16. Fewer bits to load from HBM means less memory bandwidth consumed per forward pass — exactly what the roofline model predicts should help.

But the inference time went the wrong direction. INT8 was 3.5× slower than FP16. INT4 was 4.5× slower. Both are supposed to be faster. What went wrong?

The model is too small.

`opt-125m` has 125 million parameters. At FP16, its weights consume 250MB — a fraction of the T4's 15GB VRAM. The model fits comfortably, the memory bus is not stressed, and FP16 tensor operations run natively on the T4's hardware. There is no meaningful bottleneck to attack.

Quantization introduces overhead: the conversion layer between integer weights and floating-point activations during inference. On a small model where memory bandwidth is not the real bottleneck, this overhead costs more time than the memory savings gain. You're paying the dequantization tax without collecting the bandwidth dividend.

## When quantization actually helps

The roofline model makes this precise. LLM inference is memory-bandwidth-bound when the model is large enough that weight loading dominates the forward pass time. For a 125M model on a T4, that condition isn't met. For a 7B model on the same T4 — where FP16 weights alone require 14GB, nearly the entire VRAM — it absolutely is.

At 7B scale:
- FP16 requires ~14GB VRAM. Barely fits on a T4, no room for KV cache or batch size
- INT8 drops to ~7GB. Suddenly you have headroom for concurrent users
- INT4 drops to ~3.5GB. You can run the model on hardware that couldn't load it at all

The memory reduction at large scale isn't just about speed — it determines whether you can serve the model at all. This is why quantization is standard practice in production: it's often the difference between deployable and not deployable on a given hardware budget.

## The quality question

Output comparison across precisions:

- FP16: "It's a cache that's used to store data." *(brief but correct)*
- INT8: "I'm not sure what you mean by explain what a KV cache" *(degraded)*
- INT4: "I'm not sure what a KV cache is." *(worse)*

Quality degraded noticeably at INT8 and INT4 for this model. This is expected — opt-125m is a small, older model with limited robustness to quantization. Larger, more recent models (Llama 3, Mistral) are specifically trained with quantization in mind and maintain quality well down to INT4 and even lower.

The quality tradeoff is model-dependent and needs to be evaluated per use case. For production deployments, quantized models are typically validated against a held-out benchmark before shipping.

## The hardware parallel

Weight quantization is the same engineering tradeoff as reduced-precision arithmetic in silicon. When I worked on memory layout, we had the same choice: store values at full precision and use more area and power, or store at reduced precision and add a conversion step. The conversion step costs cycles. Whether that cost is worth paying depends entirely on how constrained the storage is.

On a chip with abundant SRAM, reduced precision often isn't worth the conversion overhead. On a chip with tight memory budgets, it's essential. The threshold is the same in both domains: is memory the binding constraint? If yes, reduce precision. If no, you're paying overhead without benefit.

## What comes next

The next benchmark runs the same quantization comparison on a 7B model — where memory is the actual bottleneck and quantization should show its real benefit. Hypothesis: at 7B scale, INT4 is faster than FP16 on the T4, not slower.

---

*Previous: [vLLM vs Ollama: I ran the same benchmark on both. 99× difference.](/ai-systems/2026/05/24/vllm-vs-ollama-benchmark/)*

*Next: Quantization on a 7B model — testing the hypothesis where memory actually matters.*

*Full benchmark notebooks: [github.com/anirudhs16/llm-inference-benchmarks](https://github.com/anirudhs16/llm-inference-benchmarks)*
