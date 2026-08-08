---
layout: post
title: "The AI vendor landscape: a practical decision framework across 4 layers"
date: 2026-08-02
categories: [architecture]
tags: [vendor-comparison, llm-providers, vector-databases, inference-serving, architecture]
---

One of the most common questions when building AI systems is not which model is best. It is which combination of vendors, at which layer of the stack, makes sense for a specific use case.

After building a production RAG chatbot, running benchmarks across multiple inference frameworks, and studying enterprise AI architectures, I put together a reference matrix covering every layer of the AI stack. This is the document I wish existed when I started.

## Why a 4-layer view

Every production AI system has the same four layers underneath it, regardless of what it does on the surface:

**Layer 1 — LLM providers:** where intelligence comes from. Closed API (OpenAI, Anthropic, Google, Groq) or open-weight self-hosted (Llama, Mistral, Phi).

**Layer 2 — Vector databases:** where semantic memory lives. Used in RAG pipelines to store and retrieve embeddings. Qdrant, Pinecone, ChromaDB, pgvector.

**Layer 3 — Inference serving frameworks:** the software that takes raw model weights and turns them into a working API endpoint. Ollama, vLLM, TensorRT-LLM, llama.cpp.

**Layer 4 — Orchestration frameworks:** the glue that connects models, databases, tools, and user interfaces into multi-step workflows. LangChain, LangGraph, LlamaIndex.

The full matrix is on GitHub: [github.com/anirudhs16/llm-inference-benchmarks](https://github.com/anirudhs16/llm-inference-benchmarks)

## Layer 1 — LLM providers

The closed API vs open-weight decision is usually made first, and it drives most of the downstream choices.

**Closed APIs** (OpenAI, Anthropic, Google, Groq, Mistral) are the right default when you want to move fast, don't have GPU infrastructure, and don't have strict data privacy requirements. The tradeoff is cost at scale and vendor dependency.

**Open-weight self-hosted** (Llama 3, Mistral, Phi) is the right call when data cannot leave your infrastructure, when you're processing enough volume that API costs exceed hosting costs, or when you need to fine-tune.

The decision criteria that matter most in practice:

- Data residency and compliance constraints
- Request volume (when does self-hosting become cheaper than API?)
- Latency requirements (Groq's LPU hardware is genuinely the fastest API option for latency-sensitive workloads)
- Need for fine-tuning

One thing I learned from benchmarking: Groq is not just "another API." Their LPU (Language Processing Unit) hardware produces consistently lower latency than GPU-based serving for the same models. If you are building something where TTFT matters — chatbots, copilots, anything interactive — Groq is worth evaluating seriously.

## Layer 2 — Vector databases

The most common mistake here is choosing a vector database before knowing your scale. ChromaDB is excellent for prototyping and terrible at production scale. Pinecone is fully managed and easy but creates vendor lock-in and becomes expensive at volume. Qdrant gives you production-grade performance, hybrid search (dense + sparse vectors combined), and the ability to self-host when you outgrow the cloud tier.

My quick decision rule:

- Prototyping → ChromaDB
- Already on Postgres, want simplicity → pgvector
- Production RAG, want managed → Pinecone
- Production RAG, want control + hybrid search → Qdrant
- Billions of vectors, enterprise scale → Milvus

I used Qdrant in my production RAG system and the hybrid search capability — combining embedding similarity with keyword matching — meaningfully improved retrieval quality over embedding-only search.

## Layer 3 — Inference serving frameworks

This is the layer I have the most hands-on experience with, having benchmarked Ollama vs vLLM directly.

The key insight from that benchmark: Ollama serves requests sequentially. At 4 concurrent users, per-request latency was 1.8× worse than single-user latency. vLLM's continuous batching and PagedAttention meant per-request latency at 4 concurrent users was actually *better* than at 2 users — GPU parallelism kicking in.

That 99× speedup between the two frameworks comes from architectural decisions, not model differences. Ollama is the right tool for local development. vLLM is the right tool for production serving.

For maximum GPU performance on NVIDIA hardware, TensorRT-LLM sits above vLLM in raw throughput but requires significantly more operational complexity. For CPU-only or edge deployment, llama.cpp running GGUF-quantized models is the standard.

## Layer 4 — Orchestration frameworks

This layer is where I have the most production experience. I used LangGraph at Conflux to build stateful multi-step AI workflows — the graph-based approach to agent orchestration makes the state machine explicit and debuggable in a way that LangChain's sequential chains do not.

The rough decision rule:

- Simple RAG pipeline → LangChain (largest ecosystem, most integrations)
- Stateful agents, multi-step workflows → LangGraph (explicit state, better for production)
- Heavy document processing, ingestion pipelines → LlamaIndex (best data connectors)
- Enterprise, Microsoft/Azure ecosystem → Semantic Kernel

## The decision that actually matters

Building AI systems is not a model selection problem. It is a stack selection problem. Every layer has real tradeoffs: cost, latency, vendor dependency, operational complexity, compliance.

The matrix does not answer which vendor is best. It provides a consistent set of criteria for making the decision that matches your specific constraints. The same company might use Groq for their customer-facing chatbot (latency), Llama 3 on vLLM for their internal data processing (privacy), and ChromaDB for their hackathon prototype (speed of development).

Context determines the right answer. The matrix gives you the questions to ask.

---

*Previous: [Speculative decoding: I ran it. The results were messier than expected.](/ai-systems/2026/07/05/speculative-decoding-benchmark/)*

*Next: Enterprise AI architecture case study — designing a production customer support system from scratch.*

*Vendor comparison matrix: [github.com/anirudhs16/llm-inference-benchmarks](https://github.com/anirudhs16/llm-inference-benchmarks)*
