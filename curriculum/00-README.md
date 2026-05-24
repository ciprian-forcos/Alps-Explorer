# LLM Engineering Curriculum

A self-paced path from "I've used ChatGPT" to "I can fine-tune, deploy, and ship AI products."

Written in plain English with daily-life analogies. Every topic ends with a summary, a one-sentence mental model, common beginner mistakes, and a hands-on exercise you can run on a laptop (with notes when a GPU is needed).

## How to use this

1. Read one section per sitting. Don't rush.
2. **Do the exercises.** Reading about fine-tuning teaches you nothing — running a LoRA on a 1B model for ten minutes teaches you a lot.
3. Keep a `notes/` folder next to this one. Jot what surprised you, what broke, what you'd try next.
4. Skip ahead if a section is review, but always do its exercise — exercises are the checkpoints.

## Prerequisites

- Comfort with Python basics (functions, dicts, virtualenvs).
- A machine that can run Python 3.10+.
- Recommended: install [Ollama](https://ollama.com) early — most exercises use it because it runs CPU-only.
- For fine-tuning sections: free Google Colab (T4 GPU) is enough for everything in this curriculum.

```bash
# One-time setup for most exercises
python -m venv .venv && source .venv/bin/activate
pip install transformers datasets sentence-transformers chromadb \
            peft trl bitsandbytes accelerate \
            ollama tiktoken fastapi uvicorn
# Plus Ollama: https://ollama.com/download
ollama pull llama3.2:1b
ollama pull qwen2.5:0.5b
```

## Sections

1. [Foundations](./01-foundations.md) — what an LLM actually is, tokens, embeddings, attention, training vs inference.
2. [Datasets & Training](./02-datasets-training.md) — SFT, instruction tuning, DPO pairs, synthetic data, curation.
3. [Fine-Tuning](./03-fine-tuning.md) — LoRA, QLoRA, DPO, RLHF, quantization, GGUF.
4. [Inference & Optimization](./04-inference-optimization.md) — KV cache, Flash Attention, speculative decoding, serving, batching, GPU/VRAM.
5. [Local AI Ecosystem](./05-local-ecosystem.md) — llama.cpp, Ollama, vLLM, MLX, Hugging Face, Unsloth, Axolotl, PEFT, TRL.
6. [RAG & Memory](./06-rag-memory.md) — retrieval-augmented generation, vector DBs, chunking, semantic search, memory.
7. [Agents & Workflows](./07-agents-workflows.md) — prompting, system prompts, tool/function calling, agent loops, multi-agent, browser agents.
8. [Model Types](./08-model-types.md) — VLMs, SLMs, dense vs MoE, coding models, reasoning models.
9. [Deployment](./09-deployment.md) — local, on-device, API serving, cloud GPUs, edge.
10. [Evaluation](./10-evaluation.md) — benchmarks, human eval, cost/speed/quality, LLM-as-judge.
11. [Real-World Skills](./11-real-world-skills.md) — chatbots, copilots, automation, SaaS workflows, coding workflows, orchestration, product thinking.

## A suggested pace

- **Week 1–2:** sections 1–3 (foundations + first LoRA on Colab).
- **Week 3:** sections 4–5 (inference and the local toolbox).
- **Week 4:** section 6 (RAG — build one over your own notes).
- **Week 5:** section 7 (agents — build a tool-using agent).
- **Week 6:** sections 8–10 (model types, deployment, eval — pick what's relevant to your project).
- **Week 7+:** section 11 — ship one small AI tool you'll actually use.

## The honest rule

You don't learn LLMs by reading. You learn by running a small model, watching it fail, and figuring out why. This curriculum exists to make those moments cheaper and faster — not to replace them.
