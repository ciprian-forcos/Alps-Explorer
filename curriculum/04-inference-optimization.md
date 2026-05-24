# Inference & Optimization

Training a model is like building a kitchen; inference is cooking dinner every night for thousands of hungry guests. This chapter walks you through the tricks that make that nightly service fast, cheap, and reliable.

## KV cache

When a language model generates text, it reads every previous token to decide the next one. The "K" and "V" stand for Keys and Values, the intermediate numbers attention uses to look back at the prompt. Without a cache, the model would re-compute those numbers for every previous token on every new step. That is like re-reading the entire book from page one every time you want to read the next word.

The KV cache fixes this by saving those Keys and Values in GPU memory after the first pass, then only computing them for the brand-new token each step. Think of a waiter who writes down your order once instead of asking you to repeat the whole thing for every course.

The tradeoff is memory. A long conversation builds a long cache, and the cache lives in VRAM. If you run out of VRAM, generation slows down or crashes.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import time, torch

name = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16).to("cuda")

prompt = tok("Explain photosynthesis in one paragraph.", return_tensors="pt").to("cuda")

for use_cache in (False, True):
    start = time.time()
    out = model.generate(**prompt, max_new_tokens=120, use_cache=use_cache)
    print(f"use_cache={use_cache}: {time.time()-start:.2f}s")
```

**Summary**: The KV cache stores attention Keys and Values from past tokens so each new token only requires a tiny amount of fresh work.

**Mental model**: A notebook the waiter keeps so you do not have to repeat your order.

**Beginner mistakes**
- Disabling `use_cache` "to save memory" and getting a 10x slowdown.
- Forgetting that long prompts inflate the cache and can exhaust VRAM.
- Comparing speed without warming up the GPU first; the first call is always slower.

**Exercise**: Run the snippet above on Qwen2.5-0.5B. Record both timings. Then bump `max_new_tokens` to 400 and watch the gap grow. Roughly how many times faster is the cached version?

## Flash Attention

Standard attention multiplies big matrices and writes the result back to VRAM at every step. VRAM reads and writes are slow compared to the actual math. Flash Attention is a rewrite of the attention kernel that keeps numbers inside the GPU's tiny on-chip memory for as long as possible, only writing the final answer back.

The analogy: cooking on a small cutting board next to the stove versus running to the pantry between every chop. Same recipe, far fewer trips, much faster meal.

You usually do not write Flash Attention yourself; you turn it on with a flag. Most modern inference frameworks (vLLM, TGI, Ollama) ship with it enabled by default. In Hugging Face Transformers you opt in explicitly.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

name = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(
    name,
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",
).to("cuda")

prompt = tok("List three causes of the French Revolution.", return_tensors="pt").to("cuda")
print(tok.decode(model.generate(**prompt, max_new_tokens=80)[0], skip_special_tokens=True))
```

**Summary**: Flash Attention is a smarter way to compute the same attention numbers by minimizing slow memory traffic on the GPU.

**Mental model**: A chef who keeps the knife, board, and ingredients within arm's reach instead of walking to the pantry.

**Beginner mistakes**
- Expecting Flash Attention to change the model's outputs; it does not, only speed.
- Installing it on a GPU it does not support (it needs Ampere or newer, e.g. RTX 30xx and up).
- Thinking it helps short prompts a lot; the win grows with sequence length.

**Exercise**: Install `pip install flash-attn --no-build-isolation` on a Colab T4 or better. Time a 1024-token generation with and without `attn_implementation="flash_attention_2"`. Record both numbers.

## Speculative decoding

Generating one token at a time is the bottleneck of inference. Speculative decoding uses a small, fast "draft" model to guess the next several tokens, then a big, slow "target" model checks the guesses in a single pass. If the guesses are right, you got many tokens for the cost of one big-model step. If they are wrong, you fall back to one token, no worse than before.

Think of a junior writer drafting a paragraph and a senior editor scanning it. When the draft is good, the editor signs off in seconds. When it is bad, the editor rewrites a sentence. Either way, the editor is never slower than writing from scratch.

The win depends on how often the draft model agrees with the big one. Pair models from the same family for best results, like Qwen2.5-0.5B drafting for Qwen2.5-7B.

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    speculative_model="Qwen/Qwen2.5-0.5B-Instruct",
    num_speculative_tokens=5,
)

out = llm.generate(
    ["Write a haiku about the Alps."],
    SamplingParams(temperature=0.7, max_tokens=64),
)
print(out[0].outputs[0].text)
```

**Summary**: A tiny model guesses, a big model verifies in bulk, and you get faster generation without losing quality.

**Mental model**: A junior drafts the email, the senior glances and approves whole paragraphs.

**Beginner mistakes**
- Pairing unrelated models; agreement rates plummet and you lose all the speedup.
- Setting `num_speculative_tokens` too high; rejections become expensive.
- Expecting it to help with very short outputs; the overhead eats the gains.

**Exercise**: Run the vLLM snippet above on Colab with an A100 or T4 (use 7B in 4-bit if VRAM is tight). Compare tokens-per-second with and without the `speculative_model` argument.

## Inference optimization

Inference optimization is the umbrella term for everything that makes a trained model run faster or cheaper without retraining it. The big levers are: quantization (use 8-bit or 4-bit numbers instead of 16-bit), KV cache reuse, Flash Attention, speculative decoding, batching, and choosing a serving framework built for throughput.

Think of optimizing a delivery route. You can buy a faster truck (better GPU), pack more boxes per trip (batching), pre-write addresses (cache), or send a scout ahead (speculation). The best result usually comes from stacking several small wins.

A practical first step is to swap raw `transformers` for a serving framework. vLLM is the most popular for GPUs; Ollama wraps llama.cpp for CPU and small GPUs. Both apply many optimizations automatically.

```bash
pip install vllm
python -c "
from vllm import LLM, SamplingParams
llm = LLM(model='Qwen/Qwen2.5-0.5B-Instruct', dtype='float16')
out = llm.generate(['Name three mountains in the Alps.'], SamplingParams(max_tokens=50))
print(out[0].outputs[0].text)
"
```

**Summary**: Optimization is a toolbox of tricks (quantization, caching, batching, better kernels) that stack together to cut latency and cost.

**Mental model**: Tuning a race car: many small upgrades add up to a much faster lap.

**Beginner mistakes**
- Optimizing before measuring; you cannot improve what you have not timed.
- Chasing one trick (only quantization, only batching) and ignoring the rest.
- Forgetting that aggressive optimization can degrade output quality.

**Exercise**: Generate 200 tokens from Qwen2.5-0.5B with plain `transformers`, then again with vLLM. Compare wall-clock time. Note that vLLM has a one-time startup cost; the second call is the fair comparison.

## Model serving

A model in a notebook is a science experiment. A model behind an HTTP endpoint is a product. Model serving is the practice of running a model as a long-lived service that accepts requests, batches them, and returns responses fast.

Imagine a coffee shop. A serving framework is the espresso machine, the barista's workflow, and the queue management. vLLM, Text Generation Inference (TGI), Ollama, and TensorRT-LLM are the popular machines. They handle concurrent users, streaming responses, and OpenAI-compatible APIs so your client code is unchanged when you swap models.

Ollama is the easiest start: one binary, one command. It exposes an HTTP API on localhost:11434 that you can call from any language.

```bash
ollama pull qwen2.5:0.5b
ollama serve &
```

```python
import requests, json

resp = requests.post(
    "http://localhost:11434/api/generate",
    json={"model": "qwen2.5:0.5b", "prompt": "What is the tallest peak in the Alps?", "stream": False},
)
print(resp.json()["response"])
```

**Summary**: Model serving turns a model into a network service that real applications can talk to, with batching and concurrency handled for you.

**Mental model**: An espresso machine plus a barista, not a chemistry set.

**Beginner mistakes**
- Building your own HTTP wrapper around `transformers` and discovering it cannot handle two users at once.
- Forgetting authentication and exposing a public endpoint by mistake.
- Loading the model on every request instead of keeping it warm in memory.

**Exercise**: Install Ollama from ollama.com. Pull `qwen2.5:0.5b`. Hit the API from Python as shown above. Then send 5 requests in parallel with `concurrent.futures` and watch how response times change.

## Batch inference

When a model processes one prompt, the GPU is barely working; most of its thousands of cores sit idle. Batching means stacking many prompts into one forward pass so the GPU is busy. The total time for a batch of 8 is often only slightly more than the time for 1.

Picture a school bus. Running it for one passenger or for forty costs nearly the same fuel. Fill the bus.

There are two flavors. Static batching processes a fixed group together. Continuous batching, which vLLM pioneered, swaps finished requests out and new ones in token by token, keeping the GPU full at all times. For any real service, you want continuous batching.

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", dtype="float16")

prompts = [
    "Translate 'good morning' to French.",
    "Translate 'good morning' to German.",
    "Translate 'good morning' to Italian.",
    "Translate 'good morning' to Spanish.",
    "Translate 'good morning' to Japanese.",
    "Translate 'good morning' to Arabic.",
    "Translate 'good morning' to Hindi.",
    "Translate 'good morning' to Swahili.",
]

import time
start = time.time()
outs = llm.generate(prompts, SamplingParams(max_tokens=30))
print(f"Batch of {len(prompts)}: {time.time()-start:.2f}s")
for o in outs:
    print("-", o.outputs[0].text.strip())
```

**Summary**: Batching runs many prompts in one GPU pass, so throughput grows almost for free.

**Mental model**: One school bus instead of forty taxis.

**Beginner mistakes**
- Batching by hand in `transformers` and getting confused by padding.
- Using huge batches that exceed VRAM and crash mid-request.
- Confusing throughput (tokens per second total) with latency (time per single user).

**Exercise**: Run the snippet, then re-run with `prompts = prompts[:1]`. Compare total time. Compute tokens-per-second for each and notice how the batch of 8 is far more efficient per prompt.

## GPU basics

A GPU is a giant grid of small, simple processors. A CPU has 8 to 64 strong cores that handle one complex task each; a GPU has thousands of weak cores that handle the same simple task in parallel. Matrix multiplication, the math behind neural networks, fits this model perfectly.

Compare a chef (CPU) cooking one elaborate dish versus a hundred line cooks (GPU) each frying one egg. For omelets, the line cooks win by a mile. For a soufflé, the chef wins.

You will mostly care about three numbers: VRAM (memory on the GPU), TFLOPS (raw compute), and memory bandwidth (how fast data moves on the chip). For LLM inference, bandwidth often matters more than raw TFLOPS, because the model spends most of its time reading weights from VRAM.

```bash
nvidia-smi
```

A typical output shows GPU name, driver version, VRAM used and total, temperature, and which processes are using the card. Run it in another terminal while a model is generating to watch VRAM and utilization climb.

**Summary**: GPUs win at the parallel, repetitive math that neural networks need; CPUs are better at one thing at a time.

**Mental model**: A hundred line cooks versus one master chef. Pick the kitchen for the menu.

**Beginner mistakes**
- Buying a GPU based on TFLOPS alone and ignoring VRAM.
- Assuming any GPU with "RTX" in the name supports Flash Attention 2 (Pascal/Turing do not).
- Forgetting to install matching CUDA drivers; nothing works without them.

**Exercise**: Open two terminals. In one, run `watch -n 0.5 nvidia-smi`. In the other, run the Qwen2.5-0.5B snippet from the KV cache section. Watch GPU utilization jump from idle to near 100 percent during generation.

## VRAM basics

VRAM is the GPU's private memory. The model weights, the KV cache, activations, and any inputs all live there. Run out, and you get the dreaded `CUDA out of memory` error.

A rough rule for inference: a model in 16-bit precision needs about 2 GB of VRAM per billion parameters. So a 7B model wants around 14 GB just for weights. In 4-bit quantization that drops to about 4 GB. Add another few GB for KV cache on longer contexts.

Think of VRAM as the counter space in your kitchen. The model is the prep work; the cache is the dishes piling up as you cook. Big counter, more dishes; small counter, you have to wash mid-recipe.

```bash
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv
```

In Python you can check from inside a process:

```python
import torch
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Reserved:  {torch.cuda.memory_reserved()/1e9:.2f} GB")
print(f"Total:     {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
```

**Summary**: VRAM holds weights, cache, and activations; size your model and context length to fit, or quantize to shrink them.

**Mental model**: Kitchen counter space. Bigger model and longer chat both pile on more dishes.

**Beginner mistakes**
- Loading a 13B model in fp16 on a 12 GB card and hitting OOM.
- Ignoring KV cache growth on long chats and crashing after a few thousand tokens.
- Forgetting that other processes (browser, desktop) eat a few hundred MB of VRAM.

**Exercise**: Load Qwen2.5-0.5B in fp16, then in 4-bit (`load_in_4bit=True` with `bitsandbytes`). Print `torch.cuda.memory_allocated()` after each load and compare.

## Latency vs quality tradeoffs

Every optimization knob is a tradeoff. Quantize to 4-bit and your model loses a sliver of accuracy. Use a smaller model and answers get duller. Crank batching up and individual users wait longer in the queue. Lower `max_tokens` and replies get cut off.

The analogy is takeout food. You can have it fast, cheap, or gourmet; pick two. A live chatbot needs low latency; an overnight batch job summarizing emails can wait minutes per request if it costs ten times less.

The honest workflow is: pick a quality bar (e.g. "must score 80 percent on my eval set"), then optimize for cost and speed until you hit it. Measure on real prompts, not benchmarks someone else published.

```python
from vllm import LLM, SamplingParams
import time

llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", dtype="float16")
prompt = ["Explain why the sky is blue, in one short paragraph."]

for max_tok in (32, 128, 512):
    start = time.time()
    out = llm.generate(prompt, SamplingParams(max_tokens=max_tok, temperature=0.2))
    elapsed = time.time() - start
    text = out[0].outputs[0].text
    print(f"max_tokens={max_tok}: {elapsed:.2f}s, {len(text.split())} words")
```

**Summary**: Faster, cheaper, smarter, pick two; the right balance depends on whether a human is waiting.

**Mental model**: Takeout food. Fast, cheap, gourmet, choose any two.

**Beginner mistakes**
- Optimizing without an evaluation set, then noticing quality dropped weeks later.
- Using the same settings for an interactive chat and a nightly batch job.
- Assuming a benchmark score on someone else's data predicts your users' experience.

**Exercise**: Take one prompt from your own use case. Generate answers with Qwen2.5-0.5B and Qwen2.5-7B, both at temperature 0.2. Time each. Decide which one you would actually ship and write down why in one sentence.

## What's next

Now that you can make a model run fast, the next chapter, `05-local-ecosystem.md`, covers the tools that wrap all this up for everyday use: Ollama, LM Studio, llama.cpp, and the rest of the local-first ecosystem.
