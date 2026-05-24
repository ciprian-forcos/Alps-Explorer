# 05 - The Local AI Ecosystem: Your Toolbox

So far you have learned what models are and how they work. Now it is time to meet the tools that let you actually run, tune, and ship them on your own hardware. Think of this chapter as a hardware-store walkthrough: each shelf holds a different tool, and the trick is knowing which one to grab for the job in front of you.

## llama.cpp

`llama.cpp` is a small, fast C++ program that runs large language models on regular hardware. It does not need a giant graphics card. It can run on your laptop, on a Raspberry Pi, on an old desktop, even on a phone. It is the project that proved you do not need a data center to talk to a language model.

It is for people who care about speed, control, and tiny dependencies. If you want a single binary you can copy to any machine and just run, this is your friend. It is also the engine that powers many other tools on this list, including Ollama, so learning it gives you a peek under the hood.

You pick `llama.cpp` when you want raw control: when you want to squeeze a model into a tight memory budget using quantization (a way of shrinking models, like compressing a photo), or when you want to deploy without Python at all. You skip it when you want a friendlier interface or fancy serving features.

Install and run a tiny model:

```bash
# Clone and build
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# Download a small GGUF model (TinyLlama, about 600 MB)
curl -L -o tinyllama.gguf \
  https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# Chat with it
./llama-cli -m tinyllama.gguf -p "Hello, who are you?" -n 100
```

Use `llama.cpp` when you want low-level control and minimal dependencies. Use Ollama when you want the same engine with a friendlier wrapper.

**Summary**: `llama.cpp` is the lean, portable C++ runtime that brought local LLMs to everyday hardware.

**Mental model**: It is the engine block of local AI: not pretty, but it is what makes the car move.

**Beginner mistakes**:
- Trying to run a full-size model on 8 GB of RAM. Start with quantized 1B-3B models.
- Forgetting to `make` before running. It is C++, you must build it first.
- Downloading the wrong file format. `llama.cpp` wants `.gguf` files, nothing else.

**Exercise**: Clone the repo, build it, download the TinyLlama GGUF above, and have a 5-message conversation with it. Works on Mac, Linux, and Windows.

## Ollama

Ollama is the easy button for running models locally. It wraps `llama.cpp` (and other engines) in a friendly command-line tool plus a background service. You type one command and a model is downloaded, loaded, and ready to chat. It feels like Docker for language models.

It is for beginners, hobbyists, and anyone who wants to demo a local model in under a minute. It is also great for app developers because it exposes a simple HTTP API on `localhost:11434`, so your Python or web app can call it like any other service.

Pick Ollama when you want zero hassle, when you are still exploring which model fits your needs, or when you want to plug local models into a larger app without learning runtime details. Skip it when you need maximum throughput in production or fine-grained control.

```bash
# Install (macOS/Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull and chat with a tiny model
ollama pull llama3.2:1b && ollama run llama3.2:1b
```

You can also call it from Python:

```python
import requests

response = requests.post(
    "http://localhost:11434/api/generate",
    json={"model": "llama3.2:1b", "prompt": "Why is the sky blue?", "stream": False},
)
print(response.json()["response"])
```

Use Ollama when you want a model running in 30 seconds. Use vLLM when you need to serve many users at once.

**Summary**: Ollama is the friendly front-door to local LLMs, hiding the messy parts behind a single command.

**Mental model**: It is the espresso machine of local AI: one button, hot model.

**Beginner mistakes**:
- Pulling a huge model first. Start with `llama3.2:1b` or `phi3:mini` and work up.
- Forgetting Ollama runs as a background service. If `ollama run` hangs, check it is actually started.
- Assuming Ollama is fast enough for production. It is great for development, not for high-traffic serving.

**Exercise**: Install Ollama, pull `llama3.2:1b`, and ask it three questions. Then write a 10-line Python script that calls the local API. Works on Mac, Linux, and Windows.

## vLLM

vLLM is a serving engine built for speed and scale. It uses a clever memory trick called PagedAttention (think of it like how an operating system manages RAM in pages) to serve many users at the same time without running out of GPU memory. If you have ever wondered how companies serve thousands of chat requests per second, vLLM is one of the answers.

It is for engineers who need to deploy models behind an API, especially when many people will hit that API at once. It is the standard choice for production-grade self-hosted LLM serving on Nvidia GPUs.

You pick vLLM when you have a real GPU (or several), when latency and throughput matter, and when you want an OpenAI-compatible HTTP endpoint with one command. You skip it on Macs (no CUDA) and on CPU-only machines.

```bash
pip install vllm
```

Minimal Python usage:

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct")
sampling_params = SamplingParams(temperature=0.7, max_tokens=100)

outputs = llm.generate(["Explain gravity in one sentence."], sampling_params)
for output in outputs:
    print(output.outputs[0].text)
```

Or serve it as an API:

```bash
vllm serve meta-llama/Llama-3.2-1B-Instruct
```

Use vLLM when you need to serve a model to many users. Use Ollama when it is just you and your laptop.

**Summary**: vLLM is the production-grade serving engine for fast, concurrent LLM inference on GPUs.

**Mental model**: It is the highway with many lanes: each request gets its own lane, and nobody waits long.

**Beginner mistakes**:
- Trying to install on a Mac. vLLM needs CUDA, so it expects an Nvidia GPU on Linux (or WSL on Windows).
- Loading a 70B model on a single 8 GB GPU. Read the model card for memory requirements first.
- Confusing the offline `LLM(...)` API with the server. They share a class but have different patterns.

**Exercise**: On a Linux box with an Nvidia GPU, install vLLM and serve `meta-llama/Llama-3.2-1B-Instruct`. Call it with `curl` using the OpenAI-compatible endpoint. If you do not have an Nvidia GPU, read the quickstart docs and move on; this tool is platform-specific.

## MLX

MLX is Apple's machine learning framework, built from the ground up for Apple Silicon (the M1, M2, M3, M4 chips). It uses unified memory, which means the CPU and GPU share the same RAM with no copying. That makes Macs surprisingly capable for running and even training small to mid-size models.

It is for Mac users who want to take advantage of their hardware. The `mlx-lm` package gives you a simple way to run language models with MLX as the backend. Performance on M-series chips is often shockingly good for the price.

Pick MLX when you are on a Mac with Apple Silicon and want native, optimized performance. Skip it on Intel Macs, Windows, or Linux. It only runs on Apple Silicon.

```bash
# Apple Silicon Macs only (M1, M2, M3, M4)
pip install mlx-lm
```

Minimal usage:

```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")

prompt = "What is a good name for a pet rock?"
response = generate(model, tokenizer, prompt=prompt, max_tokens=80)
print(response)
```

Or from the command line:

```bash
mlx_lm.generate --model mlx-community/Llama-3.2-1B-Instruct-4bit \
  --prompt "Hello there"
```

Use MLX on a Mac. Use `llama.cpp` or Ollama if you want to write code that runs anywhere.

**Summary**: MLX is Apple's native ML framework, and `mlx-lm` is the fastest way to run language models on Apple Silicon.

**Mental model**: It is a sports car tuned for one specific road, and that road is your MacBook.

**Beginner mistakes**:
- Trying to install on an Intel Mac or a Linux box. It will not work.
- Downloading non-MLX model files. Use models from the `mlx-community` organization on Hugging Face.
- Expecting to train a 70B model on a laptop. Even with unified memory, hardware limits exist.

**Exercise**: On an Apple Silicon Mac, install `mlx-lm`, run the snippet above, and try a 4-bit quantized 1B model. On any other platform, skip this exercise and use Ollama instead.

## Hugging Face

Hugging Face is the GitHub of machine learning. It is a website plus a set of Python libraries where people share models, datasets, and demos. When you see a model name like `meta-llama/Llama-3.2-1B`, that is a Hugging Face repository path. Almost every tool in this chapter pulls models from Hugging Face under the hood.

It is for everyone in AI. Whether you are a researcher, a hobbyist, or an engineer, you will end up here. The two libraries you will use most are `transformers` (for loading and running models) and `datasets` (for loading training data).

Use Hugging Face when you want to find a model, share a model, or write Python code that loads any model with a few lines. It is the universal hub.

```bash
pip install transformers datasets
```

Minimal "hello world":

```python
from transformers import pipeline

pipe = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-135M-Instruct")
result = pipe("The best thing about cats is", max_new_tokens=30)
print(result[0]["generated_text"])
```

Loading a dataset is just as easy:

```python
from datasets import load_dataset

ds = load_dataset("squad", split="train[:5]")
print(ds[0])
```

Use Hugging Face `transformers` for flexibility and experimentation. Use Ollama or vLLM when you need a serving runtime.

**Summary**: Hugging Face is the central hub and toolkit for finding, loading, and sharing AI models and datasets.

**Mental model**: It is the public library of AI: anyone can borrow, anyone can publish.

**Beginner mistakes**:
- Downloading a gated model without logging in. Run `huggingface-cli login` first for models like Llama.
- Loading a 13B model on a CPU and waiting forever. Match model size to your hardware.
- Forgetting that `transformers` is not optimized for production serving. Use vLLM or Ollama for that.

**Exercise**: `pip install transformers`, run the SmolLM snippet above, then browse the Hugging Face Hub and load a different small model. Works on Mac, Linux, and Windows.

## Unsloth

Unsloth is a library that makes fine-tuning small and mid-size language models dramatically faster and lighter on memory. It rewrites key parts of the training loop in optimized kernels so you can train a 7B model on a single consumer GPU, sometimes in under an hour. It is one of the most beginner-friendly entry points to fine-tuning.

It is for people who want to teach a model a new skill or new domain without renting a cluster. Hobbyists, indie developers, and small teams love it. It plugs into the rest of the Hugging Face ecosystem, so you do not have to learn a whole new stack.

Pick Unsloth when you want to fine-tune on a single GPU and want sane defaults. Skip it if you need multi-GPU distributed training (use Axolotl or raw `transformers` + Accelerate for that).

```bash
pip install unsloth
```

Minimal model load (the first step of any fine-tune):

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Add LoRA adapters for fine-tuning
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
)
```

Use Unsloth when training on one GPU. Use Axolotl when scaling to several.

**Summary**: Unsloth is the fastest, friendliest way to fine-tune small LLMs on a single GPU.

**Mental model**: It is a turbocharger bolted onto your training script: same engine, much more speed.

**Beginner mistakes**:
- Trying to install on a Mac. Unsloth needs CUDA.
- Loading a model in full precision when 4-bit would fit. Use `load_in_4bit=True` when memory is tight.
- Skipping LoRA and trying full fine-tuning. Start with LoRA; it is faster and good enough most of the time.

**Exercise**: On a machine with an Nvidia GPU, install Unsloth and load `unsloth/Llama-3.2-1B-Instruct` in 4-bit. You do not need to actually train; just confirm it loads. If you do not have a CUDA GPU, read the docs and skip.

## Axolotl

Axolotl is a fine-tuning framework built on top of Hugging Face `transformers`. Where Unsloth is one fast Python class, Axolotl is a whole config-driven training system. You write a YAML file describing your model, dataset, and training settings, and Axolotl takes care of the rest, including multi-GPU and multi-node setups.

It is for serious tinkerers, research teams, and anyone who wants reproducible, version-controlled fine-tuning. Because everything is a YAML file, you can check it into git and re-run the exact same experiment later.

Pick Axolotl when you have multiple GPUs, when you want declarative configs, or when you want to try many training recipes without rewriting code. Skip it for quick one-off experiments where a Python script is simpler.

```bash
pip install axolotl
```

A minimal config file (`config.yml`):

```yaml
base_model: meta-llama/Llama-3.2-1B-Instruct
load_in_4bit: true

datasets:
  - path: tatsu-lab/alpaca
    type: alpaca

adapter: lora
lora_r: 16
lora_alpha: 32
lora_target_modules:
  - q_proj
  - v_proj

sequence_len: 2048
micro_batch_size: 2
num_epochs: 1
learning_rate: 0.0002
output_dir: ./out
```

Then launch training with:

```bash
axolotl train config.yml
```

Use Axolotl when you want reproducible, multi-GPU training. Use Unsloth when you want a fast single-GPU script.

**Summary**: Axolotl is a config-driven fine-tuning framework, ideal for reproducible and multi-GPU training runs.

**Mental model**: It is the recipe card of fine-tuning: write the recipe once, cook it the same way every time.

**Beginner mistakes**:
- Copying a config for a 70B model when you have a 24 GB GPU. Match the recipe to your hardware.
- Skipping the `sequence_len` setting. It controls memory use more than almost any other knob.
- Forgetting to install the right CUDA-matching PyTorch version. Follow the official install instructions carefully.

**Exercise**: Write the YAML above, but change the `base_model` to a tiny one like `HuggingFaceTB/SmolLM2-135M-Instruct`. Run `axolotl train config.yml` on a Linux box with a GPU. On other platforms, just write the config and read it back; do not try to train.

## PEFT

PEFT stands for Parameter-Efficient Fine-Tuning. It is a Hugging Face library that implements the math behind techniques like LoRA, QLoRA, and prefix tuning. The big idea: instead of updating all billions of weights when you fine-tune, you only train a tiny set of extra weights (called adapters). This makes fine-tuning massively cheaper and faster.

It is for anyone who wants to fine-tune a model without paying for a supercomputer. PEFT is the engine that Unsloth and Axolotl both use under the hood, so learning the basics here helps you understand the whole stack.

You pick PEFT directly when you want to write your own training loop with full control. You let Unsloth or Axolotl call PEFT for you when you prefer convenience.

```bash
pip install peft
```

A minimal LoRA setup:

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

That last line will show that you are only training a tiny fraction of the model's weights, often less than 1 percent.

Use PEFT when you want direct control over adapters. Use Unsloth or Axolotl when you want the same thing with less code.

**Summary**: PEFT is the Hugging Face library for adapter-based fine-tuning techniques like LoRA.

**Mental model**: It is sticky notes on a textbook: the book stays unchanged, but the notes teach it new tricks.

**Beginner mistakes**:
- Choosing `r` (the rank) too high. Start with 8 or 16; bigger is not always better.
- Targeting the wrong modules. Different model architectures name their layers differently. Check the model's config.
- Forgetting to save the adapter separately from the base model. The whole point is that the adapter is small.

**Exercise**: Install `peft` and `transformers`, load SmolLM2-135M, wrap it with the LoRA config above, and print the number of trainable parameters. Works on any platform; the model is tiny enough for CPU.

## TRL Library

TRL stands for Transformer Reinforcement Learning. It is the Hugging Face library for the training steps that come after standard fine-tuning, including supervised fine-tuning (SFT), reward modeling, DPO (Direct Preference Optimization), and PPO (Proximal Policy Optimization). It is how teams turn a raw pretrained model into something that follows instructions and behaves politely.

It is for anyone who wants to align a model to a specific style, set of preferences, or task. If you have ever wondered how chatbots learn to be helpful and not rude, this is the toolkit.

Pick TRL when you have preference data (pairs of "good" and "bad" responses) or when you want a clean, high-level API for SFT. It works seamlessly with PEFT, so you can do LoRA-based DPO without writing much code.

```bash
pip install trl
```

A minimal SFTTrainer setup:

```python
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset = load_dataset("trl-lib/Capybara", split="train[:100]")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        output_dir="./sft-out",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        max_seq_length=512,
    ),
)

trainer.train()
```

Use TRL for SFT and preference-based alignment. Use PEFT directly when you only need adapter math without the training loop.

**Summary**: TRL is the Hugging Face library for supervised fine-tuning and preference-based alignment (DPO, PPO, reward models).

**Mental model**: It is the etiquette school for models: it teaches them how to behave, not just what facts to know.

**Beginner mistakes**:
- Jumping straight to PPO. Start with SFT, then DPO. PPO is the most complex and finickiest of the bunch.
- Using a tiny dataset and expecting magic. Even SFT usually needs thousands of examples to move the needle.
- Forgetting that TRL stacks on top of PEFT and `transformers`. Learn those first.

**Exercise**: Install `trl`, `transformers`, and `datasets`. Run the SFTTrainer snippet above with SmolLM2-135M for 1 epoch on a 100-row slice of Capybara. It will run on CPU in a few minutes. Works on Mac, Linux, and Windows.

## What's Next

You now have a map of the local AI toolbox: runtimes (`llama.cpp`, Ollama, vLLM, MLX), the hub (Hugging Face), and the fine-tuning stack (Unsloth, Axolotl, PEFT, TRL). In the next chapter, [`06-rag-memory.md`](./06-rag-memory.md), we will go beyond the model itself and look at how to give it memory and access to your own documents using Retrieval-Augmented Generation (RAG).
