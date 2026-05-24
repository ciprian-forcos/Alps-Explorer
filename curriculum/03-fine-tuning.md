# Fine-Tuning Techniques

Fine-tuning is how you take a general-purpose model and teach it to do your specific job well. In this chapter we walk through the modern toolbox piece by piece, with small runnable examples and honest notes about which parts need a GPU.

## LoRA

Imagine you bought a pre-built bookshelf from a furniture store. It is great, but you want it to hold your collection of vinyl records, which are heavier and a different shape. You could rebuild the whole shelf from scratch, or you could screw on a few small reinforcement brackets. LoRA, which stands for Low-Rank Adaptation, is the bracket approach. Instead of retraining every weight in a giant model, you freeze the original weights and train a tiny pair of extra matrices that "nudge" the outputs in the direction you want.

Why does this work? Because most fine-tuning tasks do not need to rewrite what the model already knows. They just need a small correction. LoRA adds two skinny matrices (A and B) next to each big weight matrix. During training, only A and B move. The original model stays frozen, untouched. When you are done, the LoRA file is often less than 1% of the size of the base model, so you can keep many of them and swap them in and out like cartridges.

```python
# pip install peft transformers accelerate
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

lora_config = LoraConfig(
    r=8,                       # rank of the adapter matrices
    lora_alpha=16,             # scaling factor
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# e.g. trainable params: 1,126,400 || all params: 1,101,174,784 || trainable%: 0.10
```

**Summary**: LoRA trains a small set of extra parameters while freezing the base model, giving you most of the quality of full fine-tuning at a fraction of the cost.

**Mental model**: Bolt-on brackets that reshape a sturdy shelf without rebuilding it.

**Beginner mistakes**:
- Setting `r` too high (32+) on a small dataset, which overfits and wastes memory.
- Targeting only `q_proj` and forgetting the other attention projections when quality stalls.
- Forgetting to save the adapter separately; you need both the base model and the LoRA file at inference.

**Exercise**: Run the snippet above on Google Colab with a free T4 GPU (TinyLlama-1.1B fits comfortably in 8GB VRAM). Then add a small training loop using a dataset like `yahma/alpaca-cleaned` and confirm only ~0.1% of parameters are trainable.

## QLoRA

QLoRA is LoRA with one more trick: it quantizes the frozen base model down to 4 bits before training. Picture moving house. The big heavy furniture (the frozen base model) gets vacuum-sealed and compressed so it fits in your small van, while you still have room for the new boxes (the LoRA adapters). The base never gets unpacked at full size during training; the math happens with the compressed version.

This is a game changer for hobbyists. A 7B parameter model in full 16-bit precision needs about 14GB of VRAM just sitting there idle. The same model in 4-bit needs around 4GB. Suddenly you can fine-tune a 7B model on a single consumer GPU like an RTX 3060, or even on Colab's free T4. The trade-off is a tiny accuracy hit during training, which is usually invisible in practice.

```python
# pip install bitsandbytes peft transformers accelerate
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model_id = "Qwen/Qwen2.5-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)

model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
```

**Summary**: QLoRA compresses the frozen base to 4 bits so LoRA fine-tuning fits on small GPUs without meaningful quality loss.

**Mental model**: Vacuum-pack the heavy furniture so the moving van still has room for new boxes.

**Beginner mistakes**:
- Using QLoRA on a tiny model where you do not need it; plain LoRA is simpler.
- Forgetting `prepare_model_for_kbit_training`, which causes gradient instability.
- Mixing the wrong compute dtype (bf16 on GPUs that do not support it).

**Exercise**: On Colab free T4, load `Qwen/Qwen2.5-1.5B` with 4-bit quantization using the snippet above. Print the model's memory footprint with `model.get_memory_footprint() / 1e9` and confirm it is under 2GB.

## DPO

DPO stands for Direct Preference Optimization. Suppose you are teaching a kid to write thank-you cards. You could write a rubric ("be polite, mention the gift, keep it short"), or you could just show them two cards and say "this one is better than that one" and let them figure out why. DPO is the second approach for language models.

You give DPO a dataset of triplets: a prompt, a "chosen" response, and a "rejected" response. The training nudges the model to make the chosen one more likely and the rejected one less likely, relative to a frozen reference copy of itself. There is no separate reward model and no reinforcement learning loop. This makes DPO much simpler and more stable than older techniques like PPO, which is why it became the default for preference tuning in 2024.

```python
# pip install trl peft transformers datasets
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
ref_model = AutoModelForCausalLM.from_pretrained(model_id)

# Dataset must have columns: "prompt", "chosen", "rejected"
dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train[:1000]")

config = DPOConfig(
    output_dir="./dpo-tinyllama",
    per_device_train_batch_size=2,
    learning_rate=5e-6,
    num_train_epochs=1,
    beta=0.1,  # how strongly to prefer chosen over rejected
)

trainer = DPOTrainer(
    model=model, ref_model=ref_model,
    args=config, train_dataset=dataset,
    processing_class=tokenizer,
)
trainer.train()
```

**Summary**: DPO teaches a model from pairs of "better" and "worse" responses without needing a reward model or RL plumbing.

**Mental model**: "This card is nicer than that one" repeated thousands of times.

**Beginner mistakes**:
- Setting `beta` too high (above 0.5), which freezes the model in place.
- Skipping the supervised fine-tuning step first; DPO works best on a model that already speaks your format.
- Using noisy preference data; bad pairs poison the training signal.

**Exercise**: On Colab T4, run the DPO snippet on TinyLlama with the first 1000 examples of `trl-lib/ultrafeedback_binarized`. Compare generations from the base and tuned model on a few prompts to feel the difference.

## RLHF

RLHF stands for Reinforcement Learning from Human Feedback, and it is the technique that made ChatGPT feel like ChatGPT. Think of how a dog gets trained. You do not write the dog a rulebook. You give it a treat when it does something good, and slowly its behavior shifts. RLHF does the same to a language model, using a reward model as the "treat dispenser."

The classic RLHF pipeline has three stages. First, supervised fine-tuning on examples of good responses. Second, train a reward model on human preference rankings, so it learns to score any response. Third, use reinforcement learning (usually PPO) to push the language model toward responses the reward model likes. It is powerful but fiddly: the policy can collapse, the reward model can be gamed, and it eats GPUs for breakfast. That is why DPO has taken over for most use cases. But understanding RLHF is still important because frontier labs use variants of it.

```python
# pip install trl transformers accelerate
# Simplified PPO loop with TRL
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

policy = AutoModelForCausalLMWithValueHead.from_pretrained(model_id)
ref_policy = AutoModelForCausalLMWithValueHead.from_pretrained(model_id)

config = PPOConfig(
    learning_rate=1.4e-5,
    batch_size=8,
    mini_batch_size=2,
)

# In a real run you would also pass a reward_model and a value_model
# trained separately on human preference data.
# trainer = PPOTrainer(config, policy, ref_policy, tokenizer, ...)
# for batch in dataloader:
#     responses = trainer.generate(batch["input_ids"])
#     rewards = reward_model(responses)         # scalar per response
#     stats = trainer.step(batch["input_ids"], responses, rewards)
```

**Summary**: RLHF shapes a model's behavior by rewarding outputs that humans prefer, using a learned reward model and PPO.

**Mental model**: Training a dog with treats instead of writing it a rulebook.

**Beginner mistakes**:
- Trying RLHF before you have a solid SFT baseline; you will train chaos.
- Ignoring the KL penalty, which lets the policy drift into gibberish that games the reward.
- Using RLHF when DPO would do the job in a tenth of the code.

**Exercise**: Read TRL's PPO example at github.com/huggingface/trl. Sketch out, on paper, the data flow between the policy, reference policy, and reward model. Running PPO end-to-end realistically needs an A100; do not torture yourself on a T4 for a first attempt.

## Quantization

Quantization is taking numbers stored with lots of precision and rounding them to fewer bits. Picture a recipe that says "add 1.7382 grams of salt." For home cooking, "add 2 grams" is just as good and way easier to measure. Models normally store weights as 16-bit or 32-bit floating point numbers, but for inference you can often round them to 8-bit or even 4-bit integers and barely notice.

The payoff is huge. An 8B parameter model in fp16 takes 16GB of VRAM. The same model quantized to 4 bits takes 4GB and runs faster because moving bytes around is often the bottleneck, not the math. Common formats include bitsandbytes (good for training with QLoRA), GPTQ and AWQ (popular for GPU inference), and GGUF (the format for CPU/Mac inference via llama.cpp).

```python
# Load any HuggingFace model in 8-bit for inference
# pip install bitsandbytes transformers accelerate
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_8bit=True)

model_id = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)

prompt = "Quantization is useful because"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

**Summary**: Quantization shrinks model weights to fewer bits, slashing memory and often speeding up inference with minor quality loss.

**Mental model**: Rounding recipe measurements from "1.7382 grams" to "2 grams" so cooking is faster.

**Beginner mistakes**:
- Quantizing to 4-bit and complaining about quality on a tiny 1B model; small models suffer more from quantization than large ones.
- Mixing quantization formats; you cannot mostly load a GPTQ model with bitsandbytes.
- Forgetting that quantization is for inference and QLoRA-style training; pure training in 4-bit is research-grade.

**Exercise**: On any machine with a GPU (Colab T4 is fine), load `meta-llama/Llama-3.2-1B` in fp16, then in 8-bit, then in 4-bit. Print `model.get_memory_footprint()` for each and compare. Note that you need to accept the Llama license on HuggingFace first.

## Model checkpoints

A checkpoint is just a snapshot of the model's weights at a moment in time. Think of it like saving your progress in a video game. If you keep playing and lose all your gear in a boss fight, you can reload the save and try again with a different strategy. During training, you save checkpoints every few hundred steps so you can resume if your machine crashes, compare model versions, or roll back when something goes wrong.

Checkpoints usually include more than weights. They include the optimizer state (so training resumes smoothly), the learning rate scheduler position, the random seed state, and sometimes a piece of the data loader's position. Hugging Face's `Trainer` saves all of this for you in a folder. For LoRA fine-tuning, the checkpoint is tiny because only the adapter is saved, while the base model is referenced by name.

```python
# pip install transformers datasets accelerate
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling,
)
from datasets import load_dataset

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_id)

ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:200]")
def tok(b): return tokenizer(b["text"], truncation=True, max_length=128)
ds = ds.map(tok, batched=True, remove_columns=["text"])

args = TrainingArguments(
    output_dir="./checkpoints-tinyllama",
    save_steps=50,             # save every 50 steps
    save_total_limit=3,        # keep only the 3 most recent
    num_train_epochs=1,
    per_device_train_batch_size=2,
)

trainer = Trainer(
    model=model, args=args, train_dataset=ds,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
trainer.train()
# Resume later with: trainer.train(resume_from_checkpoint=True)
```

**Summary**: Checkpoints are saved snapshots of a model (and optimizer state) so you can resume training, roll back, or compare versions.

**Mental model**: Save files in a video game; load the one before the boss fight you lost.

**Beginner mistakes**:
- Saving every step and filling your disk; use `save_total_limit`.
- Saving only weights and losing the optimizer state, then wondering why resume training behaves oddly.
- Confusing a HuggingFace "checkpoint" folder with a single `.bin` file; it is a folder.

**Exercise**: Run the snippet on Colab T4, kill the runtime mid-training, then restart and resume with `trainer.train(resume_from_checkpoint=True)`. Confirm training picks up from the last saved step.

## Adapter tuning

Adapter tuning is the broader family that LoRA belongs to. The shared idea: do not touch the giant pre-trained model, just insert small trainable modules between its layers. Think of camera lenses. You own one expensive camera body and a wardrobe of lenses: wide angle for landscapes, macro for insects, telephoto for sports. Same body, swap the lens, totally different photos. Adapters work the same way for one base LLM.

Beyond LoRA there are other adapter styles: prefix tuning (prepends learnable vectors to every attention layer), prompt tuning (only trains a soft prompt at the input), and IA3 (rescales activations with tiny vectors). All of them share the same superpower: tiny files, fast training, and the ability to host hundreds of task-specific adapters on top of one base model in memory. This is how companies serve dozens of customer-tuned variants without paying for dozens of base models.

```python
# pip install peft transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import IA3Config, get_peft_model, TaskType

model_id = "Qwen/Qwen2.5-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

ia3_config = IA3Config(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["k_proj", "v_proj", "down_proj"],
    feedforward_modules=["down_proj"],
)

model = get_peft_model(model, ia3_config)
model.print_trainable_parameters()

# Save just the adapter (a few MB), not the whole base model
model.save_pretrained("./qwen-ia3-adapter")

# Later, reload base + adapter
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained(model_id)
tuned = PeftModel.from_pretrained(base, "./qwen-ia3-adapter")
```

**Summary**: Adapter tuning is a family of methods (LoRA, IA3, prefix tuning) that train tiny add-on modules while freezing the base model.

**Mental model**: One camera body, many lenses, swap as needed.

**Beginner mistakes**:
- Picking an exotic adapter type when LoRA would just work; LoRA is the safe default.
- Saving the merged model when you wanted to keep the adapter swappable.
- Forgetting to call `model.eval()` before generation, leaving dropout on.

**Exercise**: On Colab T4, train an IA3 adapter on Qwen2.5-1.5B with any small instruction dataset. Save just the adapter and confirm the saved folder is only a few MB, not gigabytes.

## GGUF models

GGUF is a file format invented by the llama.cpp project for storing quantized models that run on CPU, Mac Metal, or modest GPUs. Think of it as MP3 for language models. The original studio recording (the fp16 model on HuggingFace) is huge and pristine. The MP3 is smaller, plays on any device, and sounds nearly identical to most ears. GGUF packages the weights, the tokenizer, and the metadata into one file so you can ship a model as easily as sending a song.

GGUF really shines for local inference. With llama.cpp or Ollama or LM Studio, you can run a 7B model on a laptop with no GPU at all, just RAM. The format supports many quantization levels labeled like `Q4_K_M`, `Q5_K_S`, `Q8_0`. As a rough rule, `Q4_K_M` is the sweet spot for most users: about 4 bits per weight, very small quality loss, great speed. To create a GGUF you start from a HuggingFace model and convert it with llama.cpp's converter script, then quantize.

```bash
# Easiest path: pull a pre-made GGUF with Ollama (CPU-friendly)
ollama pull llama3.2:1b
ollama run llama3.2:1b "Explain GGUF in one sentence."

# Or convert and quantize yourself with llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# Convert a HuggingFace model to fp16 GGUF
python convert_hf_to_gguf.py /path/to/TinyLlama-1.1B-Chat-v1.0 \
    --outfile tinyllama-f16.gguf --outtype f16

# Quantize down to Q4_K_M (about 4 bits per weight)
./llama-quantize tinyllama-f16.gguf tinyllama-Q4_K_M.gguf Q4_K_M

# Run it on CPU
./llama-cli -m tinyllama-Q4_K_M.gguf -p "Hello, who are you?" -n 64
```

**Summary**: GGUF is a single-file quantized model format built for CPU and small-GPU inference via llama.cpp and Ollama.

**Mental model**: MP3 for language models, easy to share and play anywhere.

**Beginner mistakes**:
- Picking `Q2_K` to save space, then wondering why outputs feel dumb.
- Trying to fine-tune a GGUF file directly; GGUF is an inference format, train in HuggingFace then convert.
- Downloading a GGUF made for a different llama.cpp version and hitting load errors; keep tools current.

**Exercise**: Install Ollama on your laptop (no GPU needed), then run `ollama pull llama3.2:1b` and chat with it. For bonus points, convert TinyLlama to GGUF Q4_K_M with llama.cpp on your CPU and time the difference between Q4 and Q8 generations.

## What's next

You now have a working map of how modern fine-tuning happens, from parameter-efficient adapters to preference optimization and portable file formats. Next, head to `04-inference-optimization.md` to learn how to make these tuned models actually serve users fast.
