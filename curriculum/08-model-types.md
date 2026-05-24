# Model Types

Not every language model is built for the same job. In this chapter we will walk through six common types you will meet in the wild, what makes each one different, and how to actually run one on your own machine.

## VLMs (Vision-Language Models)

A regular language model is like a friend on the phone: they can only hear what you say. A Vision-Language Model (VLM) is like that same friend on a video call: now you can hold up a receipt, a photo of a plant, or a screenshot of an error message, and they can talk about what they see.

VLMs combine an image encoder (the "eyes") with a language model (the "mouth"). When you send them a picture plus a question, they first turn the image into a long list of numbers that describe shapes, colors, and objects, then feed those numbers into the language part so it can answer in plain English.

Popular VLMs include LLaVA, Qwen2-VL, and the proprietary GPT-4o. They are great for screenshot question-answering, describing photos for accessibility, reading handwritten notes, or understanding charts.

Here is a tiny example using Ollama with LLaVA, after you run `ollama pull llava`:

```python
import ollama

response = ollama.chat(
    model="llava",
    messages=[
        {
            "role": "user",
            "content": "What is in this picture? Be brief.",
            "images": ["./cat.jpg"],
        }
    ],
)
print(response["message"]["content"])
```

Or with Hugging Face transformers and Qwen2-VL:

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image

model_id = "Qwen/Qwen2-VL-2B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

image = Image.open("cat.jpg")
messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "Describe this image in one sentence."},
    ]}
]
text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=64)
print(processor.batch_decode(out, skip_special_tokens=True)[0])
```

**Summary:** VLMs accept images and text together and answer in text, so you can ask questions about pictures.

**Mental model:** A friend who can both hear you and see what you are pointing at.

**Beginner mistakes:**
- Sending tiny or blurry images and expecting perfect reading of small text.
- Forgetting that most VLMs still output only text, not new images.
- Using a VLM for pure text tasks where a smaller text-only model would be faster and cheaper.

## SLMs (Small Language Models)

If big models like GPT-4 are a fully equipped restaurant kitchen with a giant chef's knife, then Small Language Models (SLMs) are the pocket knife you keep on your keychain. They cannot do every job, but they are quick, cheap, and live in your pocket.

An SLM usually has somewhere between 0.5 and 4 billion parameters. That sounds huge, but it is tiny compared to the hundreds of billions inside frontier models. The win is that they can run on a laptop, a phone, or a Raspberry Pi without needing a giant GPU.

SLMs are perfect for narrow jobs: classifying support tickets, extracting fields from text, rewriting an email, or powering a local assistant that respects your privacy. They struggle with deep reasoning or world knowledge, but for focused tasks they often punch well above their weight.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

messages = [{"role": "user", "content": "Rewrite this in friendly tone: 'send report now'."}]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok(prompt, return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=80)
print(tok.decode(out[0], skip_special_tokens=True))
```

Other good SLMs to try: `microsoft/Phi-3-mini-4k-instruct` and `meta-llama/Llama-3.2-1B`.

| Model | Parameters | Good for |
|---|---|---|
| Qwen2.5-0.5B | 0.5B | Quick classification, rewrites |
| Llama-3.2-1B | 1B | On-device chat |
| Phi-3-mini | 3.8B | Light reasoning, summaries |

**Summary:** SLMs are small enough to run on everyday hardware and are great for focused, narrow jobs.

**Mental model:** A pocket knife: not for every task, but always with you.

**Beginner mistakes:**
- Expecting a 0.5B model to solve hard math or write long essays.
- Skipping fine-tuning when you have a very specific task an SLM could nail.
- Assuming "small" means "bad" instead of "specialized".

## Dense models

A dense model is the default kind of language model. "Dense" means that for every word you generate, every single parameter inside the model is used. Imagine a meeting where every employee in the company has to speak before any decision is made. That is thorough, but it is also slow and expensive.

Models like `Qwen/Qwen2.5-7B`, Llama 3 8B, and Mistral 7B are all dense. When the model is 7 billion parameters, all 7 billion are doing work on every token. That gives consistent quality but means cost scales directly with size.

Dense models are simple to reason about: more parameters usually means smarter answers, but also means more memory and more compute. If you want predictable behavior and clean fine-tuning, dense is the safe bet.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-7B"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

prompt = "Explain why the sky is blue in two sentences."
inputs = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=120)
print(tok.decode(out[0], skip_special_tokens=True))
```

**Summary:** Dense models use all of their parameters for every token, giving steady quality at a steady cost.

**Mental model:** A meeting where the whole company shows up for every decision.

**Beginner mistakes:**
- Thinking bigger dense models are always better; sometimes a small one is enough.
- Loading a 7B model in full precision on an 8 GB GPU and being surprised it crashes.
- Mixing up parameters (storage) with active parameters (compute).

## MoE models (Mixture of Experts)

Mixture of Experts (MoE) is what you get when you take a dense model and turn it into a team of specialists. Instead of every employee speaking at every meeting, a router picks the two or three people who actually know the topic, and only they speak. The rest stay quiet but are still on payroll.

A famous example is `mistralai/Mixtral-8x7B-Instruct-v0.1`. The name "8x7B" is misleading. It does **not** mean 8 times 7 = 56 billion active parameters. It means there are 8 expert blocks of about 7B each, but for every token only 2 experts are activated. The total stored size is about 47B parameters, but the active compute per token is closer to 13B. So you get the knowledge breadth of a big model with the speed of a smaller one.

The trade-off: MoE models need a lot of RAM to load all the experts, even though only some run at once. They are wonderful when you have memory but limited compute, and they often beat similarly-priced dense models on quality.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

messages = [{"role": "user", "content": "Give me three startup ideas for cyclists."}]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=200)
print(tok.decode(out[0], skip_special_tokens=True))
```

| Model | Total params | Active per token | Style |
|---|---|---|---|
| Qwen2.5-7B (dense) | 7B | 7B | All employees speak |
| Mixtral-8x7B (MoE) | ~47B | ~13B | Router picks 2 of 8 experts |

**Summary:** MoE models store many experts but only run a few per token, so you get big-model smarts at small-model compute cost.

**Mental model:** A consulting firm where only the relevant experts walk into the room.

**Beginner mistakes:**
- Reading "8x7B" as 56B active parameters; it is not.
- Forgetting MoE needs RAM for all experts even if only a few run.
- Trying to fine-tune MoE the same way as dense; the router adds extra care.

## Coding models

A general-purpose model is a smart friend who has read a lot. A coding model is that same friend after they spent two years working as a senior developer. They have seen thousands of pull requests, know your framework's quirks, and can read a stack trace at a glance.

Coding models are language models that were pre-trained or fine-tuned heavily on source code, commits, issues, and documentation. They are tuned to handle indentation, brackets, imports, and multi-file context. Popular examples are `Qwen/Qwen2.5-Coder-7B-Instruct`, `bigcode/starcoder2-3b`, and DeepSeek-Coder.

They shine at writing functions from a description, explaining unfamiliar code, suggesting fixes for bugs, and converting between languages. For chat about your weekend plans, though, they often feel a bit dry; that is by design.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-Coder-7B-Instruct"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

messages = [
    {"role": "user", "content": "Write a Python function that checks if a string is a palindrome. Ignore case and spaces."}
]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=200)
print(tok.decode(out[0], skip_special_tokens=True))
```

For a lighter option, `bigcode/starcoder2-3b` runs on a single modest GPU and is great for autocomplete-style tasks.

**Summary:** Coding models are language models specialized for programming tasks through heavy training on code.

**Mental model:** A senior engineer who has reviewed thousands of pull requests.

**Beginner mistakes:**
- Using a coding model for general chat and finding the tone too terse.
- Forgetting that they still hallucinate APIs; always run the code.
- Ignoring the instruct variant when you want a chat-style helper.

## Reasoning models

A regular model answers like a confident friend at a pub quiz: it blurts out the first plausible thing. A reasoning model is more like a careful student who shows their work on the side of the page before circling the final answer. That extra "thinking out loud" leads to much better results on math, logic, and multi-step planning.

Examples include OpenAI's o1 series, DeepSeek-R1, and the smaller distilled versions like `deepseek-r1:7b`. They were trained to produce a long internal chain of thought, often wrapped in special `<think>...</think>` tags, before the final user-visible answer. You usually see only the answer, but under the hood the model spent many tokens reasoning.

The cost: reasoning models are slower and use more tokens because they think before speaking. The benefit: on hard problems they often beat much larger non-reasoning models.

Here is how to talk to DeepSeek-R1-Distill via Ollama, after you run `ollama pull deepseek-r1`:

```python
import ollama

response = ollama.chat(
    model="deepseek-r1",
    messages=[
        {"role": "user", "content": "If a train leaves at 3pm going 60 km/h and another at 4pm going 90 km/h on the same track, when do they meet?"}
    ],
)
text = response["message"]["content"]
print(text)

# The output usually contains a <think>...</think> block followed by the final answer.
# You can split it like this:
if "</think>" in text:
    thoughts, answer = text.split("</think>", 1)
    print("\n--- Final answer only ---")
    print(answer.strip())
```

| Style | Speed | Token cost | Best for |
|---|---|---|---|
| Regular model | Fast | Low | Chat, summarization, simple Q&A |
| Reasoning model | Slow | High | Math, logic, planning, code debugging |

**Summary:** Reasoning models think step by step before answering, trading speed for accuracy on hard problems.

**Mental model:** A student who shows their work in the margin before circling the answer.

**Beginner mistakes:**
- Using a reasoning model for simple chat and wondering why it is slow.
- Showing users the raw `<think>` block instead of stripping it out.
- Setting a low max-token limit; reasoning needs room to breathe.

## Exercise

Install Ollama from https://ollama.com, then pull and run one model of each type:

```bash
ollama pull llava           # VLM
ollama pull phi3            # SLM
ollama pull qwen2.5-coder   # Coding model
ollama pull deepseek-r1     # Reasoning model
```

For each one, write a short Python script using the `ollama` package that sends a prompt suited to that model type (an image for `llava`, a quick rewrite for `phi3`, a function to write for `qwen2.5-coder`, a multi-step puzzle for `deepseek-r1`) and prints the response. Notice the differences in speed, output style, and token count. Bonus: time each call with `time.perf_counter()` and put the numbers in a table.

## What's next

In the next chapter, [`09-deployment.md`](09-deployment.md), we will take these models out of your notebook and learn how to deploy them as services that real apps can call.
