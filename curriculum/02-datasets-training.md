# Datasets and Training for LLMs

Before a language model can be helpful, somebody has to feed it the right examples. This guide walks you through the kinds of datasets used to teach LLMs, how people clean and shape that data, and the basic ideas behind fine-tuning so you can start experimenting on your own laptop.

## SFT datasets

SFT stands for Supervised Fine-Tuning. Think of it like a cooking class where the teacher shows the student a dish and the exact recipe to make it. The student watches, copies, and slowly learns the pattern. An SFT dataset is just a big collection of those "here is a question, here is the ideal answer" pairs.

The base model already knows grammar and a lot of facts from pretraining, but it does not know how to behave like an assistant. SFT teaches it manners and format. After SFT, the model knows that when somebody asks "How do I boil an egg?", it should produce a clear, step-by-step answer instead of just continuing the sentence like a Wikipedia article.

A single SFT row is usually just an instruction and a target response. Here is what one row from `tatsu-lab/alpaca` looks like in JSON:

```json
{
  "instruction": "Give three tips for staying healthy.",
  "input": "",
  "output": "1. Eat a balanced diet rich in fruits and vegetables.\n2. Exercise regularly, at least 30 minutes a day.\n3. Get 7-9 hours of sleep every night."
}
```

The model is trained to predict the `output` token by token, given the `instruction` (and optional `input`) as context. Loss is only computed on the output part, so the model is not penalized for the prompt it was shown.

**Summary**: SFT datasets are prompt-response pairs that teach a base model how to act like a helpful assistant.

**Mental model**: A flashcard deck where the front is a question and the back is the perfect answer.

**Beginner mistakes**:
- Mixing low-quality scraped text with curated answers; the model will copy the worst behavior.
- Forgetting that the model imitates style, not just content. Sloppy answers produce a sloppy assistant.
- Using huge datasets without checking duplicates or near-duplicates.

**Exercise**: Load 1000 rows of Alpaca and look at the shape of the data.

```python
from datasets import load_dataset

ds = load_dataset("tatsu-lab/alpaca", split="train[:1000]")
print(ds)
print(ds[0])
print("Average output length:", sum(len(r["output"]) for r in ds) / len(ds))
```

## Instruction tuning

Instruction tuning is the broader idea behind SFT: take a model and teach it to follow instructions written in natural language. Instead of training it on raw web text, you train it on a wide variety of tasks all framed as "do this thing, here is the result."

Imagine a new employee on their first day. They are smart, but they have no idea what your company does or how you like things done. Instruction tuning is the onboarding handbook. It shows them: "When a customer asks for a refund, respond like this. When somebody reports a bug, do that." After enough examples across enough situations, the employee generalizes and handles new requests they have never seen before.

The trick is variety. A model trained only on summarization will only get good at summarization. A model trained on summarization, translation, classification, creative writing, coding, and Q&A learns the meta-skill of "read the instruction and respond appropriately." Datasets like `HuggingFaceH4/ultrachat_200k` and FLAN are popular because they cover hundreds of task types.

Modern instruction-tuned models also expect a specific chat template so the model knows who said what. A typical multi-turn example looks like:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
    {"role": "user", "content": "And of Germany?"},
    {"role": "assistant", "content": "The capital of Germany is Berlin."}
  ]
}
```

**Summary**: Instruction tuning teaches a model to follow natural-language instructions across many task types using a uniform chat format.

**Mental model**: Employee onboarding with a handbook full of "if a customer says X, you respond Y" examples.

**Beginner mistakes**:
- Training only on one task type and expecting general assistant behavior.
- Ignoring the chat template the base model expects, leading to weird role confusion.
- Using only English data when you want the model to work in other languages.

**Exercise**: Inspect UltraChat and count messages per conversation.

```python
from datasets import load_dataset

ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:500]")
print(ds.column_names)
print("Turns in first example:", len(ds[0]["messages"]))
for msg in ds[0]["messages"][:4]:
    print(msg["role"], "->", msg["content"][:80])
```

## Preference datasets

A preference dataset does not show the model the one perfect answer. Instead, it shows the model two answers to the same prompt and says "this one is better, that one is worse." This is the fuel for techniques like RLHF and DPO that align a model with human taste.

Think of two cover letters written for the same job. Both are technically correct, but a recruiter would prefer one because it is more focused and less generic. You cannot easily write a single "ideal" cover letter, but you can usually point at two and say which one is stronger. Preference data captures exactly this comparative judgment.

A standard preference row, in the format used by `argilla/distilabel-intel-orca-dpo-pairs` and most DPO trainers, looks like:

```json
{
  "prompt": "Explain why the sky is blue to a six-year-old.",
  "chosen": "The sky looks blue because sunlight is made of many colors, and when it hits the air, the blue color bounces around more than the others. That bouncing blue is what fills the sky and reaches your eyes.",
  "rejected": "Rayleigh scattering causes shorter wavelengths of visible light to scatter more strongly in the atmosphere, producing the observed blue hue of the diurnal sky."
}
```

Both answers are accurate, but the `chosen` one matches the audience. The model learns to shift probability toward responses like `chosen` and away from responses like `rejected`. Notice you are teaching a direction, not an absolute target.

**Summary**: Preference datasets pair a prompt with a better and a worse response so models can learn human taste, not just correctness.

**Mental model**: A taste test where you are not told the right answer, only "this one beats that one."

**Beginner mistakes**:
- Confusing "rejected" with "wrong." It usually just means "less preferred."
- Building preference pairs where chosen and rejected differ in length only, which teaches the model to be verbose.
- Forgetting to keep the same prompt for both responses.

**Exercise**: Browse a real DPO dataset and check for length bias.

```python
from datasets import load_dataset

ds = load_dataset("argilla/distilabel-intel-orca-dpo-pairs", split="train[:200]")
print(ds.column_names)
row = ds[0]
print("PROMPT:", row["prompt"][:120])
print("CHOSEN len:", len(row["chosen"]))
print("REJECTED len:", len(row["rejected"]))

avg_chosen = sum(len(r["chosen"]) for r in ds) / len(ds)
avg_rejected = sum(len(r["rejected"]) for r in ds) / len(ds)
print(f"avg chosen={avg_chosen:.0f}  avg rejected={avg_rejected:.0f}")
```

## Synthetic datasets

Synthetic data is data generated by another model instead of written by humans. It sounds suspicious at first, like learning to cook from a robot, but in practice it has become one of the most important sources of training data because humans are slow and expensive.

The classic pattern: take a strong existing model (a "teacher"), prompt it with thousands of varied seed questions, and store its answers. You now have a fresh SFT dataset built in hours instead of months. Alpaca was famously made this way using GPT-3 outputs from 175 seed tasks. Tools like `distilabel` automate this pipeline.

The risk is amplification. If the teacher model has a quirk (always starts with "Certainly!"), the student will inherit it and exaggerate it. Worse, factual errors from the teacher get baked in. Good synthetic pipelines therefore mix teachers, add filtering steps, and often have humans spot-check samples.

A simple generation loop looks like:

```python
seed_tasks = [
    "Write a haiku about rain.",
    "Explain recursion in two sentences.",
    "Suggest a name for a coffee shop near a library.",
]

synthetic_rows = []
for task in seed_tasks:
    # Replace this call with any LLM API you have access to.
    response = call_teacher_model(task)
    synthetic_rows.append({"instruction": task, "output": response})
```

**Summary**: Synthetic datasets are generated by a strong model to cheaply produce training data for a smaller or new model.

**Mental model**: A student copying notes from the smartest kid in class, then learning from those notes.

**Beginner mistakes**:
- Using a single teacher model and inheriting all its biases and tics.
- Skipping the filter step and keeping obvious teacher hallucinations.
- Generating millions of rows that are all subtle paraphrases of each other.

**Exercise**: Look at how Alpaca (a real synthetic dataset) was structured.

```python
from datasets import load_dataset

ds = load_dataset("tatsu-lab/alpaca", split="train[:5000]")
# Count how many rows start with the same first 5 words (a cheap diversity check).
from collections import Counter
prefixes = Counter(" ".join(r["output"].split()[:5]) for r in ds)
print("Top repeated openings:")
for prefix, n in prefixes.most_common(10):
    print(f"{n:5d}  {prefix}")
```

## Data curation

Curation is the act of choosing what goes into your dataset and in what proportion. It is the most underrated step in training. Two teams can use the exact same model architecture and one ends up with a much better assistant simply because they curated better.

A music playlist is a good analogy. You could dump every song ever recorded onto a USB stick, but nobody would enjoy it. A great playlist is shorter, has variety, no duplicates, and a deliberate mix of moods. Dataset curation is the same: pick high-quality sources, balance the topic mix, and remove anything that drags the average down.

Common curation decisions: how much code vs prose, how many languages, what fraction of multi-turn vs single-turn conversations, how much reasoning data, what safety examples to include. Each ratio shifts the personality of the final model.

A simple curated mix might look like:

```python
from datasets import load_dataset, concatenate_datasets

chat = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:2000]")
alpaca = load_dataset("tatsu-lab/alpaca", split="train[:2000]")

# Normalize to a shared schema before mixing.
def to_messages(row, source):
    if source == "alpaca":
        prompt = row["instruction"]
        if row["input"]:
            prompt += "\n\n" + row["input"]
        return {"messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": row["output"]},
        ]}
    return {"messages": row["messages"]}

alpaca_norm = alpaca.map(lambda r: to_messages(r, "alpaca"), remove_columns=alpaca.column_names)
chat_norm = chat.map(lambda r: to_messages(r, "chat"), remove_columns=chat.column_names)

mixed = concatenate_datasets([alpaca_norm, chat_norm]).shuffle(seed=42)
print(len(mixed), "rows after curation")
```

**Summary**: Curation is choosing the right ingredients and proportions for your training mix; it often matters more than the algorithm.

**Mental model**: Building a great playlist instead of dumping every song you own onto one drive.

**Beginner mistakes**:
- Adding "more data" without checking if it actually improves the model.
- Letting one huge dataset dominate the mix and drown out smaller, higher-quality sources.
- Never measuring the result of a mix change, so you cannot tell what helped.

**Exercise**: Build a 4000-row mixed dataset by combining slices of two real datasets and confirm the final length and column schema match.

## Dataset cleaning

Cleaning is the unglamorous chore of removing garbage from your dataset. Empty rows, exact duplicates, broken Unicode, HTML leftovers, PII like emails and phone numbers, prompts in the wrong language, answers that are just an apology. Each one of these, multiplied by thousands of rows, teaches the model bad habits.

Picture a fridge clean-out before a dinner party. You toss the expired milk, scrape off the moldy bits, and throw away the mystery containers. What is left is safe to cook with. Dataset cleaning is the same energy applied to text.

A common rule of thumb: cleaning often removes 10 to 50 percent of a raw dataset, and the model gets better after the cut, not worse. Less but cleaner data beats more but messier data almost every time.

Here is a small cleaning pipeline:

```python
from datasets import load_dataset
import re

ds = load_dataset("tatsu-lab/alpaca", split="train")

def is_good(row):
    out = row["output"].strip()
    if len(out) < 10:
        return False
    if out.lower().startswith(("i'm sorry", "as an ai", "i cannot")):
        return False
    if re.search(r"\b[\w.-]+@[\w.-]+\.\w+\b", out):  # email-looking string
        return False
    return True

clean = ds.filter(is_good)

# Drop exact duplicates by (instruction, output) pair.
seen = set()
def dedupe(row):
    key = (row["instruction"], row["output"])
    if key in seen:
        return False
    seen.add(key)
    return True

clean = clean.filter(dedupe)
print(f"Before: {len(ds)}  After: {len(clean)}")
```

**Summary**: Cleaning removes empties, duplicates, PII, and refusals so the model is not trained on the worst parts of your data.

**Mental model**: Throwing out the moldy food before you cook dinner.

**Beginner mistakes**:
- Skipping deduplication; near-identical rows make the model memorize instead of generalize.
- Leaving in lazy refusals so the model learns to refuse easy questions.
- Forgetting that cleaning order matters; dedupe before expensive filters to save time.

**Exercise**: Run the snippet above on Alpaca and report how many rows it removed and which filter dropped the most.

## Dataset formatting

Even a clean, well-curated dataset is useless if the model cannot parse the format. Every model family expects a specific way of marking who is speaking. Llama, Mistral, Qwen, and ChatGPT-style models all have slightly different special tokens around turns. Getting this wrong silently ruins training: the model will train on tokens that never appear at inference time.

Imagine writing a letter in a language the recipient does not read. The words are perfect, the meaning is clear to you, but the recipient sees gibberish. The chat template is that language. The Hugging Face `tokenizer.apply_chat_template` method handles this for you.

Here is what a formatted training string actually looks like for a Llama-style chat model:

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

What is 2+2?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

2 + 2 equals 4.<|eot_id|>
```

You almost never type those tokens by hand. You write `messages` as a list of dicts and let the tokenizer expand them:

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "2 + 2 equals 4."},
]
text = tok.apply_chat_template(messages, tokenize=False)
print(text)
```

**Summary**: Formatting wraps your messages in the exact special tokens the model expects so training matches inference.

**Mental model**: Putting your message in the right envelope so the post office actually delivers it.

**Beginner mistakes**:
- Hardcoding `<|user|>` style tokens instead of using `apply_chat_template`.
- Mixing two different chat templates inside one training run.
- Forgetting `add_generation_prompt=True` when generating, which silently breaks output.

**Exercise**: Take one UltraChat conversation, run it through two different tokenizers (a Llama and a Mistral instruct model), and diff the resulting strings.

## Fine-tuning basics

Fine-tuning means taking a pretrained model and nudging its weights using a smaller, focused dataset. You are not teaching it the language from scratch; you are specializing it. With modern adapter methods like LoRA, you can fine-tune a 7B-parameter model on a single consumer GPU.

A pretrained model is like a talented general-purpose chef who knows every cuisine at a basic level. Fine-tuning is sending them to a two-week intensive at a sushi restaurant. They keep all their old skills, but now they make great sushi. LoRA in particular is like teaching them only the new sushi-specific moves without retraining their knife grip from scratch.

The minimum ingredients are: a base model, a formatted dataset, a tokenizer, a loss function (cross-entropy on the assistant tokens), and an optimizer. The TRL library wraps all of this into an `SFTTrainer`. A tiny example:

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

model_id = "HuggingFaceTB/SmolLM2-135M"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
ds = load_dataset("tatsu-lab/alpaca", split="train[:200]")

def fmt(row):
    prompt = row["instruction"] + ("\n" + row["input"] if row["input"] else "")
    return {"text": f"### Instruction:\n{prompt}\n\n### Response:\n{row['output']}"}

ds = ds.map(fmt)

cfg = SFTConfig(output_dir="out", num_train_epochs=1, per_device_train_batch_size=2)
trainer = SFTTrainer(model=model, tokenizer=tok, train_dataset=ds, args=cfg)
trainer.train()
```

Key knobs to know: learning rate (usually 1e-5 to 5e-5 for full fine-tune, 1e-4 to 3e-4 for LoRA), batch size, number of epochs (1 to 3 is typical), and sequence length.

**Summary**: Fine-tuning specializes a pretrained model on a focused dataset; LoRA makes it cheap enough for a laptop or single GPU.

**Mental model**: Sending a generalist chef to a two-week sushi intensive instead of culinary school all over again.

**Beginner mistakes**:
- Setting the learning rate too high and "destroying" the base model's prior knowledge.
- Training for too many epochs on small data, which causes memorization.
- Skipping evaluation; you have no idea if you improved anything.

**Exercise**: Run the snippet above on the SmolLM2-135M model with 200 Alpaca rows. It should finish on CPU in a few minutes. Inspect the loss curve in the trainer logs.

## Continued pretraining

Continued pretraining (sometimes called domain-adaptive pretraining) is different from fine-tuning. Instead of teaching the model to follow instructions, you keep feeding it raw text from a specialized domain to expand its base knowledge. Then you fine-tune on top.

Think of a doctor who already finished medical school and now spends six months reading nothing but cardiology journals. They are still a doctor, they did not lose general medicine, but their cardiology knowledge is now much deeper. Continued pretraining does that for an LLM: more legal documents to make a legal model, more code to make a code model, more Japanese text to make a model better in Japanese.

The data format is dead simple. No instructions, no chat template. Just plain text chunks, often concatenated and split into fixed-length sequences. The training objective is the same next-token prediction used in the original pretraining.

```python
from datasets import load_dataset
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")

# Pretend this is a domain-specific corpus (medical, legal, code).
corpus = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1000]")

def tokenize(batch):
    return tok(batch["text"], truncation=True, max_length=512)

tokenized = corpus.map(tokenize, batched=True, remove_columns=["text"])
print("Ready for continued pretraining:", tokenized)
```

The order matters: continued pretraining first, then SFT, then (optionally) preference tuning. Skipping the SFT step after continued pretraining usually gives you a model that knows more facts but cannot follow instructions.

**Summary**: Continued pretraining feeds more raw text to expand domain knowledge before any instruction tuning happens.

**Mental model**: A doctor doing a six-month deep dive in one specialty after medical school.

**Beginner mistakes**:
- Treating continued pretraining as a replacement for SFT; it is a complement.
- Using a tiny corpus that the model basically memorizes.
- Forgetting that catastrophic forgetting can happen if you train too aggressively on a narrow domain.

**Exercise**: Load 1000 lines of wikitext, tokenize with a model's tokenizer, and report the average tokens per row. This is the same data shape you would use for real continued pretraining.

## Hallucination reduction

A hallucination is when the model produces something that sounds confident but is wrong. The data side has a lot of influence here. Models hallucinate more when training data rewards confident-sounding answers regardless of truth, or when the model has been trained to never say "I do not know."

A confident tour guide who makes up dates because they hate admitting ignorance is a classic example. Tourists do not catch the errors in the moment, but the guide is still misleading them. We reduce hallucinations by training the guide to say "I am not sure, let me check" when they really do not know.

Concrete data techniques that help: include examples where the assistant correctly refuses or admits uncertainty, train on grounded responses that quote provided context, penalize hallucinations in preference data by marking confident-but-wrong answers as `rejected`. Retrieval-augmented generation also helps at inference time, but the data still has to teach the model how to use retrieved context.

A small preference row that targets hallucination:

```json
{
  "prompt": "Who won the 2027 World Series?",
  "chosen": "I do not have reliable information about a 2027 World Series. If it has not happened yet from your perspective, no winner exists. Please check a recent source to confirm.",
  "rejected": "The 2027 World Series was won by the New York Yankees in seven games against the Los Angeles Dodgers."
}
```

The `rejected` answer is fluent and specific. That is exactly what makes it dangerous, and exactly why it must be marked as worse.

**Summary**: Hallucinations drop when training data teaches the model to say "I do not know," to ground answers in context, and when confident-but-wrong responses are explicitly down-ranked.

**Mental model**: A tour guide trained to say "let me check" instead of inventing dates on the spot.

**Beginner mistakes**:
- Filtering out every "I do not know" example, which teaches the model to always pretend it knows.
- Rewarding length and confidence in preference data, which encourages fluent fabrication.
- Assuming RAG alone fixes hallucinations without any data work to teach grounded answering.

**Exercise**: Build five hand-written DPO rows in the JSON format above where the `chosen` answer admits uncertainty and the `rejected` answer fabricates confidently. Save them to a `.jsonl` file and load with `datasets.load_dataset("json", data_files="my_uncertainty.jsonl")`.

## What's next

You now know what kinds of data feed an LLM, how to clean and shape it, and the basic moving parts of fine-tuning. In `03-fine-tuning.md` we go deeper into the actual training loop: LoRA, QLoRA, learning rate schedules, evaluation, and shipping a fine-tuned model.
