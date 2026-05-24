# Foundations of Modern LLM Engineering

Before you can build with large language models, you need a clear mental picture of what they are and how they behave. This chapter gives you the vocabulary and intuition you'll lean on for every other chapter in this curriculum.

## 1. LLM Basics

A Large Language Model (LLM) is a program that has read an enormous amount of text and learned to guess the next word. That's it. Everything fancy you've seen, chatbots, code assistants, summarizers, is built on top of that one trick: predict what comes next.

Think of an LLM like a very well-read friend who has skimmed millions of books, websites, and conversations. When you ask them a question, they don't "look up" the answer. They produce words one at a time, each word chosen because it sounds like the most natural continuation of what came before.

This matters because it explains both their magic and their flaws. They can write a poem in the style of Shakespeare because they've seen patterns of Shakespeare. They can also confidently invent a fact because "a confident-sounding fact" is often the most natural continuation.

```python
# A tiny taste using Hugging Face transformers
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")
print(generator("The best thing about mornings is", max_new_tokens=20)[0]["generated_text"])
```

**Summary**: An LLM is a next-word predictor trained on huge piles of text.

**Mental model**: A very fast autocomplete that has read the internet.

**Beginner mistakes**:
- Thinking the model "knows" facts the way a database does.
- Believing more confident output means more correct output.
- Assuming the model understands you the way a human does.

**Exercise**: Install `transformers` and run the snippet above with three different prompts. Notice how the same prompt can produce different outputs each time.

## 2. How AI Models Work

At their core, modern AI models are math. They take numbers in, multiply them by other numbers (called weights), and produce numbers out. The numbers going in represent your text. The numbers coming out represent probabilities for the next piece of text.

Picture a giant kitchen with millions of dials. Each dial nudges the recipe slightly. During training, the chef tastes the output and gently turns dials to make tomorrow's dish taste better. After billions of tastings, the dials are set so the kitchen reliably produces sensible sentences.

When you use the model (called inference), the dials are frozen. Your prompt walks through the kitchen, gets transformed at each station, and a recommendation pops out the other end.

**Summary**: AI models are layers of math whose internal "dials" were tuned by seeing tons of examples.

**Mental model**: A giant pinball machine where your prompt is the ball and the bumpers are fixed by training.

**Beginner mistakes**:
- Thinking the model "reasons" in the way you do; it computes.
- Confusing the training phase with the inference phase.
- Assuming bigger always means better for your use case.

**Exercise**: Run `pip install transformers torch` and load `gpt2`. Print `sum(p.numel() for p in model.parameters())` to see how many dials that model has.

## 3. Tokens

A token is the unit a model actually sees. It's not exactly a word and not exactly a letter, it's a chunk in between. Common short words like "the" are one token. Longer or rarer words get split: "tokenization" might become "token" + "ization".

Imagine you're packing a suitcase but you're only allowed to put in pre-made fabric squares. Common phrases get nice big squares. Weird words get cut into smaller scraps. The model only ever sees the squares, never your original words.

This matters for two practical reasons: cost (APIs charge per token) and limits (models have a maximum number of tokens they can handle at once).

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
ids = tok.encode("Tokenization is surprisingly tricky.")
print(ids)
print([tok.decode([i]) for i in ids])
```

**Summary**: Tokens are the bite-sized chunks of text the model reads and writes.

**Mental model**: Lego bricks for language, sometimes a whole word, sometimes just a syllable.

**Beginner mistakes**:
- Counting words instead of tokens when estimating cost.
- Assuming one token equals one word.
- Forgetting that whitespace and punctuation are part of tokens too.

**Exercise**: Tokenize the same sentence with `gpt2` and `Qwen/Qwen2.5-0.5B` tokenizers. Compare how many tokens each produces.

## 4. Tokenization

Tokenization is the recipe that turns your raw text into tokens. Different models use different recipes, so the same sentence might be 10 tokens for one model and 14 for another. Most modern LLMs use a method called Byte Pair Encoding (BPE) or a close cousin.

Think of tokenization like chopping vegetables before cooking. A chef who chops onions into perfect cubes will get a different stew than one who slices them into rings, even with the same recipe. Different tokenizers chop your text differently, and the model was trained to expect a specific style.

This is why you must always pair a model with its own tokenizer. Mixing them is like sending diced onions to a chef who was trained on rings, the math still runs, but the results get weird.

```python
from transformers import AutoTokenizer

text = "Don't forget: pre-training matters!"
for name in ["gpt2", "Qwen/Qwen2.5-0.5B"]:
    tok = AutoTokenizer.from_pretrained(name)
    print(name, "->", tok.tokenize(text))
```

**Summary**: Tokenization is the model-specific rulebook for slicing text into tokens.

**Mental model**: The vegetable chopper that has to match the chef.

**Beginner mistakes**:
- Using the wrong tokenizer for a model.
- Ignoring special tokens like end-of-sequence markers.
- Forgetting that emojis and non-English text often cost extra tokens.

**Exercise**: Tokenize a short paragraph of English, then the same paragraph in Japanese or Arabic if you can find one. Compare token counts.

## 5. Context Windows

The context window is the maximum number of tokens a model can hold in its head at once, prompt plus answer combined. GPT-2 had a 1,024-token window. Modern models stretch from 8K to over a million tokens.

Think of context as the model's short-term memory written on a whiteboard. Everything in the current conversation lives on that whiteboard. The moment you exceed its size, the oldest writing gets erased, and the model forgets it ever existed.

This is why long chatbot conversations sometimes "forget" things you said earlier, and why feeding a model a huge PDF can fail. Engineering around the window (summarizing, retrieving relevant chunks, chunking documents) is half of what LLM engineers do.

**Summary**: The context window is the model's working memory, measured in tokens.

**Mental model**: A whiteboard with a fixed size, when it's full, old writing gets wiped.

**Beginner mistakes**:
- Forgetting that the output tokens also count against the window.
- Assuming the model remembers past API calls (it doesn't, unless you resend them).
- Pasting a giant document and expecting perfect recall.

**Exercise**: Load `gpt2` and print `model.config.n_positions`. Then load `Qwen/Qwen2.5-0.5B` and check its `max_position_embeddings`. Note the difference.

## 6. Embeddings

An embedding is a list of numbers that represents the meaning of a piece of text. Words or sentences with similar meanings end up with similar lists. "Dog" and "puppy" land near each other. "Dog" and "spreadsheet" land far apart.

Imagine a giant map of meaning where every concept has GPS coordinates. "Paris" and "London" are close because both are European capitals. "Paris" and "baguette" are close in a different direction (food and France). Embeddings are those coordinates, just in hundreds or thousands of dimensions instead of two.

Embeddings power search, recommendation, and Retrieval Augmented Generation (RAG). You convert documents to embeddings, store them, and later find the closest ones to a user's question.

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
vecs = model.encode(["I love dogs", "Puppies are great", "Tax returns are due"])
def cos(a, b): return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
print("dog vs puppy :", cos(vecs[0], vecs[1]))
print("dog vs taxes :", cos(vecs[0], vecs[2]))
```

**Summary**: Embeddings turn text into numerical coordinates of meaning.

**Mental model**: A GPS for ideas, similar things share neighborhoods.

**Beginner mistakes**:
- Comparing embeddings from different models (the maps don't line up).
- Using raw distance instead of cosine similarity.
- Forgetting embeddings can encode bias along with meaning.

**Exercise**: Install `sentence-transformers` and run the snippet. Add five sentences of your own and find which two are most similar.

## 7. Transformers

The Transformer is the architecture, the blueprint, behind nearly every modern LLM. Introduced in 2017, it replaced older designs because it could process many tokens in parallel instead of one at a time, making training on huge data practical.

Picture a busy newsroom. Older models were like a single reporter reading every sentence in order before writing a summary. A Transformer is a whole room of reporters who each grab the full document, focus on the parts relevant to them, then combine their notes. Same job, much faster, and often better.

A Transformer is built from stacked layers, and each layer has two main parts: an attention mechanism (next section) and a small feed-forward network. Stack enough of these and you get GPT, Llama, Qwen, and friends.

**Summary**: Transformers are the parallel-processing blueprint that made modern LLMs possible.

**Mental model**: A newsroom of reporters who all read the document at once and pool their notes.

**Beginner mistakes**:
- Confusing the Transformer architecture with a specific model.
- Thinking "more layers" automatically means "smarter."
- Assuming all Transformers are decoder-only (some are encoder-only or both).

**Exercise**: Load `gpt2` with `from transformers import GPT2Model; m = GPT2Model.from_pretrained("gpt2")` and `print(m)`. Count the repeated blocks, those are your Transformer layers.

## 8. Attention Mechanism

Attention is how a Transformer decides which earlier tokens matter most when predicting the next one. For each new token, the model assigns a weight to every previous token, "this word is highly relevant, that one barely matters."

Imagine reading a mystery novel and reaching the sentence "She picked it up." Your brain instantly flips back to figure out what "it" refers to. You don't reread the whole book; you focus on the relevant earlier bits. Attention is the math version of that flip-back.

This is the secret sauce. Without attention, the model would treat all earlier words equally and produce mushy output. With attention, it can keep track of who did what, even across long passages.

**Summary**: Attention lets each token look back at the most relevant earlier tokens.

**Mental model**: Mental highlighter that picks out which earlier words matter right now.

**Beginner mistakes**:
- Thinking attention "understands" meaning; it just weighs relevance.
- Ignoring that attention cost grows roughly with the square of context length.
- Confusing self-attention (within one sequence) with cross-attention (between two).

**Exercise**: Run a `gpt2` forward pass with `output_attentions=True` and inspect the shape of one attention tensor. You'll see weights for every token pair.

## 9. Parameters

Parameters are the dials we mentioned earlier, the numbers the model learned during training. A model with "7B parameters" has roughly 7 billion of these numbers. Each one nudges the math a tiny bit.

Think of a parameter count like the number of knobs on a mixing board. A small radio has a few knobs and can shape simple sounds. A professional studio has thousands and can produce any track imaginable. More parameters means more capacity, but also more memory, more compute, and more cost.

That said, parameter count is not the whole story. A well-trained 8B model often beats a sloppy 70B model. Quality of training data and method matters as much as raw size.

| Model | Parameters | Runs on a laptop? |
|---|---|---|
| GPT-2 small | 124M | Yes, easily |
| TinyLlama | 1.1B | Yes |
| Phi-3-mini | 3.8B | Yes, with patience |
| Llama 3 8B | 8B | Yes, with quantization |
| Llama 3 70B | 70B | Not really |

**Summary**: Parameters are the learned numbers inside a model; more usually means more capable but also more expensive.

**Mental model**: Knobs on a mixing board, more knobs, more nuance, more cost.

**Beginner mistakes**:
- Treating parameter count as the only quality signal.
- Forgetting that bigger models need more RAM and VRAM.
- Ignoring quantization, which lets big models run on small machines.

**Exercise**: Pull `phi3:mini` with `ollama pull phi3:mini` (after installing Ollama) and run it. Then pull `tinyllama` and compare answer quality on the same question.

## 10. Training vs Inference

Training is when the model learns; inference is when you use it. Training takes weeks, thousands of GPUs, and millions of dollars for frontier models. Inference takes milliseconds on your laptop or a server.

Picture learning to ride a bike. Training is the wobbly weeks of falling, getting back up, and slowly tuning your balance. Inference is every smooth ride after that. The training was expensive and slow; each ride afterward is cheap and fast. You don't relearn how to ride every time you hop on.

As an LLM engineer, you'll spend almost all your time on inference: prompting, fine-tuning small models, building retrieval pipelines. Knowing the gap helps you understand why you can't "just teach" a model a new fact at runtime, you'd have to retrain or feed the fact in via context.

| Aspect | Training | Inference |
|---|---|---|
| When | Once (or occasionally) | Every user request |
| Cost | Huge | Small |
| Hardware | Clusters of GPUs | One GPU or CPU |
| Changes weights? | Yes | No |

**Summary**: Training builds the model; inference uses it. Almost all engineering happens at inference time.

**Mental model**: Training is learning to ride; inference is every ride after.

**Beginner mistakes**:
- Thinking you can "teach" a model new facts during a chat.
- Confusing fine-tuning (more training) with prompting (inference).
- Forgetting that inference cost adds up at scale.

**Exercise**: Time how long it takes `gpt2` to generate 50 tokens using `time.perf_counter()`. That's inference. Now look up how long GPT-2 took to train (days on many GPUs). Feel the gap.

## 11. Open-Source vs Closed-Source Models

Closed-source models (GPT-4, Claude, Gemini) live behind APIs. You send text, you get text back, you pay per token. You never see the weights. Open-source models (Llama, Qwen, Mistral, Phi) publish their weights so you can download them, run them locally, and modify them.

Think of closed-source as eating at a restaurant: convenient, consistent, but you can't change the recipe or take the kitchen home. Open-source is buying ingredients: more work, but full control over what ends up on the plate.

Both have a place. Closed models often lead on raw quality. Open models win on privacy, cost at scale, customization, and the ability to run offline. Most production systems end up using both.

| Aspect | Closed-source | Open-source |
|---|---|---|
| Access | API only | Download weights |
| Cost | Per token | Pay for hardware |
| Privacy | Data leaves your machine | Stays local |
| Customization | Limited fine-tuning | Full control |
| Examples | GPT-4, Claude, Gemini | Llama, Qwen, Mistral, Phi |

**Summary**: Closed models trade control for convenience; open models trade convenience for control.

**Mental model**: Restaurant versus home kitchen, both feed you, in very different ways.

**Beginner mistakes**:
- Assuming closed always beats open on every task.
- Underestimating the ops work of running open models in production.
- Sending sensitive data to a closed API without checking the data policy.

**Exercise**: Install Ollama, run `ollama pull qwen2.5:0.5b`, then `ollama run qwen2.5:0.5b "Explain embeddings in one sentence."` You're now running an open-source LLM entirely on your machine, no API key required.

## What's Next

You now have the vocabulary and mental models to think clearly about LLMs. Next, in `02-datasets-training.md`, we'll dig into where these models actually come from: the data they eat, the training loop that shapes them, and how you can train or fine-tune your own.
