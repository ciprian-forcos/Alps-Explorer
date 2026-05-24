# Evaluation

You have learned to build with models, but how do you know if a model is actually good for your job? This chapter teaches you how to measure, compare, and choose models like a careful shopper rather than a wishful one.

## AI benchmarks

Think of AI benchmarks the way you think of standardized school tests. The SAT does not tell you who will be a great person, but it gives a rough idea of how a student handles certain problems compared to everyone else who took the same test. Benchmarks for language models work the same way: a fixed set of questions, a fixed way to grade them, and a single score you can compare across models.

The most common ones you will hear about are **MMLU** (Massive Multitask Language Understanding, a mix of school-style multiple-choice questions across 57 subjects), **GSM8K** (grade-school math word problems), **HumanEval** (small Python coding tasks where the model writes a function and gets graded by running unit tests), **IFEval** (does the model follow specific formatting instructions like "answer in exactly three bullet points"), and **BFCL** (Berkeley Function Calling Leaderboard, for testing tool use). For broader reports, **HELM** from Stanford aggregates many benchmarks into one big picture.

The standard tool to run these yourself is `lm-evaluation-harness` from EleutherAI. It is a command-line tool that loads a model, runs a benchmark, and prints a score.

```bash
# Install once
pip install lm-eval

# Run MMLU on a small local model via HuggingFace
lm_eval --model hf \
        --model_args pretrained=Qwen/Qwen2.5-0.5B \
        --tasks mmlu \
        --device cpu \
        --batch_size 1 \
        --limit 20    # only 20 questions for a quick smoke test
```

Drop `--limit` for a real run. Use `--tasks gsm8k,humaneval,ifeval` to chain several at once.

**Summary.** Benchmarks are fixed exams that give you a rough, comparable score across models.

**Mental model.** A benchmark is a standardized test: useful for ranking, terrible for predicting how someone will behave on a specific job.

**Beginner mistakes.**
- Trusting a leaderboard number blindly without checking if the benchmark resembles your real task.
- Comparing scores from different reports that used different prompts or different few-shot settings.
- Ignoring benchmark contamination (the model may have seen the test questions during training).

## Human evals

Benchmarks measure what a machine can grade. But many things that matter, like "is this answer pleasant to read" or "did it actually solve my problem", need a human in the loop. Human evaluation is like a food tasting panel: you give several judges the same dish from different chefs, hide the labels, and ask which one they prefer.

The most famous human eval today is **Chatbot Arena** (also called **LMSys Arena**). Real users type a prompt, see two anonymous model answers side by side, and vote. Over millions of votes, an Elo rating emerges, the same system used to rank chess players. Another well-known setup is **MT-Bench**, where a strong model like GPT-4 judges short multi-turn conversations, which is a cheap substitute for human judges.

You can run your own mini Arena at home with two local models and a third as judge. This pattern is called **LLM-as-judge**.

```python
import ollama  # pip install ollama; needs `ollama serve` running

PROMPT = "Explain why the sky is blue to a curious 8-year-old."

a = ollama.chat(model="llama3.2:3b",
                messages=[{"role": "user", "content": PROMPT}])["message"]["content"]
b = ollama.chat(model="qwen2.5:3b",
                messages=[{"role": "user", "content": PROMPT}])["message"]["content"]

judge_prompt = f"""You are a fair judge. Two assistants answered this question:

QUESTION: {PROMPT}

ANSWER A: {a}

ANSWER B: {b}

Pick the better answer. Reply with only "A" or "B" and a one-sentence reason."""

verdict = ollama.chat(model="llama3.1:8b",
                      messages=[{"role": "user", "content": judge_prompt}])
print(verdict["message"]["content"])
```

For real human evals, always randomize the order (A/B should not always be the same model) and hide which model produced which answer, otherwise the judge has a bias.

**Summary.** Human evals (or LLM-as-judge as a stand-in) measure quality on things benchmarks cannot capture, like style and helpfulness.

**Mental model.** A blind taste test: judges only know what is in their mouth, not whose kitchen it came from.

**Beginner mistakes.**
- Letting judges see which model is which (positional and brand bias creep in).
- Using only one judge; a single grader has moods and blind spots.
- Asking vague questions like "which is better" without defining the criteria (helpful? safe? short?).

## Cost-per-token analysis

Running a model costs money, and the unit you pay in is the **token**, roughly three quarters of an English word. Think of it like an electricity bill. You do not pay per appliance, you pay per kilowatt-hour. Likewise you do not pay per "question", you pay per token of input and output.

Before you call an API, you can count tokens locally with `tiktoken` (for OpenAI-family tokenizers) or with the model's own tokenizer. This lets you estimate the bill before you press send.

```python
# pip install tiktoken
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o")
text = "Explain why the sky is blue to a curious 8-year-old."
n_in = len(enc.encode(text))

# Suppose the answer comes back at about 200 tokens
n_out = 200

# Illustrative prices, in US dollars per 1 million tokens (check the vendor for current rates)
PRICE_IN  = 2.50
PRICE_OUT = 10.00

cost = (n_in / 1_000_000) * PRICE_IN + (n_out / 1_000_000) * PRICE_OUT
print(f"Input tokens: {n_in}, output tokens: {n_out}, estimated cost: ${cost:.6f}")
```

Approximate prices vary a lot. Here is a rough, illustrative table as of early 2025 (please verify with each provider before relying on it):

| Model                              | Where it runs   | Input ($ / 1M tok) | Output ($ / 1M tok) |
| ---------------------------------- | --------------- | ------------------ | ------------------- |
| GPT-4o                             | OpenAI API      | ~2.50              | ~10.00              |
| Claude Sonnet 3.5/4                | Anthropic API   | ~3.00              | ~15.00              |
| Llama 3.3 70B                      | Together AI     | ~0.88              | ~0.88               |
| Llama 3.2 3B (Ollama)              | Your laptop     | 0 (electricity)    | 0 (electricity)     |

These numbers are illustrative; treat them as ballpark, not gospel.

**Summary.** Cost scales with tokens in and tokens out, and different providers charge very different rates.

**Mental model.** An electricity bill in kWh: you pay for what you draw, not for the appliance itself.

**Beginner mistakes.**
- Forgetting that output tokens are usually 3 to 5 times more expensive than input tokens.
- Putting huge prompts on every call instead of caching the static parts.
- Ignoring the free option (a small local model) when the task is simple.

## Speed benchmarking

Cost tells you how much you pay; speed tells you how long users wait. Two numbers matter: **time to first token** (how soon the first character appears, like the moment a tap starts pouring water) and **tokens per second** after that (how fast the rest streams in).

A good way to feel this is to time a local model. Pick a prompt, ask the model to write something of roughly known length, and measure.

```python
import time, ollama

PROMPT = "Write a 200-word bedtime story about a friendly fox."

start = time.time()
first_token_time = None
total_tokens = 0

stream = ollama.chat(
    model="llama3.2:3b",
    messages=[{"role": "user", "content": PROMPT}],
    stream=True,
)

for chunk in stream:
    if first_token_time is None:
        first_token_time = time.time() - start
    piece = chunk["message"]["content"]
    total_tokens += max(1, len(piece.split()))  # rough word count as a token proxy

elapsed = time.time() - start
print(f"Time to first token: {first_token_time:.2f} s")
print(f"Total time:          {elapsed:.2f} s")
print(f"Approx tokens/sec:   {total_tokens / elapsed:.1f}")
```

For a more accurate token count, ask the model's tokenizer to count the final text instead of using word splitting.

**Summary.** Speed has two parts: how quickly the answer starts, and how fast it streams afterwards.

**Mental model.** Turning on a tap: time-to-first-token is the pause before water comes, tokens-per-second is the flow rate.

**Beginner mistakes.**
- Only measuring total time and missing that users care most about the first second.
- Benchmarking on a warm cache after many runs and reporting numbers that a real first-time user will never see.
- Comparing a tiny prompt to a huge one and pretending the tokens-per-second are comparable.

## Quality benchmarking

Quality is the hardest to measure because "good" depends on your job. The right way is to build a small **eval set of your own**: maybe 20 to 100 prompts that look like real user requests, with notes about what a good answer looks like. Then every time you change the prompt, the model, or the temperature, you re-run the eval set and compare.

You can mix three graders. A **rule-based** grader checks easy things like "did the JSON parse" or "did the answer contain the customer's name". A **reference-based** grader compares to a known good answer using string overlap or embedding similarity. And an **LLM-as-judge** grader, like in the human evals section, handles the fuzzy stuff. The MT-Bench paper showed that a strong judge model agrees with humans about 80 percent of the time, which is good enough for early iteration.

Public quality benchmarks you can borrow ideas from: **MT-Bench** for multi-turn chat, **Arena-Hard** for harder Arena-style prompts, **HELM** for a wide sweep, **IFEval** for instruction following, and **BFCL** for tool calling.

```python
# A 3-prompt quality eval that mixes a rule check and an LLM judge.
import ollama, json

EVAL = [
    {"prompt": "Return only the JSON {\"city\": \"Paris\"} and nothing else.",
     "rule": lambda s: json.loads(s.strip()) == {"city": "Paris"}},
    {"prompt": "What is 17 * 23?", "rule": lambda s: "391" in s},
    {"prompt": "Greet me politely in one sentence.", "rule": None},
]

for case in EVAL:
    out = ollama.chat(model="llama3.2:3b",
                      messages=[{"role": "user", "content": case["prompt"]}])
    answer = out["message"]["content"]
    if case["rule"]:
        try:
            ok = case["rule"](answer)
        except Exception:
            ok = False
        print("RULE", "PASS" if ok else "FAIL", "->", answer[:60])
    else:
        judge = ollama.chat(model="llama3.1:8b", messages=[{
            "role": "user",
            "content": f"Is this a polite one-sentence greeting? Reply yes or no.\n\n{answer}"
        }])["message"]["content"]
        print("JUDGE", judge.strip()[:30], "->", answer[:60])
```

**Summary.** True quality is measured on your own eval set, with a mix of rule checks and a judge model.

**Mental model.** A quality eval is a recipe taste test you run after every kitchen change; if today's batch tastes worse than yesterday's, you know something broke.

**Beginner mistakes.**
- Not having an eval set at all and judging by vibes after one or two tries.
- Optimizing for a metric that does not match what users want (for example, longest answer wins).
- Letting your eval set leak into your prompts so the model "learns the test".

## Exercise

Build a tiny eval harness in one Python file.

1. Pick two local Ollama models, for example `llama3.2:3b` and `qwen2.5:3b`. Make sure both are pulled.
2. Write a list of 10 prompts that look like real questions you would ask (mix of factual, creative, coding, and formatting tasks).
3. For each prompt and each model, record: the answer, the token count (use the model's tokenizer or `tiktoken` as an approximation), and the elapsed time in seconds.
4. Print a side-by-side markdown table with columns: prompt, model A answer (truncated), model B answer (truncated), tokens A, tokens B, latency A, latency B.
5. After the table, send each (prompt, answer A, answer B) triple to a third model as judge (for example `llama3.1:8b`) and ask it to pick A or B with a one-sentence reason. Randomize which model is "A" per prompt so the judge cannot cheat by position.
6. Print a final tally: how often A won, how often B won, average latency, and average tokens.

Bonus: add a `--save results.json` flag and reload across runs to track regressions when you change prompts.

## What's next

You now know how to measure cost, speed, and quality, and how to build your own eval harness. Next, in `11-real-world-skills.md`, we turn these measurements into shipping habits: how to monitor a live system, handle bad outputs, and keep your app trustworthy after it leaves your laptop.
