# Deployment

You have a model that works on your laptop. Now you want other people, other apps, or your phone to use it. This chapter walks through the main places a model can live, from your own machine all the way to a tiny computer strapped to a robot.

## Local inference

Running a model locally is like cooking dinner in your own kitchen. You control the ingredients, the heat, and the timing. Nothing leaves the house, no one charges you per meal, and if the stove breaks it is your problem to fix. For small to mid-size models on a decent laptop, this is the cheapest and most private way to play.

The easiest path today is Ollama. It packages a model, a runtime, and a tiny HTTP server into one tool. You start the server once and it listens on port 11434 for chat requests, very much like a kitchen always ready to take an order from anyone in the house.

Start the server and pull a small model.

```bash
# in one terminal
ollama serve

# in another terminal
ollama pull llama3.2:1b
```

Now talk to it over HTTP like any other web API.

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "llama3.2:1b",
  "messages": [{"role": "user", "content": "Say hi in one short sentence."}],
  "stream": false
}'
```

A rough sense of trade-offs:

| Setup | Cost | Latency | Control |
|---|---|---|---|
| Local CPU | free | slow (1-5 tok/s on a 7B) | full |
| Local GPU (consumer) | free after hardware | fast (30-80 tok/s on a 7B) | full |
| Local Mac with MLX | free | very fast on Apple silicon | full |

**Summary.** Local inference means the model runs on your own hardware and answers requests through a local port. It is private, free per request, and ideal for development and personal tools.

**Mental model.** Your laptop becomes a tiny private ChatGPT that never phones home.

**Beginner mistakes.**
- Trying to run a 70B model on 16 GB of RAM and wondering why the laptop freezes.
- Forgetting that `ollama serve` must be running before any curl call works.
- Leaving the port open on a shared network without any auth.

## On-device AI

On-device AI is the same idea as local inference, but the device is something you carry or wear. Picture a microwave that already knows how to defrost chicken without phoning the manufacturer. The intelligence is baked into the appliance.

Phones and tablets now ship with small models built for this. Apple uses Core ML and increasingly the Apple Intelligence stack. Google ships Gemma Nano inside Android via AICore. Cross-platform projects like MLC LLM compile models down to run on the GPU of almost any phone. On a Mac, Apple's MLX framework lets you run mid-size models directly on the unified memory of M-series chips.

A taste of MLX on a Mac.

```bash
pip install mlx-lm
python -c "from mlx_lm import load, generate; \
  model, tok = load('mlx-community/Llama-3.2-1B-Instruct-4bit'); \
  print(generate(model, tok, prompt='Write one line about the Alps.', max_tokens=40))"
```

The constraints on a phone are real: limited RAM, a battery to protect, and a thermal budget. So models are heavily quantized (often 4-bit) and usually small (1B to 4B parameters). In exchange you get instant responses, full offline use, and no per-request bill.

| Device | Typical model size | Why pick it |
|---|---|---|
| iPhone / Android | 1B-4B quantized | offline, private, free |
| Mac M-series (MLX) | 7B-30B | great speed, no GPU needed |
| Laptop without GPU | 1B-3B | still useful for small tasks |

**Summary.** On-device AI ships the model with the app, so inference runs on the user's own phone or laptop with no network call.

**Mental model.** The brain rides in the pocket, not in a faraway data center.

**Beginner mistakes.**
- Picking a 13B model for a phone and shipping an app that drains the battery in ten minutes.
- Forgetting that model weights count against app download size.
- Not testing on a cold device; the first inference after a long pause is much slower.

## API serving

Sooner or later you want other programs, on other machines, to use your model. That is API serving: turning the model into a restaurant. Customers sit down, place an order through a menu (the API schema), the kitchen prepares it, and a waiter brings it back. The customer never sees the stove.

The simplest pattern is to wrap a local backend (Ollama, vLLM, llama.cpp) with FastAPI. Below is a complete working server you can run on the same machine where Ollama is already running.

```python
# server.py
# pip install fastapi uvicorn requests
from fastapi import FastAPI
from pydantic import BaseModel
import requests

app = FastAPI()
OLLAMA = "http://localhost:11434/api/chat"

class ChatIn(BaseModel):
    prompt: str
    model: str = "llama3.2:1b"

@app.post("/chat")
def chat(body: ChatIn):
    r = requests.post(OLLAMA, json={
        "model": body.model,
        "messages": [{"role": "user", "content": body.prompt}],
        "stream": False,
    }, timeout=120)
    r.raise_for_status()
    return {"reply": r.json()["message"]["content"]}

# run with: uvicorn server:app --host 0.0.0.0 --port 8000
```

For high throughput, vLLM gives you an OpenAI-compatible server in one line. It batches requests automatically, which is the single biggest reason production deployments use it.

```bash
pip install vllm
vllm serve meta-llama/Llama-3.2-1B-Instruct --port 8000
```

**Summary.** API serving exposes your model over HTTP so any client, language, or device can use it through a stable contract.

**Mental model.** You are the chef opening a restaurant; design the menu carefully because changing it later breaks every customer.

**Beginner mistakes.**
- No rate limiting; one buggy client can hog the whole GPU.
- No request timeouts; a slow generation blocks the worker forever.
- Hardcoding the model name in the URL instead of as a parameter, forcing redeploys for every swap.

## Cloud GPUs

Some workloads outgrow your laptop. Maybe you need an 80 GB GPU for a 70B model, or twenty GPUs for a Friday night traffic spike. Buying that hardware is silly. Renting it by the hour is the move. Think of it as a community industrial oven you book when you need to bake a wedding cake; the rest of the time someone else is using it.

The market has several flavors, and the right pick depends on whether you want bare metal, a serverless feel, or a hosted endpoint someone else manages.

| Provider | Style | Good for |
|---|---|---|
| Runpod | rent a pod, pay per minute | cheap, hands-on, full SSH |
| Modal | serverless Python, scale to zero | "just run my function on a GPU" |
| Lambda | reserved GPU instances | longer training jobs |
| Together | hosted open models behind an API | drop-in chat completions |
| Replicate | hosted models with versioned APIs | demos and prototypes |
| Hugging Face Inference Endpoints | one-click model to URL | teams that already live on HF |

Modal is a friendly entry point because it feels like writing a normal Python script that happens to run on a beefy GPU. Notice how you ask for hardware right in the decorator.

```python
# app.py
# pip install modal && modal token new
import modal

image = modal.Image.debian_slim().pip_install("transformers", "torch", "accelerate")
app = modal.App("hello-gpu")

@app.function(gpu="A10G", image=image, timeout=600)
def generate(prompt: str) -> str:
    from transformers import pipeline
    pipe = pipeline("text-generation", model="gpt2", device=0)
    return pipe(prompt, max_new_tokens=40)[0]["generated_text"]

@app.local_entrypoint()
def main():
    print(generate.remote("The Alps are"))
```

Run it with `modal run app.py`. Modal spins up an A10G, runs your code, returns the result, and shuts the GPU back down. You pay only for the seconds it ran.

On Runpod the same idea uses a template: pick a community image (for example "vLLM 0.6 + Llama 3"), pick a GPU (A10, A100, H100), and you get a pod with an HTTP endpoint in under a minute.

**Summary.** Cloud GPUs let you rent the right amount of compute for the right amount of time, instead of owning hardware that sits idle.

**Mental model.** You are renting the industrial oven by the hour, not buying the bakery.

**Beginner mistakes.**
- Spinning up an A100 to serve a 1B model that runs fine on a $0.20/hr T4.
- Forgetting about cold starts; serverless GPUs can take 10-60 seconds to warm.
- Leaving a pod running overnight by accident; always set an idle timeout.

## Edge AI basics

Edge AI is what happens when the model has to live somewhere weird: a drone, a doorbell, a tractor, a Raspberry Pi in a barn with flaky internet. It is the food truck of deployment. Small kitchen, limited menu, but it shows up exactly where the customers are and does not need a building.

The two friendly entry points are NVIDIA Jetson boards (Orin Nano, Orin NX) and the Raspberry Pi 5. Jetson has a real GPU and is built for vision models and small LLMs. The Pi 5 has no GPU but its CPU is fast enough to run a quantized 1B or 3B model through llama.cpp at a few tokens per second, which is plenty for many automation tasks.

A minimal recipe for a Pi 5 (or any Linux box):

```bash
# build llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make -j

# grab a tiny quantized model
curl -L -o qwen.gguf \
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf

# run it
./llama-cli -m qwen.gguf -p "In one sentence, what is an edge device?" -n 60
```

Edge constraints push you toward small models, aggressive quantization (Q4 or even Q2), and very narrow tasks. A doorbell does not need to write poems; it needs to say "person" or "package" in 100 ms.

| Edge device | Compute | Realistic model |
|---|---|---|
| Raspberry Pi 5 | CPU only | 0.5B-3B GGUF, Q4 |
| Jetson Orin Nano | 8 GB GPU | 7B GGUF or small vision models |
| Jetson Orin NX | 16 GB GPU | 7B-13B quantized |

**Summary.** Edge AI runs small, quantized models on cheap hardware close to where the data is generated, trading raw power for latency, privacy, and offline reliability.

**Mental model.** Your model is a food truck: small kitchen, fast service, parked right next to the customer.

**Beginner mistakes.**
- Picking a 7B model for a Pi 5 and getting one token every two seconds.
- Forgetting that the device may have no internet, so model weights must be shipped with the firmware.
- Ignoring thermals; a Jetson under sustained load needs a fan or it will throttle hard.

## Exercise

Wrap a local Ollama model in a FastAPI endpoint and call it from a separate script.

1. Make sure `ollama serve` is running and you have pulled `llama3.2:1b`.
2. Save the `server.py` from the API serving section above and start it with `uvicorn server:app --port 8000`.
3. In a new file `client.py`, write a tiny client and run it with `python client.py`:

```python
# client.py
import requests

r = requests.post("http://localhost:8000/chat", json={
    "prompt": "Name three mountains in the Alps in one short sentence."
})
print(r.json()["reply"])
```

Once it works, try changing the `model` field in the request to a different model you have pulled, and add a `max_tokens` parameter that you forward to Ollama's `options`.

## What's next

You can now put a model in front of real users. The next question is: is it any good? Head to `10-evaluation.md` to learn how to measure that.
