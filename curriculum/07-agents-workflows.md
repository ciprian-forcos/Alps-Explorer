# Agents and Workflows

So far you have been talking to a model one message at a time. In this chapter we level up: we teach the model how to follow instructions, use tools, take actions, and even coordinate with other models to get real work done.

## Prompt engineering

Prompt engineering sounds fancy, but it is really just the craft of asking well. Think of it like ordering coffee. If you walk up and say "coffee," you might get black drip in a paper cup. If you say "a medium oat-milk latte, extra shot, not too hot," you get exactly what you wanted. The model is the barista. Your prompt is your order.

A good prompt usually has four parts: who the model should be, what task it should do, what context it needs, and what shape the answer should take. You do not need fancy words; you need clear ones. If your friend could follow the instructions, the model probably can too.

A common trick is to show examples. This is called few-shot prompting. Instead of describing the style you want, you demonstrate it. Two or three examples are usually enough to lock the model into a pattern, the same way watching a coworker do a task twice teaches you faster than reading a manual.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

prompt = """Classify the sentiment as positive, negative, or neutral.

Review: "The hike was breathtaking but the trail was muddy."
Sentiment: neutral

Review: "Best fondue I've ever had."
Sentiment: positive

Review: "The cable car was broken and the staff were rude."
Sentiment:"""

response = client.chat.completions.create(
    model="llama3.2",
    messages=[{"role": "user", "content": prompt}],
)
print(response.choices[0].message.content)
```

**Summary:** Prompt engineering is the skill of writing inputs that reliably produce the outputs you want.

**Mental model:** A prompt is a coffee order; vague orders get vague drinks.

**Beginner mistakes:**
- Writing one giant blob with no structure (use headings, bullets, examples).
- Asking for the answer before giving the context.
- Forgetting to say what format you want back (JSON? a list? one sentence?).

**Exercise:** Write three versions of a prompt that asks the model to summarize a news article. Version 1 is one line. Version 2 adds a role and a word limit. Version 3 adds two example summaries. Compare the outputs.

## System prompts

If a user prompt is the question, a system prompt is the job description. Imagine hiring someone for a help desk. Before any customer calls, you sit them down and explain: "You work for Alps Explorer. Be friendly, always recommend safe trails, never give medical advice." That briefing is the system prompt. Every user message after that is filtered through it.

System prompts shape personality, scope, and rules. They are sticky: the model keeps re-reading them on every turn. A weak system prompt is like a manager who says "just do good work" and walks away. A strong one tells the model who it is, what it can and cannot do, and how to respond when things go off-script.

Keep system prompts focused. Long, contradictory system prompts confuse the model just like a confused job description confuses a new hire. If you find yourself writing rule number 47, your prompt is too complicated.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

system_prompt = """You are a Swiss Alps trail guide for the Alps Explorer app.
Rules:
- Recommend hikes only in Switzerland, Austria, or Italy.
- Always mention difficulty (easy, moderate, hard) and approximate time.
- If asked about something outside hiking, politely redirect to trails.
- Reply in under 80 words."""

response = client.chat.completions.create(
    model="llama3.2",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "I have one free afternoon near Interlaken. What should I do?"},
    ],
)
print(response.choices[0].message.content)
```

**Summary:** A system prompt is a persistent job description that shapes every reply the model gives.

**Mental model:** System prompt is the employee handbook; user messages are the daily tickets.

**Beginner mistakes:**
- Putting task-specific data in the system prompt instead of the user message.
- Writing contradictory rules ("be concise but explain everything").
- Forgetting to test what happens when users ask things outside the scope.

**Exercise:** Take the trail-guide system prompt above and add one new rule. Then ask the model something off-topic, like a math question, and see if it sticks to the rules.

## Tool calling

A language model alone cannot check the weather, query a database, or send an email. It only knows how to predict text. Tool calling is how we hand it a calculator. You describe a tool ("here is a function called get_weather, it takes a city name and returns the forecast") and the model decides when to use it.

Picture a worker at a desk. They are smart but they only have a pen. You hand them a calculator, a phone, and a filing cabinet. They still do the thinking, but now they can also do things. The model does not run the tool itself; it tells you "please call get_weather with city=Zermatt" and waits for you to bring back the answer. You feed the result back, and it continues.

This is the bridge between language models and the real world. Without it, a model can only chat. With it, the model can plan trips, book flights, and update spreadsheets, as long as you give it the right tools.

```python
from openai import OpenAI
import json

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city in the Alps.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. Zermatt"},
            },
            "required": ["city"],
        },
    },
}]

def get_weather(city):
    fake_db = {"Zermatt": "sunny, 12C", "Innsbruck": "cloudy, 8C"}
    return fake_db.get(city, "unknown")

messages = [{"role": "user", "content": "What's the weather in Zermatt right now?"}]
response = client.chat.completions.create(
    model="llama3.2", messages=messages, tools=tools, tool_choice="auto"
)
call = response.choices[0].message.tool_calls[0]
args = json.loads(call.function.arguments)
print("Model wants to call:", call.function.name, args)
print("Result:", get_weather(args["city"]))
```

**Summary:** Tool calling lets the model ask your code to do things it cannot do on its own, like fetch data or run actions.

**Mental model:** The model is a smart desk worker; tools are the calculator and phone on that desk.

**Beginner mistakes:**
- Writing tool descriptions that are too vague ("does stuff with weather").
- Forgetting the model does not run the tool; you do, then send the result back.
- Giving the model 20 tools at once and wondering why it picks the wrong one.

**Exercise:** Add a second tool called convert_celsius_to_fahrenheit and re-run the example. Ask the model: "How warm is Zermatt in Fahrenheit?" See if it chains both tools.

## Function calling

You will hear "function calling" and "tool calling" used as if they were the same thing, and most of the time they are. Originally, OpenAI shipped a feature called function calling: you described one or more functions, the model picked one. Later they renamed and expanded it to tools, because tools can be more than just functions (think file search, code interpreters, web browsers). Under the hood the idea is identical: a structured request from the model to call something external.

So when a tutorial uses functions= and tool_choice="auto" together, it is mixing old and new wording. New code should use tools= and tools entries that look like {"type": "function", "function": {...}}. The behavior is the same; the wrapping is just slightly different. Ollama and many other open-source servers accept both styles for compatibility.

Why does the distinction matter? Because as you grow, you will see "tools" become an umbrella term. A tool might be a function you wrote, but it might also be a built-in capability like web search or code execution. Function calling is one kind of tool calling.

```python
from openai import OpenAI
import json

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# New "tools" style (preferred). Functionally equivalent to the old "functions" parameter.
tools = [{
    "type": "function",
    "function": {
        "name": "add_numbers",
        "description": "Add two numbers and return the sum.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["a", "b"],
        },
    },
}]

response = client.chat.completions.create(
    model="llama3.2",
    messages=[{"role": "user", "content": "What is 17 plus 25?"}],
    tools=tools,
    tool_choice="auto",
)
call = response.choices[0].message.tool_calls[0]
args = json.loads(call.function.arguments)
print(args["a"] + args["b"])
```

**Summary:** Function calling and tool calling describe the same mechanism; "tools" is the modern, broader name.

**Mental model:** Function calling is the original recipe; tool calling is the same recipe with more ingredients allowed.

**Beginner mistakes:**
- Assuming they are different features and learning both from scratch.
- Mixing old `functions=` with new `tools=` in the same request.
- Forgetting that the JSON schema is what teaches the model how to call your code.

**Exercise:** Rewrite the weather example from the previous section using the deprecated `functions=` and `function_call="auto"` arguments (if your client still supports them). Compare the response shape with the modern `tools=` version.

## AI agents

A plain chat model answers, then stops. An agent keeps going. It thinks, picks a tool, observes the result, thinks again, and only stops when the goal is met. Picture an intern you sent to plan a weekend trip. They do not come back after one Google search. They check weather, find trains, compare hotels, and only then hand you the itinerary.

That loop, often written as think -> act -> observe -> think, is the heart of every agent. The model is the brain. Tools are the hands. Memory is the notebook. A loop in your code keeps feeding outputs back as inputs until the model says "I am done."

The simplest agent is a while loop. The fanciest one uses a framework like LangGraph, CrewAI, or the Claude Agent SDK. They all do the same dance underneath. Start simple. Once you understand the loop by hand, the frameworks will feel like shortcuts rather than magic.

```python
from openai import OpenAI
import json

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

def get_distance(city_a, city_b):
    table = {("Zurich", "Lucerne"): 52, ("Lucerne", "Interlaken"): 68}
    return table.get((city_a, city_b), 100)

tools = [{
    "type": "function",
    "function": {
        "name": "get_distance",
        "description": "Distance in km between two Swiss cities.",
        "parameters": {
            "type": "object",
            "properties": {"city_a": {"type": "string"}, "city_b": {"type": "string"}},
            "required": ["city_a", "city_b"],
        },
    },
}]

messages = [{"role": "user", "content": "How far is it from Zurich to Lucerne, then Lucerne to Interlaken? Total?"}]

for step in range(5):
    resp = client.chat.completions.create(model="llama3.2", messages=messages, tools=tools)
    msg = resp.choices[0].message
    messages.append(msg)
    if not msg.tool_calls:
        print("Final:", msg.content)
        break
    for call in msg.tool_calls:
        args = json.loads(call.function.arguments)
        result = get_distance(**args)
        messages.append({"role": "tool", "tool_call_id": call.id, "content": str(result)})
```

**Summary:** An AI agent is a model wrapped in a loop that lets it use tools repeatedly until a goal is reached.

**Mental model:** An agent is an intern with a notebook who keeps working until the task is checked off.

**Beginner mistakes:**
- No stop condition; the loop runs forever or until you go broke on tokens.
- Forgetting to append the tool result back into messages.
- Giving the agent vague goals like "do the thing" and hoping for the best.

**Exercise:** Take the loop above and add a hard limit of 5 steps plus a "give up gracefully" message. Then add a second tool, get_train_time, and ask a question that needs both.

## Agentic workflows

An agent decides what to do next on its own. A workflow tells it what to do in what order. Most real systems are a mix. Think of a restaurant kitchen. A single chef who handles every order from scratch is an agent. A kitchen with stations (prep, grill, plating) where each step is fixed is a workflow. Reality is usually a workflow with one creative chef on the line.

Workflows are predictable, cheaper, and easier to debug. Agents are flexible but slower and harder to test. A good rule: if the steps are mostly the same every time, use a workflow. If the steps depend heavily on what the user wants, lean agentic. Most production systems start as workflows and only become agentic where they truly need to.

LangGraph, n8n, and similar tools let you draw the workflow as a graph: nodes are steps, edges are decisions. Even without a framework, a series of well-named Python functions chained together is a workflow.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

def ask(system, user):
    r = client.chat.completions.create(
        model="llama3.2",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return r.choices[0].message.content

def workflow(topic):
    outline = ask("You write short outlines.", f"Outline a blog post about {topic} in 3 bullets.")
    draft = ask("You write friendly blog posts.", f"Expand this outline into 120 words:\n{outline}")
    title = ask("You write catchy titles.", f"Give one title for this post:\n{draft}")
    return title, draft

t, d = workflow("hiking the Eiger trail")
print(t, "\n\n", d)
```

**Summary:** A workflow is a fixed sequence of model calls; an agentic workflow lets the model deviate when needed.

**Mental model:** A workflow is the recipe; an agent is the chef who can improvise when the milk runs out.

**Beginner mistakes:**
- Going full agent when a 3-step workflow would do.
- Hiding the workflow logic inside one giant prompt.
- Not logging intermediate steps, so you cannot debug failures.

**Exercise:** Extend the workflow above with a fourth step that fact-checks the draft against a list of allowed claims. If the check fails, loop back and rewrite the draft. Now you have an agentic workflow.

## Multi-agent systems

Sometimes one agent is not enough. You want one to research, one to write, one to review. This is a multi-agent system. It is the same idea as a team at work: a manager hands tasks to specialists, each specialist focuses on their craft, and someone pulls the results together.

The trick is communication. Agents talk by passing messages, often through a shared memory or a coordinator. Frameworks like CrewAI, AutoGen, and LangGraph give you ready-made patterns: hierarchies, debates, round-robins. Without a framework, you can build it with plain Python: each agent is a function that takes a message and returns one.

Multi-agent systems shine when tasks are clearly separable. They struggle when agents talk past each other or get stuck in loops of "no, you do it." Start with two agents. Make sure the handoff is clean. Only add a third when the second can no longer do its job alone.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

def agent(role, task):
    r = client.chat.completions.create(
        model="llama3.2",
        messages=[
            {"role": "system", "content": f"You are a {role}. Be brief."},
            {"role": "user", "content": task},
        ],
    )
    return r.choices[0].message.content

researcher = lambda q: agent("travel researcher who lists 3 facts", q)
writer = lambda facts: agent("travel writer who turns facts into a paragraph", facts)

facts = researcher("Quick facts about Lake Bled, Slovenia.")
paragraph = writer(facts)
print(paragraph)
```

**Summary:** A multi-agent system splits a big task across specialized agents that pass results between each other.

**Mental model:** Multi-agent systems are a small team; pick your teammates carefully or meetings never end.

**Beginner mistakes:**
- Creating five agents when two would do.
- Letting agents chat freely without a coordinator, so they loop forever.
- Not defining what a "handoff" looks like, so output formats mismatch.

**Exercise:** Add a third agent, an editor, that takes the writer's paragraph and trims it to under 50 words. Pass the output along the chain: researcher -> writer -> editor.

## Browser agents

A browser agent is an agent whose tools control a web browser. It can click, type, scroll, and read pages. This unlocks tasks that have no API: filling out forms, scraping dashboards, booking tickets on stubborn websites. Think of it as giving your intern a laptop and saying "go figure it out on the web."

The two common approaches are Playwright with custom tool wrappers, and higher-level libraries like browser-use that already wire the model into the browser. Anthropic's "computer use" is a related idea but for the whole desktop, not just the browser. All of them follow the same pattern: take a screenshot or read the DOM, ask the model what to do next, execute the click, repeat.

Browser agents are powerful and fragile. Websites change, captchas appear, layouts shift. Treat them like a smart but easily-distracted intern: give clear goals, narrow the task, and always supervise the first few runs.

```python
# pip install playwright && playwright install chromium
from playwright.sync_api import sync_playwright

def fetch_title(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url)
        title = page.title()
        browser.close()
        return title

# Pretend an agent decided to call this tool.
print(fetch_title("https://example.com"))
# Next step: feed the title back into the model and let it decide the next action.
```

**Summary:** Browser agents drive a real browser so the model can act on websites that have no clean API.

**Mental model:** A browser agent is an intern at a laptop, clicking around for you while you do other work.

**Beginner mistakes:**
- Running them on production accounts before testing in a sandbox.
- Ignoring robots.txt and rate limits.
- Forgetting that pages change; selectors that worked yesterday may break today.

**Exercise:** Build the full progression. Start with a strong system prompt for an Alps trip planner. Add one tool that calls fetch_title on a URL. Wrap it in a 3-step agent loop. Then add a second agent (an editor) that polishes the planner's output. By the end you will have touched every concept in this chapter.

## What's next

Next up: [08-model-types.md](08-model-types.md) covers the different families of models you will meet, from tiny on-device models to giant frontier ones, and how to pick the right tool for the job.
