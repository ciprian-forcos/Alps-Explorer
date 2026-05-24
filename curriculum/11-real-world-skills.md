# 11. Real-World Skills

This is the capstone. You have learned the pieces; now we wire them together into things people actually use. Each section below picks one common shape of AI product, shows the moving parts, and ends with a small project you can ship.

## Building chatbots

A chatbot is a tireless receptionist. It never gets bored, it remembers what you said two minutes ago (if you let it), and it can answer the same question a thousand times without rolling its eyes. The receptionist is not the doctor and not the surgeon, but they are the front door, and a good front door makes the whole building feel friendly.

The simplest useful chatbot has three moving parts: a place to type, a model to think, and a list to remember the conversation. That is it. Frameworks like Streamlit or Gradio give you the typing box for free, Ollama gives you a local model, and a Python list gives you the memory.

Here is a minimal chat loop using Streamlit and a local Ollama model. Save it as `chat.py`, run `streamlit run chat.py`, and you have a working chatbot.

```python
import streamlit as st
import ollama

st.title("My First Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask me anything"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = ollama.chat(
            model="llama3.2",
            messages=st.session_state.messages,
        )
        reply = response["message"]["content"]
        st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
```

Notice how the entire message history is sent every turn. That is the memory. If you drop the list, the bot becomes a goldfish.

**Summary:** A chatbot is a UI plus a model plus a message list. Streamlit and Ollama can give you a working one in twenty lines.

**Mental model:** A receptionist with a notebook that they re-read before every sentence.

**Beginner mistakes:**
- Building a chatbot when a search bar or a form would serve users better.
- Forgetting to pass message history, so the bot has no memory.
- No system prompt, so the bot has no personality or guardrails.

**Exercise:** Build a CLI or Streamlit chatbot over your own notes folder using Ollama. Add a system prompt that gives the bot a clear job (for example, "You are my study buddy for Spanish vocabulary"). Use it daily for a week and iterate on the system prompt.

## Building AI copilots

If a chatbot is a receptionist, a copilot is a pair-programmer sitting next to you. The difference is context and action. A receptionist waits at the desk; a copilot can see your screen, knows what file you opened, and can change things for you when you ask.

A copilot has two superpowers a plain chatbot lacks: it gathers context from your environment (open files, current selection, your codebase, your calendar) and it calls tools (run code, edit a file, send an email). The chat box is still there, but most of the magic happens before and after the model sees your message.

Here is the shape of a copilot loop. It is not full code, it is the pattern you will see in Cursor, Claude Code, and GitHub Copilot Chat.

```
1. User types a request in chat.
2. Copilot gathers context:
     - currently open file
     - selected text
     - recent terminal output
     - relevant files from the repo (via search or embeddings)
3. Copilot builds a prompt:
     [system: "You can call tools: read_file, write_file, run_shell"]
     [context: gathered files + selection]
     [user: their request]
4. Model responds with either text OR a tool call.
5. If tool call: run it, append the result, loop back to step 4.
6. If text: show to user, wait for next message.
```

In Python with the Anthropic SDK, a tool call cycle looks roughly like this.

```python
from anthropic import Anthropic

client = Anthropic()

tools = [{
    "name": "read_file",
    "description": "Read a file from disk",
    "input_schema": {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
}]

messages = [{"role": "user", "content": "Summarize README.md"}]

while True:
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        tools=tools,
        messages=messages,
    )
    if response.stop_reason == "tool_use":
        tool_use = next(b for b in response.content if b.type == "tool_use")
        with open(tool_use.input["path"]) as f:
            result = f.read()
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": [{
            "type": "tool_result",
            "tool_use_id": tool_use.id,
            "content": result,
        }]})
    else:
        print(response.content[0].text)
        break
```

**Summary:** A copilot equals chat plus context gathering plus tool calling, in a loop.

**Mental model:** A pair-programmer who can read your screen and press your keys, but only when you nod.

**Beginner mistakes:**
- Dumping the whole codebase into context instead of retrieving what is relevant.
- Giving the model fifty tools when three would do.
- Letting the loop run forever with no max iterations or human checkpoint.

**Exercise:** Extend your chatbot from the previous exercise. Add one tool that reads files from a specific folder, and one tool that writes a new note. Now your bot can answer "summarize my notes from this week" and "save this idea as a new note."

## AI automation

Automation is the boring, beautiful cousin of chatbots. There is no chat window, no user typing. A schedule fires, a script runs, an LLM does the smart bit in the middle, and a result lands somewhere useful, like your inbox or a Slack channel.

Think of a small office where every morning at 8am, an assistant reads yesterday's mail, highlights the urgent ones, and leaves a one-page summary on your desk. That assistant does not need a UI. They need a clock, a stack of mail, a pen, and somewhere to put the summary.

Here is a cron-style script that summarizes today's unread emails. Schedule it with `cron`, `launchd`, or a GitHub Actions cron trigger.

```python
import imaplib
import email
from email.header import decode_header
import ollama
import datetime

HOST = "imap.gmail.com"
USER = "you@example.com"
PASS = "your-app-password"

def fetch_today_unread():
    mail = imaplib.IMAP4_SSL(HOST)
    mail.login(USER, PASS)
    mail.select("inbox")
    today = datetime.date.today().strftime("%d-%b-%Y")
    _, data = mail.search(None, f'(UNSEEN SINCE {today})')
    bodies = []
    for num in data[0].split():
        _, msg_data = mail.fetch(num, "(RFC822)")
        msg = email.message_from_bytes(msg_data[0][1])
        subject = decode_header(msg["Subject"])[0][0]
        if isinstance(subject, bytes):
            subject = subject.decode(errors="ignore")
        bodies.append(f"Subject: {subject}\nFrom: {msg['From']}\n")
    mail.logout()
    return "\n---\n".join(bodies)

def summarize(text):
    response = ollama.chat(
        model="llama3.2",
        messages=[
            {"role": "system", "content": "Summarize emails into 5 bullets. Flag anything urgent."},
            {"role": "user", "content": text},
        ],
    )
    return response["message"]["content"]

if __name__ == "__main__":
    emails = fetch_today_unread()
    if emails.strip():
        summary = summarize(emails)
        with open("daily-summary.md", "w") as f:
            f.write(f"# {datetime.date.today()}\n\n{summary}\n")
        print("Summary written.")
    else:
        print("No new mail.")
```

The structure is always: trigger, fetch, think, deliver. Swap any layer freely.

**Summary:** Automation is a script on a schedule with an LLM in the middle. No UI, just useful output landing somewhere you check.

**Mental model:** An assistant who shows up before you do and leaves a tidy note on your desk.

**Beginner mistakes:**
- Forgetting to handle the empty case (no emails today) and shipping junk summaries.
- No logging, so when it breaks at 3am you have no idea why.
- Trusting the LLM without a sanity check on costs or token usage.

**Exercise:** Pick one repeating chore in your life (RSS reading, expense categorisation, journal prompts, GitHub issue triage). Write a script that runs once a day, uses an LLM for the smart step, and delivers the output to a file, an email, or a Slack channel. Schedule it. Let it run for a week.

## AI SaaS workflows

A SaaS workflow is what happens when other people start using your tool. Suddenly your script-on-your-laptop becomes a service that strangers hit at unpredictable times. You need a front door (the API), a waiting room (the queue), a kitchen (the workers), and a filing cabinet (the database).

Imagine a restaurant. The waiter takes the order (API), writes it on a ticket and hangs it on the rail (queue), the chefs pull tickets and cook (workers), and every meal gets logged in the receipts book (DB). LLM calls are slow and sometimes fail, so you almost never want to do them inside the API request itself. The waiter does not cook.

Here is the architecture sketch.

```
[ Browser ]
     |
     v
[ FastAPI endpoint ]  ---creates job---> [ Postgres: jobs table ]
     |                                          ^
     |  enqueue job_id                          | poll status
     v                                          |
[ Queue: Inngest / Trigger.dev / Modal ]        |
     |                                          |
     v                                          |
[ Worker function ]                             |
     - load job from DB                         |
     - call LLM (may take 30s)                  |
     - call tools, retry on failure             |
     - write result back to DB  -----------------
```

Tools like Modal, Inngest, and Trigger.dev exist exactly to make this pattern easy. They give you a `@function` decorator, handle retries, and let you scale workers up and down without thinking about servers. A tiny Modal example.

```python
import modal

app = modal.App("summarizer")
image = modal.Image.debian_slim().pip_install("anthropic")

@app.function(image=image, secrets=[modal.Secret.from_name("anthropic")])
def summarize(text: str) -> str:
    from anthropic import Anthropic
    client = Anthropic()
    msg = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=512,
        messages=[{"role": "user", "content": f"Summarize: {text}"}],
    )
    return msg.content[0].text

@app.local_entrypoint()
def main():
    print(summarize.remote("Modal turns Python functions into cloud workers."))
```

Run with `modal run app.py` and your function runs in the cloud, on demand, with retries and logs.

**Summary:** A SaaS workflow separates the fast API from the slow LLM work using a queue and workers, with a DB tracking job state.

**Mental model:** A restaurant kitchen. Waiters take orders, the rail holds tickets, chefs cook, the receipts book never lies.

**Beginner mistakes:**
- Calling the LLM inside the request handler, then watching the page time out.
- No job table, so users have no way to ask "is it done yet?".
- Skipping retries; LLM APIs fail more often than you expect.

**Exercise:** Take the email summarizer from the automation exercise. Wrap it in a FastAPI endpoint that accepts a user's email credentials (or a pasted block of text), enqueues a job to Modal or Inngest, stores the result in SQLite, and exposes a `/status/{job_id}` endpoint. Ship it on a free tier somewhere.

## AI coding workflows

The fastest way to get better at building with AI is to use AI to write the code. There are three families of tools and they each shine in different spots.

Cursor is an editor with AI baked in. You see the cursor, you press a keystroke, you get a diff. Best for: editing files you already know, autocompleting in the middle of a function, quick refactors you can eyeball.

Claude Code (and similar terminal agents) live in your shell. You give a task, the agent reads files, runs tests, writes code, and reports back. Best for: tasks that span many files, anything where running tests in a loop is useful, scripted refactors, bootstrapping a new project.

Aider is the original git-aware CLI coder. You add files to a chat, it edits them and commits. Best for: small focused changes with a clear scope, when you want every change to be a clean commit.

A rough rule of thumb.

```
Need:                              Use:
- a tweak in a function I see      Cursor inline edit
- a whole new feature, many files  Claude Code
- a clean commit-per-change loop   Aider
- a one-off script                 ChatGPT/Claude in browser
```

The meta-skill is writing good prompts for code. Three habits compound fast.

1. Give the agent the goal, not the steps. "Add a /status endpoint that returns job state from the jobs table" beats "open routes.py, add an import, write a function".
2. Show it the contract. Paste the type, the schema, the failing test. Models follow examples better than descriptions.
3. Ask for tests in the same turn. "Add the feature and a pytest that covers it" forces verifiable output.

```python
# A prompt pattern that works well, dropped into Claude Code or Aider:
"""
Goal: Add rate limiting to the /summarize endpoint.
Limit: 10 requests per minute per IP.
Use: slowapi (already in requirements.txt).
Tests: add tests/test_rate_limit.py that hits the endpoint 11 times
       and asserts the 11th returns 429.
Run: pytest -x after the change.
"""
```

**Summary:** Pick the right AI coding tool for the shape of the change, and write prompts that include goal, contract, and tests.

**Mental model:** Cursor is a power-tool drill, Claude Code is a contractor you brief, Aider is a careful mason laying one brick at a time.

**Beginner mistakes:**
- Using the agent for tiny edits you could type faster yourself.
- Not reading the diff before accepting it.
- Skipping tests because the code "looks right".

**Exercise:** Take the SaaS workflow from the previous exercise. Use Claude Code or Aider to add: user authentication, a rate limit, and a simple HTML status page. For each change, ask the agent to also write tests, and review every diff before committing.

## AI orchestration systems

When your workflow has more than three steps, branches, retries, and humans in the loop, you need orchestration. Orchestration is the conductor in front of the orchestra. Each musician (LLM call, API call, database write) knows their part; the conductor decides when each plays and what to do if someone misses a note.

Common orchestration tools: LangGraph (graph of LLM steps, great for agent-style flows), Inngest (event-driven workflows with retries, great for SaaS), Temporal (industrial-strength durable workflows, great when you cannot lose state). Pick the lightest one that fits.

Here is a sketch of a multi-step pipeline: a research agent that takes a question, searches the web, reads the top three pages, drafts an answer, and asks a human to approve before sending.

```
              +-------------------+
              |  receive question |
              +---------+---------+
                        |
                        v
              +-------------------+
              |   web_search      |  (retry x3 on failure)
              +---------+---------+
                        |
              +---------+---------+
              |  for each result: |
              |   fetch + extract |  (parallel, retry x2)
              +---------+---------+
                        |
                        v
              +-------------------+
              |   draft_answer    |  (LLM call, retry x1)
              +---------+---------+
                        |
                        v
              +-------------------+
              |  await human OK   |  (durable wait, could be hours)
              +---------+---------+
                        |
                        v
              +-------------------+
              |   send answer     |
              +-------------------+
```

In Inngest-style Python pseudocode the same flow reads:

```python
import inngest

client = inngest.Inngest(app_id="research-agent")

@client.create_function(
    fn_id="answer-question",
    trigger=inngest.TriggerEvent(event="question.asked"),
)
async def answer(ctx, step):
    q = ctx.event.data["question"]

    results = await step.run("search", lambda: web_search(q))

    pages = await step.parallel([
        ("fetch_" + str(i), lambda r=r: fetch_and_extract(r))
        for i, r in enumerate(results[:3])
    ])

    draft = await step.run("draft", lambda: llm_draft(q, pages))

    approval = await step.wait_for_event(
        "approval",
        event="answer.approved",
        timeout="24h",
        if_=f"async.data.question_id == '{q}'",
    )
    if approval is None:
        return {"status": "timed_out"}

    await step.run("send", lambda: send_answer(q, draft))
    return {"status": "sent"}
```

The key win: each `step.run` is independently retried and cached. If the process crashes after `draft` succeeds, on restart the workflow resumes at "await human" without redoing the LLM call.

**Summary:** Orchestration adds retries, parallelism, and durable state to multi-step LLM pipelines, so half-finished work survives crashes and slow humans.

**Mental model:** A conductor with a score. Each player can fumble; the conductor restarts only that bar, never the whole symphony.

**Beginner mistakes:**
- Reaching for Temporal when a single Python script would do.
- Wrapping a step but not making it idempotent, so retries duplicate work.
- No timeout on human-in-the-loop waits, leaving workflows hanging forever.

**Exercise:** Convert your email summarizer into a multi-step Inngest or LangGraph workflow: fetch emails, classify each one (urgent / FYI / spam), summarize the urgent group with an LLM, send the summary to your phone, wait up to 4 hours for you to reply with "archive rest" or "keep". On reply, take the action.

## AI product thinking

The hardest skill is not technical. It is knowing what to build, for whom, and how small you can start. The best AI product people I know follow a few habits that keep them out of the swamp.

Start with one painful problem in one person's life. Yours counts. "I waste 20 minutes a day triaging email" is a problem. "AI for productivity" is not. Specific beats clever every single time.

Fake it before you build it. Before writing a line of code, do the task by hand with ChatGPT and a Notion doc. If that workflow does not save you time, no amount of FastAPI and LangGraph will. The wizard-of-oz version teaches you the prompt, the inputs, and the output format you actually need.

Decide your AI lever in this order: prompt, then RAG, then fine-tune. Most problems are solved by a better prompt and the right context. RAG is what you reach for when the model needs facts it does not have, like your private docs. Fine-tuning is for when you have thousands of examples of a specific style or format and the prompt is too long. Jumping to fine-tuning first burns weeks and usually loses to a good system prompt.

Always build an eval loop, even a crappy one. Twenty input-output pairs in a CSV and a script that scores them is better than nothing. Without evals you cannot tell if your new prompt is better or worse, and you will ship regressions.

A practical scoping checklist before you write code.

```
1. Who is the one user?  (name them)
2. What is the one job?  (one sentence)
3. What does success look like?  (a measurable thing)
4. What is the dumbest version that solves it?  (UI-less, manual)
5. What evals will tell me it works?  (5-20 examples)
6. What is the smallest shippable v1?  (one feature, one user)
```

**Summary:** Pick one real problem, fake the solution by hand first, choose the lightest AI lever, and measure before you ship.

**Mental model:** A doctor: diagnose before prescribing, try aspirin before surgery, take the patient's temperature before and after.

**Beginner mistakes:**
- Building a general "AI assistant" instead of a tool for a specific job.
- Fine-tuning before trying a better prompt or RAG.
- Shipping with no evals, then arguing about whether the latest tweak helped.

**Exercise:** Write a one-page product spec for an AI tool that solves a real annoyance in your life. Include: the one user, the one job, the dumbest manual version, the 10 eval examples, and the v1 feature list. Build the dumbest manual version first, this week. Use it. Then build v1.

## Where to go from here

You have the full kit now: models, prompts, embeddings, RAG, agents, tools, evals, deployment, and the product instincts to point them at something worth doing. Head back to the [curriculum index](./README.md) any time you want to revisit a topic.

Your homework, and it is the most important one: pick one real problem in your life this week. Build a tiny AI tool for it. Ship it to one user. You count as a user. The point is not the tool, it is finishing the loop. Once you have shipped one thing, end to end, the next ten get faster.

Go build.
