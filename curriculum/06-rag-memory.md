# RAG and Memory

Large language models are smart, but they forget everything between chats and they do not know your private files. In this chapter we will fix both problems by teaching the model to look things up and remember what matters.

## RAG

Imagine you are taking an exam. A closed-book exam is hard, because you have to remember every fact perfectly. An open-book exam is much easier: you do not need to memorize, you just need to know where to look. RAG, which stands for Retrieval-Augmented Generation, turns every LLM call into an open-book exam.

The recipe is simple. When the user asks a question, we first search a collection of documents for the most relevant snippets. Then we paste those snippets into the prompt and tell the model: "Use these notes to answer." The model writes a fluent answer, but the facts come from your trusted source. This is why RAG is the go-to trick for chatbots that need to answer from a company handbook, a product manual, or your personal notes.

Without RAG, the model guesses from training data that might be old or wrong. With RAG, the model becomes a smart assistant who reads your folder before speaking. The model does not get smarter, but the answers get more correct.

Here is the smallest possible RAG using `sentence-transformers` and plain Python.

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

docs = [
    "The office Wi-Fi password is alpine2026.",
    "Lunch is served between 12:00 and 13:30.",
    "Fire drills happen on the first Monday of each month.",
]
doc_vectors = model.encode(docs, convert_to_tensor=True)

question = "When can I eat lunch?"
q_vector = model.encode(question, convert_to_tensor=True)

scores = util.cos_sim(q_vector, doc_vectors)[0]
best = scores.argmax().item()
print("Context:", docs[best])
print("Now feed that context plus the question to your LLM.")
```

**Summary.** RAG fetches relevant text first, then asks the LLM to answer using that text as context.

**Mental model.** RAG is an open-book exam where you write the notes the model is allowed to peek at.

**Beginner mistakes.**
- Stuffing the whole document into the prompt instead of retrieving the relevant slice.
- Forgetting to tell the model "answer only from the context," so it still hallucinates.
- Skipping a citation step, so users cannot verify where an answer came from.

**Exercise.** Embed the five sentences above plus two more of your own. Ask three questions and print which sentence wins each time. Notice when the cosine score is low - that is the model telling you "I do not really know."

## Vector databases

A vector database is a library with a very clever index card system. A normal library sorts books by title or author. A vector library sorts them by meaning. Two books about hiking sit close together even if one is called "Alpine Trails" and the other "Mountain Walks for Beginners."

Under the hood, every chunk of text becomes a list of numbers, called an embedding. The vector database stores those numbers and, given a new query vector, returns the nearest neighbors in milliseconds. The popular options are Chroma (easy, runs locally), FAISS (fast, from Meta), Qdrant (production-ready, has a server), and Pinecone (managed cloud). For a beginner laptop, Chroma is the friendliest.

You do not need to understand the math to use it. You only need to remember three verbs: `add` documents, `query` for neighbors, and `delete` when things get stale.

```python
import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_store")
collection = client.get_or_create_collection("handbook")

docs = [
    "The office Wi-Fi password is alpine2026.",
    "Lunch is served between 12:00 and 13:30.",
    "Fire drills happen on the first Monday of each month.",
]
embeddings = model.encode(docs).tolist()
collection.add(
    ids=[f"doc-{i}" for i in range(len(docs))],
    documents=docs,
    embeddings=embeddings,
)

q = "what time is food?"
q_vec = model.encode([q]).tolist()
results = collection.query(query_embeddings=q_vec, n_results=2)
print(results["documents"])
```

**Summary.** A vector database stores embeddings and finds nearest neighbors quickly, so retrieval scales from ten documents to ten million.

**Mental model.** It is a library where books are shelved by what they mean, not by their title.

**Beginner mistakes.**
- Mixing embeddings from two different models in the same collection - they live in different number-spaces and the distances become nonsense.
- Storing only the vector and losing the original text, so you cannot show the user the source.
- Forgetting to persist the client, then wondering why everything disappears when the script ends.

**Exercise.** Save a Chroma collection to disk with `PersistentClient`. Close the script, open a new one, reload the collection, and run a query. Confirm the data survived the restart.

## Chunking

You cannot hand a model a 300-page PDF and expect a good answer. You also cannot embed the whole book as a single vector, because the meaning gets averaged into mush. So we cut the book into index cards. That cutting is called chunking.

A good chunk is large enough to make sense on its own and small enough to be specific. A common starting point is 300 to 800 tokens per chunk with a small overlap of 50 to 100 tokens so that sentences are not sliced in half. Think of overlap as the tape that keeps the edges of two index cards readable.

Where you cut matters. Cutting on paragraph or sentence boundaries works far better than cutting every 500 characters blindly. For code, cut on functions. For Markdown, cut on headings. The goal is always: each chunk should answer a question on its own.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text = open("handbook.md", "r", encoding="utf-8").read()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80,
    separators=["\n## ", "\n\n", "\n", ". ", " "],
)
chunks = splitter.split_text(text)

for i, c in enumerate(chunks[:3]):
    print(f"--- chunk {i} ({len(c)} chars) ---")
    print(c[:200], "...\n")
```

**Summary.** Chunking turns big documents into bite-sized passages that embed cleanly and retrieve accurately.

**Mental model.** You are cutting a textbook into index cards, one idea per card, with a little overlap so nothing is lost at the seams.

**Beginner mistakes.**
- Chunks too small (50 tokens) - each card has no context and the model cannot answer.
- Chunks too large (4000 tokens) - retrieval pulls in lots of noise around the actual answer.
- Splitting in the middle of code blocks, tables, or sentences, which destroys meaning.

**Exercise.** Take a Markdown file you wrote (notes, a README, anything). Try chunk sizes of 200, 500, and 1500 characters. Print how many chunks you get and read a few. Pick the size that feels like a useful index card.

## Retrieval pipelines

A retrieval pipeline is the assembly line that turns a user question into a final answer. Picture a kitchen: the question is the order, the chunks are the ingredients, retrieval is the pantry trip, the LLM is the chef. A pipeline is just the sequence of steps between order and plate.

A solid pipeline has five stages. First, rewrite the question if it is vague (turn "what about lunch?" into a self-contained query). Second, embed the query. Third, search the vector store and grab the top-k chunks. Fourth, optionally re-rank those chunks with a smarter model to put the truly relevant ones first. Fifth, build the prompt and call the LLM.

Top-k is usually 4 to 10. Re-ranking with a cross-encoder catches cases where the first search was loose. The whole loop runs in well under a second on a laptop.

```python
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

embedder = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
client = chromadb.PersistentClient(path="./chroma_store")
collection = client.get_or_create_collection("handbook")

def retrieve(question, k=8, top_n=3):
    q_vec = embedder.encode([question]).tolist()
    hits = collection.query(query_embeddings=q_vec, n_results=k)
    candidates = hits["documents"][0]
    pairs = [(question, c) for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, candidates), reverse=True)
    return [c for _, c in ranked[:top_n]]

context = "\n\n".join(retrieve("when is lunch?"))
prompt = f"Answer using only this context:\n{context}\n\nQuestion: when is lunch?"
print(prompt)
```

**Summary.** A pipeline chains query rewriting, embedding, vector search, re-ranking, and prompting into one repeatable flow.

**Mental model.** It is a kitchen assembly line: order in, search the pantry, sort the best ingredients, the chef cooks the answer.

**Beginner mistakes.**
- Skipping re-ranking and trusting raw vector scores, which are often noisy.
- Using a different embedding model at query time than at index time.
- Sending the model 20 chunks "just in case" and burying the right one in noise.

**Exercise.** Take the Chroma collection from the last exercise. Add the re-ranker shown above. Compare the top result with and without re-ranking on three tricky questions. Note which version sounds more accurate.

## AI memory systems

A vanilla chatbot has the memory of a goldfish. Every new conversation starts from zero. Memory systems give the assistant something closer to a notebook it carries from meeting to meeting.

There are three flavors worth knowing. Short-term memory is the current chat window - just the last N messages stuffed back into the prompt. Summary memory is a paragraph the assistant rewrites after each turn to keep the gist without the word count. Long-term memory is a vector store of past facts ("the user lives in Zurich," "the user prefers brief answers") that the assistant retrieves on demand, exactly like RAG but over its own past.

The trick is knowing what to remember. Not every message is worth saving. A good memory layer writes down facts, decisions, and preferences, and ignores small talk. When the user comes back tomorrow, the system queries memory the same way RAG queries documents.

```python
import chromadb
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_store")
memory = client.get_or_create_collection("user_memory")

def remember(user_id, fact):
    vec = embedder.encode([fact]).tolist()
    memory.add(
        ids=[f"{user_id}-{abs(hash(fact))}"],
        documents=[fact],
        metadatas=[{"user": user_id}],
        embeddings=vec,
    )

def recall(user_id, query, k=3):
    vec = embedder.encode([query]).tolist()
    hits = memory.query(
        query_embeddings=vec,
        n_results=k,
        where={"user": user_id},
    )
    return hits["documents"][0]

remember("ciprian", "Ciprian prefers concise, bullet-point answers.")
remember("ciprian", "Ciprian is learning RAG this week.")
print(recall("ciprian", "how does the user like answers formatted?"))
```

**Summary.** Memory systems use the same retrieval tricks as RAG to give an assistant continuity across chats.

**Mental model.** The assistant carries a small notebook between meetings and flips to the right page when you walk in.

**Beginner mistakes.**
- Saving every message and drowning real facts in chit-chat.
- Forgetting to scope memory by user, so users see each other's history.
- Never expiring or updating memory, so old preferences override new ones forever.

**Exercise.** Build a tiny chat loop that, after each user message, asks the LLM "Is there a durable fact here? If so, summarize it in one line." Save the answer with `remember`. Recall before each new reply.

## Semantic search

Old-school search is keyword search: it matches the letters you typed. If you search "car" it will not find "automobile." Semantic search matches meaning, so "car" finds "vehicle," "auto," and "ride." It is the same vector trick you have already learned, repackaged as a feature instead of a step in RAG.

The win is real. Users do not type like a search engineer. They ask in everyday language, often with typos. Semantic search forgives all of that because the embedding model already knows that "lunch break" and "midday meal" point to the same idea. The best modern search bars are hybrid: they combine keyword search (BM25) with semantic search and merge the two ranked lists. Keywords are great for rare proper nouns; semantics are great for paraphrases.

You will recognize the code below - it is the same `encode` and `query` calls. That is the point: once you have an embedding model and a vector store, semantic search is almost free.

```python
import chromadb
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_store")
search = client.get_or_create_collection("articles")

articles = [
    "How to brew the perfect espresso at home.",
    "A beginner's guide to alpine hiking gear.",
    "Why your sourdough starter keeps dying.",
    "Top ten budget tents for weekend camping.",
]
search.add(
    ids=[f"a-{i}" for i in range(len(articles))],
    documents=articles,
    embeddings=embedder.encode(articles).tolist(),
)

query = "outdoor adventures for the weekend"
res = search.query(
    query_embeddings=embedder.encode([query]).tolist(),
    n_results=2,
)
for doc, dist in zip(res["documents"][0], res["distances"][0]):
    print(f"{dist:.3f}  {doc}")
```

**Summary.** Semantic search ranks results by meaning rather than exact words, making search forgiving of phrasing and typos.

**Mental model.** It is a search bar that understands synonyms because it speaks the language of ideas, not letters.

**Beginner mistakes.**
- Throwing away keyword search entirely - a hybrid score usually beats pure semantic.
- Ignoring distance thresholds, so the top result is always returned even when nothing fits.
- Using a tiny model for a multilingual catalog - pick a multilingual embedder if your users speak more than English.

**Exercise.** Capstone time. Take any Markdown file on your laptop. Chunk it with the splitter from the chunking section. Embed the chunks with `all-MiniLM-L6-v2` and store them in ChromaDB. Build a `retrieve()` function. Then call a local LLM through Ollama (`ollama run llama3.2`) using the `ollama` Python package, feeding it your retrieved context and a question. You now have a tiny private RAG that works offline.

## What's next

You can now stuff models with knowledge and give them memory. Next we wire them up to take actions: head to [07-agents-workflows.md](./07-agents-workflows.md) to learn agents and workflows.
