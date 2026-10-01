# PDF RAG Chatbot (LangGraph + ChromaDB)

Upload a PDF, ask questions about it, and get answers that come **only from that PDF**. Every answer shows the text chunks it was based on and a confidence score.

## How it works (plain English)

An LLM doesn't know what's inside your PDF, so we help it in two phases.

**Phase 1: Prepare the PDF (once per upload)**
1. **Read** the PDF page by page.
2. **Chunk**: cut the text into small pieces (~160 words each).
3. **Embed**: turn every chunk into a list of numbers that captures its meaning.
4. **Store** the chunks and numbers in ChromaDB, a database built for searching by meaning.

**Phase 2: Answer a question (every time)**
1. **Retrieve**: turn the question into numbers and find the 4 most similar chunks.
2. **Decide**: if even the best chunk is not similar enough, say "not found" without calling the LLM.
3. **Generate**: otherwise, send the chunks and the question to the LLM with strict instructions to answer only from them.
4. **Return** the answer, the chunks, and a confidence score.

## Project structure

| File | What it does |
|---|---|
| `rag.py` | The core logic: chunking, ChromaDB, and the LangGraph pipeline. Start here. |
| `app.py` | A small Flask web server with the routes `/upload`, `/ask`, `/documents`. |
| `static/index.html` | The chat page you see in the browser. |
| `requirements.txt` | Python packages to install. |
| `.env` | Template for your settings and API key. |

## Setup

You need Python 3.10+ and a free Groq API key (create one at console.groq.com).


Open `.env` and paste your key after `GROQ_API_KEY=`. Then run:

```bash
python app.py
```

Open http://127.0.0.1:5000, upload a PDF, choose it in the dropdown, and ask away.
The first start downloads the embedding model (~90 MB), so give it a minute.
Set `DEBUG=1` in `.env` to see each pipeline step printed in the terminal.

## API

| Route | Method | Send | Get back |
|---|---|---|---|
| `/upload` | POST | form-data field `pdf` | `source`, `chunks` |
| `/documents` | GET | nothing | list of uploaded PDFs |
| `/ask` | POST | `{"query": "...", "source": "file.pdf"}` | `answer`, `confidence`, `grounded`, `chunks` |

Each item in `chunks` has `text`, `page`, and `score`.

```bash
curl -X POST http://127.0.0.1:5000/ask -H "Content-Type: application/json" \
  -d '{"query": "What problem does the paper address?", "source": "paper.pdf"}'
```


## Architecture

```
PDF -> read pages -> chunks (160 words, 30 overlap) -> embeddings -> ChromaDB

question -> [retrieve] -> best score >= 0.30 ? --yes--> [generate] -> answer + chunks + score
                                      \--no---> [refuse]   -> "not found" + chunks + score
```

**LangGraph** runs the answering phase as a small graph with a shared **state** (a dictionary holding the question, chunks, confidence, and answer). Each node reads the state and returns only what it adds. One conditional edge picks `generate` or `refuse`.

**Keeping answers grounded:**
1. A relevance gate: low similarity means the LLM is never called.
2. A strict prompt: use only the context, cite `[n]`, and use a fixed refusal sentence.
3. Temperature 0 for consistent output.
4. The retrieved chunks are shown in the UI so you can check any answer.

**Confidence score:** the similarity (0 to 1) of the best retrieved chunk. It shows how relevant the retrieved text was, not whether the answer is guaranteed correct.

## Key terms

- **RAG**: retrieve relevant text first, then let an LLM answer from it.
- **Chunk**: a small piece of the PDF text.
- **Embedding**: a list of numbers representing a text's meaning.
- **Vector database**: a database that finds the closest embeddings (here, ChromaDB).
- **Cosine similarity**: how alike two embeddings are (1 = very similar, 0 = unrelated).
- **State / node / edge** (LangGraph): the shared data, a step that works on it, and a connection deciding what runs next.

## Troubleshooting

- **"No text found"**: the PDF is a scan (images). This project has no OCR.
- **Everything says "not found"**: lower `MIN_SCORE` in `.env` (for example 0.2).
- **Wrong answers slip through**: raise `MIN_SCORE`.
- **Groq errors**: check that `GROQ_API_KEY` is set correctly in `.env`.

## Limitations

- Only 4 chunks are used per question, so answers spread across many pages can be incomplete.
- Each question stands alone; there is no chat memory.
- Text-only PDFs; tables and images are not understood.