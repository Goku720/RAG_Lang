"""
rag.py - the brain of the chatbot.

Two jobs:
  1. INGEST: read a PDF, cut it into chunks, turn each chunk into numbers
     (an "embedding"), and save everything in ChromaDB.
  2. ANSWER: given a question, find the most relevant chunks, then ask an LLM
     to answer using ONLY those chunks. This flow is built with LangGraph.

Tip: set DEBUG=1 in your .env file to print what happens at every step.
"""
import os
from typing import List, TypedDict

import chromadb
from dotenv import load_dotenv
from groq import Groq
from langgraph.graph import END, START, StateGraph
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

load_dotenv(".env")  # reads GROQ_API_KEY etc. from the .env file

# ------------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------------
TOP_K = 6                                         # how many chunks to retrieve per question
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.30"))  # below this similarity, we refuse to answer
NOT_FOUND = "I couldn't find that in the document."
DEBUG = os.getenv("DEBUG") == "1"


def log(message):
    """Print only when DEBUG=1 (helps you see what the pipeline is doing)."""
    if DEBUG:
        print(f"[debug] {message}")


# ------------------------------------------------------------------
# SETUP (runs once when the app starts)
# ------------------------------------------------------------------
# Turns text into a list of 384 numbers. Similar meaning = similar numbers.
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# The LLM client (talks to Groq's servers).
groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ChromaDB = our vector database. It saves to the ./chroma_db folder.
# "cosine" means: measure how close two embeddings are by their angle.
chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_or_create_collection("pdf_chunks", metadata={"hnsw:space": "cosine"})


# ------------------------------------------------------------------
# PART 1: INGESTION  (PDF -> chunks -> embeddings -> ChromaDB)
# ------------------------------------------------------------------
def split_into_chunks(text, chunk_size=160, overlap=30):
    """
    Cut text into pieces of ~160 words. Each piece repeats the last 30 words
    of the previous one, so a sentence at a boundary isn't lost.
    (160 words keeps us under the embedding model's ~256-token limit.)
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + chunk_size]))
        if start + chunk_size >= len(words):  # that was the last chunk
            break
        start += chunk_size - overlap         # move forward, keeping some overlap
    return chunks


def ingest_pdf(path, name):
    """Read a PDF and store all its chunks in ChromaDB. Returns the chunk count."""
    texts = []      # the text of every chunk
    metadatas = []  # a label for every chunk: which file and which page

    # Go through the PDF page by page so we can remember page numbers.
    reader = PdfReader(path)
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""   # scanned pages give no text
        for chunk in split_into_chunks(page_text):
            texts.append(chunk)
            metadatas.append({"source": name, "page": page_number})

    if not texts:
        raise ValueError("No text found in this PDF (it may be a scanned image).")

    # If this file was uploaded before, remove the old chunks first.
    collection.delete(where={"source": name})

    # Turn every chunk into numbers, then save text + numbers + labels.
    embeddings = embedder.encode(texts).tolist()
    ids = [f"{name}-{i}" for i in range(len(texts))]   # every chunk needs a unique id
    collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)

    log(f"Stored {len(texts)} chunks from {name}")
    return len(texts)


def list_documents():
    """Names of all PDFs stored so far (used by the dropdown in the UI)."""
    labels = collection.get(include=["metadatas"])["metadatas"]
    return sorted({label["source"] for label in labels})


# ------------------------------------------------------------------
# PART 2: THE LANGGRAPH PIPELINE
# ------------------------------------------------------------------
# The STATE is a shared notebook. Every step (node) can read it and add to it.
# It starts with just "question" and "source"; the rest gets filled in as we go.
class RAGState(TypedDict, total=False):
    question: str        # what the user asked
    source: str          # which PDF to search
    chunks: List[dict]   # retrieved chunks: {"text", "page", "score"}
    confidence: float    # similarity of the best chunk (0 to 1)
    answer: str          # the final answer
    grounded: bool       # True if the answer came from the document


# ----- NODE 1: retrieve -----
# A node is just a function: it takes the state and returns ONLY the fields it
# wants to add or change.
def retrieve(state):
    # Turn the question into numbers, using the same model as the chunks.
    question_embedding = embedder.encode(state["question"]).tolist()

    # Ask ChromaDB for the closest chunks, only from the chosen PDF.
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=TOP_K,
        where={"source": state["source"]},
    )

    # Chroma returns lists-inside-lists (it supports many questions at once),
    # so we take [0] to get the results for our single question.
    texts = results["documents"][0]
    labels = results["metadatas"][0]
    distances = results["distances"][0]

    chunks = []
    for text, label, distance in zip(texts, labels, distances):
        similarity = 1 - distance   # Chroma gives distance; similarity = 1 - distance
        chunks.append({"text": text, "page": int(label["page"]), "score": round(similarity, 3)})

    # Results are sorted best-first, so the first chunk gives our confidence.
    confidence = chunks[0]["score"] if chunks else 0.0
    log(f"retrieve: {len(chunks)} chunks, best score = {confidence}")
    return {"chunks": chunks, "confidence": confidence}


# ----- DECISION: which node comes next? -----
# This is not a node (it changes nothing). It just returns the NAME of the next step.
def decide(state):
    if state["confidence"] >= MIN_SCORE:
        return "generate"   # chunks look relevant -> ask the LLM
    return "refuse"         # nothing relevant -> don't even call the LLM


# ----- NODE 2: generate -----
def generate(state):
    # Put the retrieved chunks into the prompt, numbered so the LLM can cite them.
    context = ""
    for number, chunk in enumerate(state["chunks"], start=1):
        context += f"[{number}] (page {chunk['page']}) {chunk['text']}\n\n"

    # The prompt is what keeps the answer "grounded" in the PDF.
    prompt = (
        "You are a strict document Q&A assistant.\n"
        "Answer ONLY using the numbered context below. Do not use outside knowledge.\n"
        "Cite the chunks you used, like [1].\n"
        f"If the context does not contain the answer, reply exactly: {NOT_FOUND}\n"
        "If it only partly answers the question, give what it does say and cite it.\n\n"
        f"Context:\n{context}\n"
        f"Question: {state['question']}\n\nAnswer:"
    )

    response = groq.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,   # 0 = least creative, most consistent
    )
    answer = response.choices[0].message.content.strip()

    # If the LLM itself said "not found", mark the answer as not grounded.
    grounded = not answer.startswith(NOT_FOUND)
    log(f"generate: grounded = {grounded}")
    return {"answer": answer, "grounded": grounded}


# ----- NODE 3: refuse -----
def refuse(state):
    log("refuse: confidence too low, LLM not called")
    return {"answer": NOT_FOUND, "grounded": False}


# ----- BUILD THE GRAPH (connect the steps) -----
#
#   START -> retrieve -> (decide) -> generate -> END
#                            \------> refuse ---> END
#
builder = StateGraph(RAGState)

builder.add_node("retrieve", retrieve)   # register each step under a name
builder.add_node("generate", generate)
builder.add_node("refuse", refuse)

builder.add_edge(START, "retrieve")      # always begin with retrieve
builder.add_conditional_edges(           # after retrieve, call decide() to pick the path
    "retrieve", decide, {"generate": "generate", "refuse": "refuse"}
)
builder.add_edge("generate", END)        # both paths finish the run
builder.add_edge("refuse", END)

graph = builder.compile()                # lock it in; now it can be run


# ------------------------------------------------------------------
# The function app.py calls
# ------------------------------------------------------------------
def ask(question, source):
    """Run the whole pipeline for one question and return what the API needs."""
    final_state = graph.invoke({"question": question, "source": source})
    return {
        "answer": final_state["answer"],
        "confidence": final_state["confidence"],
        "grounded": final_state["grounded"],
        "chunks": final_state["chunks"],
    }