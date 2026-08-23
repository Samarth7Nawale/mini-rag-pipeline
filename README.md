# Mini RAG (Retrieval-Augmented Generation) Demo

A small, from-scratch implementation of a RAG pipeline, built to understand
(and demonstrate) each stage of the process rather than treat it as a
black box: **chunking → embedding → storing → retrieval → augmentation →
generation.**

Two versions are included:

## 1. `mini_rag_demo.py` — command-line version
Implements the pipeline in Python using TF-IDF vectors (`scikit-learn`) for
embeddings and cosine similarity for retrieval. No API keys or internet
access required — runs fully offline.

```bash
pip install -r requirements.txt
python mini_rag_demo.py
```

It runs a few sample queries against a small knowledge base (AWS services,
RAG concepts, cricket, vector databases) and prints, for each one:
- the retrieved chunks and their similarity scores
- the final prompt that would be sent to an LLM
- a simulated grounded answer

## 2. `mini_rag_demo_v2_semantic.py` — same pipeline, real embeddings
Identical structure to v1, but Step 2 (Embedding) uses a real neural
embedding model (`sentence-transformers/all-MiniLM-L6-v2`, 384-dim vectors)
instead of TF-IDF. This is the upgrade that takes retrieval from
"keyword overlap" to "semantic meaning" — for example, it correctly
matches the query *"How do I store files in the cloud?"* to the AWS S3
chunk, even though the two share almost no words. The TF-IDF version
retrieves the wrong chunk (EC2) for that exact query — run both scripts
side by side to see the difference.

```bash
pip install -r requirements.txt
python mini_rag_demo_v2_semantic.py
```

The first run downloads the ~90MB model from Hugging Face (needs internet
once; cached locally after that). If `sentence-transformers` isn't
installed, the script automatically falls back to the v1 TF-IDF behavior
so it still runs.

## 3. `rag_live_demo.html` — interactive browser version
A single-file HTML/JS app that runs the same pipeline live in the browser,
plus a real call to the Claude API for the generation step (so the final
answer is genuinely LLM-generated, not simulated). Open the file directly
in a browser, type a question, and watch each pipeline stage light up with
real intermediate values — retrieval scores, the constructed prompt, and
the model's grounded response.

*Note: the live demo's API call is wired for the Claude API and expects
that endpoint to be reachable with credentials configured by whatever
environment serves the page — swap in your own key/proxy if hosting it
elsewhere.*

## Why this project
RAG is the standard technique for grounding an LLM's answers in a specific
set of documents (company docs, a codebase, anything outside the model's
training data) instead of relying on the model's own memory. This project
breaks the pipeline into its individual steps to show what's actually
happening at each stage:

- **Chunking** — splitting source documents into retrievable units
- **Embedding** — converting text into vectors that capture similarity
- **Storing** — keeping vectors + source text together (a vector store)
- **Retrieval** — finding the most relevant chunks for a query via
  cosine similarity
- **Augmentation** — building a prompt that injects retrieved context
- **Generation** — having the LLM answer using only that context

## Possible extensions
- Swap the in-memory vector store for FAISS, Pinecone, or pgvector
- Add re-ranking after initial retrieval
- Add source citations in the generated answer
