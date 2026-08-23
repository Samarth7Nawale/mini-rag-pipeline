"""
Mini RAG (Retrieval-Augmented Generation) Demo -- v2, semantic embeddings
--------------------------------------------------------------------------
Same 6-step pipeline as v1, but Step 2 (Embedding) now uses a real neural
embedding model (`sentence-transformers/all-MiniLM-L6-v2`) instead of
TF-IDF. This is the single change that takes the pipeline from "keyword
matching" to "meaning matching" -- e.g. it correctly connects the query
"store files in the cloud" to the AWS S3 chunk, even though the two share
almost no words in common.

  1. Chunking      -> split documents into small pieces
  2. Embedding     -> sentence-transformers encodes each chunk as a
                       384-dim vector that captures semantic meaning
  3. Storing       -> keep vectors + text together ("vector store")
  4. Retrieval     -> cosine similarity between query vector and chunk
                       vectors
  5. Augmentation  -> build a prompt with retrieved context
  6. Generation    -> (simulated) answer using only the retrieved context

Requires an internet connection the first time it runs, to download the
~90MB model from Hugging Face. After that it's cached locally and runs
offline.
"""

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except ImportError:
    _HAS_ST = False

# -------------------------------------------------------------
# STEP 0: Knowledge base (same as v1)
# -------------------------------------------------------------
documents = [
    "AWS S3 is an object storage service that offers industry-leading "
    "scalability, data availability, security, and performance.",

    "AWS Lambda lets you run code without provisioning or managing servers. "
    "You pay only for the compute time you consume.",

    "RAG (Retrieval-Augmented Generation) combines a retrieval system with "
    "a language model so answers are grounded in real documents.",

    "IPL is the Indian Premier League, a professional Twenty20 cricket "
    "league in India, contested by ten city-based franchise teams.",

    "A vector database stores embeddings and allows fast similarity search, "
    "commonly using cosine similarity or approximate nearest neighbor search.",

    "Amazon EC2 provides resizable compute capacity in the cloud, letting "
    "you launch virtual servers called instances on demand.",
]

# -------------------------------------------------------------
# STEP 1: Chunking (docs are already short -> one chunk each)
# -------------------------------------------------------------
chunks = documents

# -------------------------------------------------------------
# STEP 2: Embedding
# -------------------------------------------------------------
if _HAS_ST:
    print("Loading embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(texts):
        return model.encode(texts, normalize_embeddings=True)

    chunk_vectors = embed(chunks)
    BACKEND = "sentence-transformers (semantic)"
else:
    # Fallback so the script still runs somewhere without internet/the
    # package installed -- falls back to v1's TF-IDF behavior.
    print("sentence-transformers not available, falling back to TF-IDF. "
          "Run: pip install sentence-transformers")
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words="english")
    chunk_vectors = vectorizer.fit_transform(chunks).toarray()

    def embed(texts):
        return vectorizer.transform(texts).toarray()

    BACKEND = "TF-IDF (keyword) -- fallback"

# -------------------------------------------------------------
# STEP 3: Storing (in-memory here; FAISS/Pinecone/pgvector in production)
# -------------------------------------------------------------

def cosine_sim(a, b):
    a, b = np.asarray(a), np.asarray(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def retrieve(query, top_k=2):
    """STEP 4: Retrieval."""
    query_vector = embed([query])[0]
    sims = [cosine_sim(query_vector, v) for v in chunk_vectors]
    top_indices = np.argsort(sims)[::-1][:top_k]
    return [(chunks[i], sims[i]) for i in top_indices]


def augment_prompt(query, retrieved_chunks):
    """STEP 5: Augmentation."""
    context = "\n".join(f"- {text}" for text, score in retrieved_chunks)
    return (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        f"Answer using only the context above."
    )


def fake_generate(query, retrieved_chunks):
    """STEP 6: Generation (simulated -- see rag_live_demo.html for a real
    Claude API call using this same retrieved context)."""
    best_chunk, score = retrieved_chunks[0]
    return f"(grounded in retrieved chunk, similarity={score:.3f}): {best_chunk}"


if __name__ == "__main__":
    print(f"\nEmbedding backend: {BACKEND}\n")

    queries = [
        "How do I store files in the cloud?",   # the case TF-IDF got wrong
        "What is RAG used for?",
        "Tell me about the cricket league",
        "How does similarity search work?",
    ]

    for q in queries:
        print("=" * 70)
        print(f"QUERY: {q}")
        results = retrieve(q, top_k=2)

        print("\n-- Retrieved chunks --")
        for text, score in results:
            print(f"  [{score:.3f}] {text}")

        print("\n-- Prompt sent to LLM --")
        print(augment_prompt(q, results))

        print("\n-- Final (simulated) answer --")
        print(fake_generate(q, results))
        print()
