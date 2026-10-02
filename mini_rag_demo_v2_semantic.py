"""
Mini RAG (Retrieval-Augmented Generation) Demo -- v2, semantic embeddings
--------------------------------------------------------------------------
Same 6-step pipeline as v1, but with two real upgrades:

  1. Chunking now actually splits longer source documents into overlapping
     word-count windows (not just "one short sentence = one chunk").
  2. Embedding (Step 2) uses a real neural embedding model
     (`sentence-transformers/all-MiniLM-L6-v2`) instead of TF-IDF, so
     retrieval matches on MEANING, not just shared words -- e.g. it
     correctly connects the query "store files in the cloud" to the AWS
     S3 chunk, even though the two share almost no words in common.

  1. Chunking      -> split each source document into overlapping windows
  2. Embedding     -> sentence-transformers encodes each chunk as a
                       384-dim vector that captures semantic meaning
  3. Storing       -> keep vectors + chunk text together ("vector store")
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
# STEP 0: Knowledge base
# Longer source documents this time, so chunking actually has
# something to do.
# -------------------------------------------------------------
documents = [
    # Doc 0: AWS S3
    "Amazon S3 (Simple Storage Service) is an object storage service that "
    "offers industry-leading scalability, data availability, security, "
    "and performance. Customers of all sizes and industries can use S3 to "
    "store and protect any amount of data for a range of use cases, such "
    "as data lakes, websites, mobile applications, backup and restore, "
    "archive, enterprise applications, IoT devices, and big data analytics. "
    "S3 provides management features so you can organize your data and "
    "configure finely-tuned access controls to meet your specific business, "
    "organizational, and compliance requirements. Data in S3 is stored as "
    "objects within resources called buckets, and each object can be up to "
    "5 terabytes in size, making it a common choice for storing files "
    "in the cloud rather than on local servers.",

    # Doc 1: EC2
    "Amazon EC2 (Elastic Compute Cloud) provides resizable compute capacity "
    "in the cloud, letting you launch virtual servers called instances on "
    "demand. It is designed to make web-scale cloud computing easier for "
    "developers by giving you complete control over your computing "
    "resources. You can choose from a wide selection of instance types "
    "optimized for different workloads, such as compute-heavy, "
    "memory-heavy, or storage-heavy tasks. EC2 instances can be scaled up "
    "or down automatically based on demand, and you only pay for the "
    "compute capacity you actually use, which makes it well suited for "
    "running applications and services rather than storing files directly.",

    # Doc 2: RAG concept
    "Retrieval-Augmented Generation, or RAG, is a technique that combines "
    "a retrieval system with a language model so that answers are grounded "
    "in real, specific documents instead of relying purely on what the "
    "model memorized during training. A typical RAG pipeline works by "
    "splitting documents into chunks, converting each chunk into a vector "
    "embedding, storing those vectors in a searchable index, and then, at "
    "query time, retrieving the most relevant chunks and inserting them "
    "into the prompt before generation. This approach is especially useful "
    "for answering questions about private company data, recent events "
    "after a model's training cutoff, or any information the model was "
    "never trained on in the first place.",

    # Doc 3: Vector databases
    "A vector database is a specialized system for storing embeddings and "
    "performing fast similarity search over them, which is essential for "
    "retrieval-augmented generation at scale. Rather than scanning every "
    "vector one by one, vector databases use indexing structures such as "
    "HNSW or IVF to approximate the nearest neighbors of a query vector "
    "much more quickly. Popular examples include Pinecone, Weaviate, "
    "Chroma, Qdrant, and the pgvector extension for PostgreSQL. Most "
    "support metadata filtering alongside vector search, so you can, for "
    "example, restrict a search to documents from a specific date range "
    "or source before ranking them by embedding similarity.",

    # Doc 4: Lambda (kept short, still useful as a distractor)
    "AWS Lambda lets you run code without provisioning or managing "
    "servers. You pay only for the compute time you consume, and Lambda "
    "automatically scales your application by running code in response "
    "to triggers such as file uploads, API calls, or scheduled events.",

    # Doc 5: IPL (distractor, unrelated topic)
    "The IPL, or Indian Premier League, is a professional Twenty20 "
    "cricket league in India, contested every year by ten city-based "
    "franchise teams. It was founded in 2008 and has since become one of "
    "the most-watched sporting leagues in the world, attracting top "
    "international players alongside domestic talent.",
]

# -------------------------------------------------------------
# STEP 1: Chunking
# Real overlapping-window chunking: each document is split into
# word-count windows so long documents become multiple retrievable
# chunks, with some overlap so an idea near a boundary isn't lost.
# -------------------------------------------------------------

def chunk_text(text, chunk_size=40, overlap=10):
    """Split `text` into overlapping chunks of ~chunk_size words each,
    with `overlap` words repeated between consecutive chunks."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks_out = []
    start = 0
    step = chunk_size - overlap
    while start < len(words):
        window = words[start:start + chunk_size]
        chunks_out.append(" ".join(window))
        if start + chunk_size >= len(words):
            break
        start += step
    return chunks_out


chunks = []
chunk_sources = []  # which original document index each chunk came from
for doc_idx, doc in enumerate(documents):
    for c in chunk_text(doc, chunk_size=40, overlap=10):
        chunks.append(c)
        chunk_sources.append(doc_idx)

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


def retrieve(query, top_k=3):
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


def run_query(q):
    print("=" * 70)
    print(f"QUERY: {q}")
    results = retrieve(q, top_k=3)

    print("\n-- Retrieved chunks --")
    for text, score in results:
        print(f"  [{score:.3f}] {text}")

    print("\n-- Prompt sent to LLM --")
    print(augment_prompt(q, results))

    print("\n-- Final (simulated) answer --")
    print(fake_generate(q, results))
    print()


if __name__ == "__main__":
    print(f"\nEmbedding backend: {BACKEND}")
    print(f"{len(documents)} source documents split into {len(chunks)} chunks\n")

    # A few preset queries so you see it work immediately with no typing.
    preset_queries = [
        "How do I store files in the cloud?",   # the case TF-IDF got wrong
        "What is RAG used for?",
    ]
    for q in preset_queries:
        run_query(q)

    # Now type your own questions. Press Enter with nothing typed to quit.
    print("Preset queries done. Type your own question below")
    print("(or just press Enter with nothing typed to quit).\n")

    while True:
        user_query = input("Your question: ").strip()
        if not user_query:
            print("Goodbye!")
            break
        run_query(user_query)
