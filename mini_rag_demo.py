"""
Mini RAG (Retrieval-Augmented Generation) Demo
------------------------------------------------
This is a tiny, self-contained pipeline that shows every RAG step:
  1. Chunking      -> split documents into small pieces
  2. Embedding     -> turn text into vectors (using TF-IDF here, simple & offline)
  3. Storing       -> keep vectors + text together ("vector store")
  4. Retrieval     -> find the chunks most similar to a query
  5. Augmentation  -> build a prompt with retrieved context
  6. Generation    -> (simulated) answer using only the retrieved context

We use TF-IDF instead of a neural embedding model so this runs instantly,
offline, with no downloads. The CONCEPT is identical to real RAG systems
(OpenAI/Anthropic embeddings + Pinecone/FAISS) -- just swap the embedding
function and you have a production pipeline.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# -------------------------------------------------------------
# STEP 0: Our "knowledge base" -- imagine these are pulled from
# PDFs, wiki pages, Slack messages, etc.
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
# STEP 1: Chunking
# (Our docs are already short, so each doc = one chunk here.
#  In real systems you'd split long docs into ~200-500 word pieces.)
# -------------------------------------------------------------
chunks = documents

# -------------------------------------------------------------
# STEP 2: Embedding
# TF-IDF turns each chunk into a vector based on word importance.
# (A real system would use something like OpenAI/Cohere/Anthropic
#  embeddings, which capture MEANING, not just word overlap --
#  but the retrieval math below is exactly the same.)
# -------------------------------------------------------------
vectorizer = TfidfVectorizer(stop_words="english")
chunk_vectors = vectorizer.fit_transform(chunks)

# -------------------------------------------------------------
# STEP 3: Storing
# Here that's just `chunk_vectors` + `chunks` living together in memory.
# In production this would be Pinecone / FAISS / pgvector / Weaviate.
# -------------------------------------------------------------

def retrieve(query, top_k=2):
    """STEP 4: Retrieval -- embed the query, compare to all chunks,
    return the most similar ones."""
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, chunk_vectors)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [(chunks[i], similarities[i]) for i in top_indices]


def augment_prompt(query, retrieved_chunks):
    """STEP 5: Augmentation -- build the final prompt that would be
    sent to an LLM."""
    context = "\n".join(f"- {text}" for text, score in retrieved_chunks)
    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        f"Answer using only the context above."
    )
    return prompt


def fake_generate(query, retrieved_chunks):
    """STEP 6: Generation -- normally this is a real LLM call
    (e.g. Claude/GPT). Here we just simulate it by returning the
    top chunk, to show how the final answer stays grounded in
    the retrieved text instead of the model's own memory."""
    best_chunk, score = retrieved_chunks[0]
    return f"(grounded in retrieved chunk, similarity={score:.2f}): {best_chunk}"


if __name__ == "__main__":
    queries = [
        "How do I store files in the cloud?",
        "What is RAG used for?",
        "Tell me about the cricket league",
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
