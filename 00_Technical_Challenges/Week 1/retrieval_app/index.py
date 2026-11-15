"""Stage 3 and 4: embed the chunks, store them, and search them.

YOUR CODE GOES HERE. `Hit` and the OpenAI client are written for you, because
plumbing is not the lesson. `embed_texts`, `cosine_similarity`, `build_index`
and `search` are the lesson.
"""

from dataclasses import dataclass

import numpy as np
from openai import OpenAI

EMBEDDING_MODEL = "text-embedding-3-small"

_cached: OpenAI | None = None


def client() -> OpenAI:
    """The OpenAI client, built on first use and reused after that.

    It is built lazily, not at import time, so that .env has already been
    loaded by the time it looks for OPENAI_API_KEY.
    """
    global _cached
    if _cached is None:
        _cached = OpenAI()
    return _cached


@dataclass
class Hit:
    """One retrieved chunk, and everything you need to cite it.

    `chunk_id` is what makes an answer traceable: when an answer is wrong in
    Week 4, this is how you find out which chunk caused it. Keep it.
    """

    chunk_id: int
    text: str
    score: float

    def citation(self) -> str:
        """A short, human-readable label for this chunk. Printed under answers."""
        preview = " ".join(self.text.split())[:60]
        return f"[chunk {self.chunk_id}] {preview}... (score {self.score:.3f})"


def embed_texts(texts: list[str]) -> np.ndarray:
    """Turn a list of strings into a matrix of embedding vectors.

    Args:
        texts: the strings to embed. One API call can take many at once —
            do NOT loop and call once per chunk.

    Returns:
        An array of shape (len(texts), 1536), one row per input string,
        in the same order as `texts`.

    Hint: `client().embeddings.create(input=texts, model=EMBEDDING_MODEL)`.
    The response has a `.data` list; each item has an `.embedding`. Wrap the
    result in np.array(). Print `response.usage` once — knowing what a corpus
    costs to embed is the number that decides architecture at scale.
    """
    embeddings_list = []
    response = client().embeddings.create(input=texts, model=EMBEDDING_MODEL)
    print(response.usage)
    embeddings_data = response.data
    for embedding_data in embeddings_data:
        embeddings_list.append(embedding_data.embedding)
    return np.array(embeddings_list)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """How similar are two vectors? 1.0 is identical, 0.0 is unrelated.

    Hint: the dot product, divided by the product of the two norms.
    NumPy has `np.dot` and `np.linalg.norm`.
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a * norm_b == 0:
        raise ValueError("Can't divide by zero.")

    return dot_product / (norm_a * norm_b)


def build_index(chunks: list[str]) -> np.ndarray:
    """Embed every chunk once, and return the matrix to search against.

    Args:
        chunks: from corpus.split_corpus().

    Returns:
        The embedding matrix. Row i corresponds to chunks[i] — that
        correspondence is the only thing holding your citations together,
        so do not sort or filter one without the other.

    Hint: this is mostly a call to embed_texts(). Notice that it happens once
    per process, and that a fresh process has no memory of the last one.
    """
    if len(chunks) == 0:
        raise ValueError("Chunks shouldn't be empty.")

    return embed_texts(chunks)


def search(query: str, chunks: list[str], matrix: np.ndarray, k: int = 3) -> list[Hit]:
    """Find the k chunks most similar to the query.

    Args:
        query: the user's question, as typed.
        chunks: the chunk texts.
        matrix: the embedding matrix from build_index().
        k: how many hits to return.

    Returns:
        Up to k Hits, most similar first.

    Hints:
      - Embed the query with the SAME model you built the index with. A query
        embedded by a different model is not comparable, and the failure is
        silent — you get plausible-looking nonsense, not an error.
      - Score the query against every row, then sort.
      - Read the signature again: it says "up to k". Nothing here forces you to
        return exactly k. That freedom is the whole of Step 4's threshold option.
    """
    if k <= 0:
        raise ValueError("The k value must be positive.")

    if len(chunks) != len(matrix):
        raise ValueError(
            "The number of chunks must match the number of embedding rows."
        )

    response = client().embeddings.create(input=query, model=EMBEDDING_MODEL)
    print(response.usage)
    query_embedding = np.array(response.data[0].embedding)

    hits = []

    for index, chunk in enumerate(chunks):
        comparison_value = cosine_similarity(query_embedding, matrix[index])
        hit = Hit(chunk_id=index, text=chunk, score=comparison_value)
        hits.append(hit)

    hits.sort(key=lambda hit_obj: hit_obj.score, reverse=True)

    return hits[0:k]
