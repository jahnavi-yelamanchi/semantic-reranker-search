from __future__ import annotations

import hashlib
import math
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .chunking import Chunk, tokenize


@dataclass(frozen=True)
class RankedChunk:
    chunk: Chunk
    score: float


class BM25Retriever:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.tokens = [tokenize(chunk.text) for chunk in chunks]
        self.doc_freqs = Counter(token for doc in self.tokens for token in set(doc))
        self.avgdl = sum(len(doc) for doc in self.tokens) / max(1, len(self.tokens))

    def search(self, query: str, top_k: int = 5) -> list[RankedChunk]:
        if not self.chunks:
            return []
        query_terms = tokenize(query)
        scored = []
        total_docs = len(self.chunks)
        for chunk, doc_tokens in zip(self.chunks, self.tokens):
            tf = Counter(doc_tokens)
            doc_len = len(doc_tokens)
            score = 0.0
            for term in query_terms:
                if term not in tf:
                    continue
                idf = math.log(1 + (total_docs - self.doc_freqs[term] + 0.5) / (self.doc_freqs[term] + 0.5))
                denom = tf[term] + self.k1 * (1 - self.b + self.b * doc_len / max(self.avgdl, 1e-9))
                score += idf * (tf[term] * (self.k1 + 1)) / denom
            if score > 0:
                scored.append(RankedChunk(chunk=chunk, score=float(score)))
        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]


class EmbeddingRetriever:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._load_attempted = False

    @property
    def status(self) -> str:
        return "sentence-transformers" if self._model else "deterministic-hashing-fallback"

    def _load_model(self) -> None:
        if self._model is not None or self._load_attempted:
            return
        self._load_attempted = True
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        except Exception:
            self._model = None

    def encode(self, texts: list[str]) -> np.ndarray:
        self._load_model()
        if self._model is not None:
            vectors = self._model.encode(texts, normalize_embeddings=True)
            return np.asarray(vectors, dtype=np.float32)
        return np.vstack([hashing_embedding(text) for text in texts]).astype(np.float32)

    def search(self, query: str, chunks: list[Chunk], top_k: int = 5) -> list[RankedChunk]:
        if not chunks:
            return []
        started = time.perf_counter()
        query_vec = self.encode([query])[0]
        chunk_vecs = self.encode([chunk.text for chunk in chunks])
        scores = chunk_vecs @ query_vec
        order = np.argsort(-scores)[:top_k]
        _ = time.perf_counter() - started
        return [RankedChunk(chunk=chunks[i], score=float(scores[i])) for i in order]


def hashing_embedding(text: str, dims: int = 384) -> np.ndarray:
    vec = np.zeros(dims, dtype=np.float32)
    for token in tokenize(text):
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        index = int(digest[:8], 16) % dims
        sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
        vec[index] += sign
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec
