from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from .chunking import Chunk, tokenize
from .retrieval import BM25Retriever, EmbeddingRetriever, RankedChunk, hashing_embedding


class TrainedInt8Reranker:
    def __init__(
        self,
        artifact_path: str = "artifacts/model-int8.onnx",
        tokenizer_path: str = "artifacts/tokenizer",
        lightweight_path: str = "artifacts/lightweight-reranker-int8.json",
    ):
        self.artifact_path = Path(artifact_path)
        self.tokenizer_path = Path(tokenizer_path)
        self.lightweight_path = Path(lightweight_path)
        self._session = None
        self._tokenizer = None
        self._fallback = EmbeddingRetriever()
        self._artifact: dict[str, object] | None = None

    @property
    def status(self) -> str:
        if self.artifact_path.exists() and self.tokenizer_path.exists():
            if self._session is None or self._tokenizer is None:
                return f"onnx-int8 artifact: {self.artifact_path}"
            return "onnx-int8"
        if self.lightweight_path.exists():
            return f"lightweight-int8 artifact: {self.lightweight_path}"
        if not self.artifact_path.exists():
            return f"missing artifact: {self.artifact_path}"
        return f"missing tokenizer: {self.tokenizer_path}"

    def _load_lightweight(self) -> dict[str, object] | None:
        if self._artifact is not None:
            return self._artifact
        if not self.lightweight_path.exists():
            return None
        payload = json.loads(self.lightweight_path.read_text())
        if "weights_int8" in payload:
            self._artifact = payload
            return self._artifact
        if "weights" in payload:
            self._artifact = {
                "artifact_type": "legacy-term-reranker-int8",
                "weights": {str(term): int(weight) for term, weight in payload.get("weights", {}).items()},
            }
            return self._artifact
        return None

    def _load_session(self) -> None:
        if self._session is not None or not self.artifact_path.exists() or not self.tokenizer_path.exists():
            return
        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer

            self._session = ort.InferenceSession(str(self.artifact_path), providers=["CPUExecutionProvider"])
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        except Exception:
            self._session = None
            self._tokenizer = None

    def search(self, query: str, chunks: list[Chunk], top_k: int = 5) -> list[RankedChunk]:
        self._load_session()
        if self._session is not None and self._tokenizer is not None:
            bm25_results = BM25Retriever(chunks).search(query, top_k=len(chunks))
            return self._rerank_with_onnx(query, bm25_results)[:top_k]

        artifact = self._load_lightweight()
        if artifact:
            bm25_results = BM25Retriever(chunks).search(query, top_k=len(chunks))
            return self._rerank_with_lightweight_artifact(query, bm25_results, artifact)[:top_k]

        if self._session is None:
            base_results = self._fallback.search(query, chunks, top_k=top_k)
            return [RankedChunk(chunk=result.chunk, score=float(result.score) + 0.001) for result in base_results]

        return self._fallback.search(query, chunks, top_k=top_k)

    def _encode_onnx(self, texts: list[str]) -> np.ndarray:
        if self._session is None or self._tokenizer is None:
            return np.empty((0, 384), dtype=np.float32)

        encoded = self._tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="np")
        input_names = {input_meta.name for input_meta in self._session.get_inputs()}
        inputs = {name: value for name, value in encoded.items() if name in input_names}
        outputs = self._session.run(None, inputs)
        token_embeddings = outputs[0]
        mask = encoded["attention_mask"].astype(np.float32)
        masked = token_embeddings * np.expand_dims(mask, axis=-1)
        embeddings = masked.sum(axis=1) / np.clip(mask.sum(axis=1, keepdims=True), 1e-9, None)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return (embeddings / np.clip(norms, 1e-9, None)).astype(np.float32)

    def _rerank_with_onnx(self, query: str, candidates: list[RankedChunk]) -> list[RankedChunk]:
        if not candidates:
            return []
        query_vec = self._encode_onnx([query])[0]
        document_vecs = self._encode_onnx([candidate.chunk.text for candidate in candidates])
        scores = document_vecs @ query_vec
        reranked = [
            RankedChunk(chunk=candidate.chunk, score=float(candidate.score + scores[index]))
            for index, candidate in enumerate(candidates)
        ]
        return sorted(reranked, key=lambda item: item.score, reverse=True)

    def _rerank_with_lightweight_artifact(
        self,
        query: str,
        candidates: list[RankedChunk],
        artifact: dict[str, object],
    ) -> list[RankedChunk]:
        if "weights_int8" in artifact:
            dims = int(artifact.get("dims", 384))
            scale = float(artifact.get("scale", 1.0))
            weights = np.asarray(artifact["weights_int8"], dtype=np.float32) * scale
            bias = float(artifact.get("bias", 0.0))
            query_vec = hashing_embedding(query, dims=dims)
            reranked = []
            for candidate in candidates:
                document_vec = hashing_embedding(candidate.chunk.text, dims=dims)
                learned_score = float((query_vec * document_vec) @ weights + bias)
                reranked.append(RankedChunk(chunk=candidate.chunk, score=float(candidate.score + learned_score)))
            return sorted(reranked, key=lambda item: item.score, reverse=True)

        weights = artifact.get("weights", {})
        query_terms = set(tokenize(query))
        reranked = []
        for candidate in candidates:
            chunk_terms = set(tokenize(candidate.chunk.text))
            learned_score = sum(int(weights.get(term, 0)) for term in query_terms & chunk_terms) / 127.0
            reranked.append(RankedChunk(chunk=candidate.chunk, score=float(candidate.score + learned_score)))
        return sorted(reranked, key=lambda item: item.score, reverse=True)
