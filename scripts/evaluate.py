from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.metrics import benchmark_latency, recall_at_k
from app.retrieval import BM25Retriever, EmbeddingRetriever
from app.reranker import OnnxInt8Reranker
from app.store import SearchStore


def load_pairs(path: Path) -> list[dict[str, str]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="data/training_pairs.jsonl")
    parser.add_argument("--out", default="artifacts/metrics.json")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    pairs_path = Path(args.pairs)
    if not pairs_path.exists():
        from generate_dataset import build_examples

        pairs = build_examples(args.limit, seed=11)
    else:
        pairs = load_pairs(pairs_path)[: args.limit]

    store = SearchStore()
    eval_cases: list[tuple[str, set[str]]] = []
    for index, pair in enumerate(pairs):
        document, chunks = store.add_document(f"Example {index}", pair["positive"])
        eval_cases.append((pair["query"], {chunk.id for chunk in chunks}))
        store.add_document(f"Negative {index}", pair["negative"])

    bm25 = BM25Retriever(store.chunks)
    base = EmbeddingRetriever()
    finetuned = OnnxInt8Reranker()

    methods = [
        ("BM25", lambda query: bm25.search(query, top_k=5), None),
        ("Base embedding model", lambda query: base.search(query, store.chunks, top_k=5), None),
        (
            "Fine-tuned INT8 reranker",
            lambda query: finetuned.search(query, store.chunks, top_k=5),
            artifact_size_mb("artifacts/model-int8.onnx") or artifact_size_mb("artifacts/lightweight-reranker-int8.json"),
        ),
    ]

    rows = []
    queries = [query for query, _ in eval_cases]
    for name, search_fn, size_mb in methods:
        recalls = [recall_at_k(search_fn(query), relevant_ids, k=5) for query, relevant_ids in eval_cases]
        rows.append(
            {
                "model": name,
                "recall_at_5": round(sum(recalls) / max(1, len(recalls)), 3),
                "p95_latency_ms": round(benchmark_latency(search_fn, queries), 2),
                "size_mb": size_mb,
            }
        )

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


def artifact_size_mb(path: str) -> float | None:
    artifact = Path(path)
    if not artifact.exists():
        return None
    size = artifact.stat().st_size / (1024 * 1024)
    return round(size, 4) if size < 1 else round(size, 2)


if __name__ == "__main__":
    main()
