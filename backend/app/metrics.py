from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

from .retrieval import RankedChunk
from .schemas import BenchmarkRow


def recall_at_k(results: list[RankedChunk], relevant_chunk_ids: set[str], k: int = 5) -> float:
    if not relevant_chunk_ids:
        return 0.0
    retrieved = {result.chunk.id for result in results[:k]}
    return len(retrieved & relevant_chunk_ids) / len(relevant_chunk_ids)


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return ordered[index]


def benchmark_latency(search_fn: Callable[[str], list[RankedChunk]], queries: list[str]) -> float:
    latencies = []
    for query in queries:
        started = time.perf_counter()
        search_fn(query)
        latencies.append((time.perf_counter() - started) * 1000)
    return p95(latencies)


def load_benchmark_rows(path: str = "artifacts/metrics.json") -> list[BenchmarkRow]:
    metrics_path = Path(path)
    if metrics_path.exists():
        rows = json.loads(metrics_path.read_text())
        return [BenchmarkRow(**row) for row in rows]
    return [
        BenchmarkRow(model="BM25", recall_at_5=0.0, p95_latency_ms=0.0, size_mb=None),
        BenchmarkRow(model="Base embedding model", recall_at_5=0.0, p95_latency_ms=0.0, size_mb=None),
        BenchmarkRow(model="Fine-tuned ONNX INT8", recall_at_5=0.0, p95_latency_ms=0.0, size_mb=None),
    ]

