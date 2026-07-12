from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from generate_dataset import build_examples
from app.retrieval import hashing_embedding


def pair_features(query: str, document: str, dims: int) -> np.ndarray:
    query_vec = hashing_embedding(query, dims=dims)
    document_vec = hashing_embedding(document, dims=dims)
    return query_vec * document_vec


def build_training_matrix(pairs: list[dict[str, str]], dims: int) -> tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []
    for pair in pairs:
        features.append(pair_features(pair["query"], pair["positive"], dims))
        labels.append(1.0)
        features.append(pair_features(pair["query"], pair["negative"], dims))
        labels.append(0.0)
    return np.vstack(features).astype(np.float32), np.asarray(labels, dtype=np.float32)


def train_logistic_reranker(
    pairs: list[dict[str, str]],
    dims: int,
    epochs: int,
    learning_rate: float,
    l2: float,
) -> tuple[np.ndarray, float]:
    features, labels = build_training_matrix(pairs, dims)
    weights = np.zeros(dims, dtype=np.float32)
    bias = np.float32(0.0)

    for _ in range(epochs):
        logits = np.clip(features @ weights + bias, -30, 30)
        predictions = 1.0 / (1.0 + np.exp(-logits))
        errors = predictions - labels
        weights -= learning_rate * ((features.T @ errors) / len(labels) + l2 * weights)
        bias -= np.float32(learning_rate * float(errors.mean()))

    return weights, float(bias)


def quantize_int8(weights: np.ndarray) -> tuple[list[int], float]:
    max_abs = float(np.max(np.abs(weights))) if weights.size else 0.0
    scale = max(max_abs / 127.0, 1e-8)
    quantized = np.clip(np.rint(weights / scale), -127, 127).astype(np.int8)
    return [int(value) for value in quantized.tolist()], scale


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="data/training_pairs.jsonl")
    parser.add_argument("--out", default="artifacts/lightweight-reranker-int8.json")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--dims", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=2.5)
    parser.add_argument("--l2", type=float, default=0.001)
    args = parser.parse_args()

    pairs_path = Path(args.pairs)
    if pairs_path.exists():
        pairs = [json.loads(line) for line in pairs_path.read_text().splitlines() if line.strip()]
    else:
        pairs = build_examples(args.count, seed=7)

    training_pairs = pairs[: args.count]
    weights, bias = train_logistic_reranker(
        training_pairs,
        dims=args.dims,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
    )
    quantized_weights, scale = quantize_int8(weights)
    payload = {
        "artifact_type": "pairwise-logistic-reranker-int8",
        "description": "INT8-quantized pairwise logistic reranker trained from synthetic positive/negative query-document pairs.",
        "training_examples": len(training_pairs),
        "dims": args.dims,
        "feature": "hashing_embedding(query) * hashing_embedding(document)",
        "scale": scale,
        "bias": bias,
        "weights_int8": quantized_weights,
    }

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(quantized_weights)} int8 weights to {output}")


if __name__ == "__main__":
    main()
