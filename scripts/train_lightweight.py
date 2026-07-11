from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from generate_dataset import build_examples


def tokenize(text: str) -> list[str]:
    import re

    return re.findall(r"[a-z0-9]+", text.lower())


def train_weights(pairs: list[dict[str, str]], max_terms: int) -> dict[str, int]:
    positive = Counter()
    negative = Counter()
    for pair in pairs:
        positive.update(set(tokenize(pair["query"])) & set(tokenize(pair["positive"])))
        negative.update(set(tokenize(pair["query"])) & set(tokenize(pair["negative"])))

    raw: dict[str, float] = {}
    for term in set(positive) | set(negative):
        raw[term] = math.log((positive[term] + 1.0) / (negative[term] + 1.0))

    top = sorted(raw.items(), key=lambda item: abs(item[1]), reverse=True)[:max_terms]
    max_abs = max((abs(value) for _, value in top), default=1.0)
    return {term: int(round((value / max_abs) * 127)) for term, value in top if value != 0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="data/training_pairs.jsonl")
    parser.add_argument("--out", default="artifacts/lightweight-reranker-int8.json")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--max-terms", type=int, default=512)
    args = parser.parse_args()

    pairs_path = Path(args.pairs)
    if pairs_path.exists():
        pairs = [json.loads(line) for line in pairs_path.read_text().splitlines() if line.strip()]
    else:
        pairs = build_examples(args.count, seed=7)

    weights = train_weights(pairs[: args.count], args.max_terms)
    payload = {
        "artifact_type": "lightweight-reranker-int8",
        "description": "INT8 term-weight reranker trained from synthetic positive/negative query-document pairs.",
        "training_examples": min(len(pairs), args.count),
        "weights": weights,
    }

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(weights)} weights to {output}")


if __name__ == "__main__":
    main()

