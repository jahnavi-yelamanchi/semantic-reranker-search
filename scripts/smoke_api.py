from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from fastapi.testclient import TestClient

from app.main import app


def main() -> None:
    client = TestClient(app)
    health = client.get("/health")
    health.raise_for_status()

    document = client.post(
        "/documents",
        json={
            "title": "Smoke Test FAQ",
            "text": "Remote employees receive a monthly workspace stipend and can expense ergonomic equipment.",
        },
    )
    document.raise_for_status()

    search = client.post(
        "/search",
        json={"query": "workspace stipend for remote employees", "mode": "bm25", "top_k": 3},
    )
    search.raise_for_status()
    payload = search.json()
    if not payload["results"]:
        raise SystemExit("search returned no results")

    metrics = client.get("/metrics")
    metrics.raise_for_status()
    if not metrics.json():
        raise SystemExit("metrics returned no rows")

    print("smoke ok")


if __name__ == "__main__":
    main()
