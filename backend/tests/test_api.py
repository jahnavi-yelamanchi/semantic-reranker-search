from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_document_and_search_flow():
    document = client.post(
        "/documents",
        json={"title": "Remote Work FAQ", "text": "Remote workers receive a monthly workspace stipend."},
    )
    assert document.status_code == 200

    search = client.post("/search", json={"query": "workspace stipend", "mode": "bm25", "top_k": 3})
    assert search.status_code == 200
    payload = search.json()
    assert payload["mode"] == "bm25"
    assert len(payload["results"]) >= 1


def test_metrics_shape():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert {"model", "recall_at_5", "p95_latency_ms", "size_mb"} <= set(response.json()[0])

