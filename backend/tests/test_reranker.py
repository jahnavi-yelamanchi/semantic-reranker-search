import json

from app.reranker import TrainedInt8Reranker
from app.store import SearchStore


def test_lightweight_reranker_uses_int8_weights(tmp_path):
    artifact = tmp_path / "lightweight-reranker-int8.json"
    artifact.write_text(json.dumps({"weights": {"stipend": 127}}))

    store = SearchStore()
    target, _ = store.add_document("Benefits", "Remote workers receive a workspace stipend.")
    store.add_document("Billing", "Invoices and plan renewals are in billing settings.")

    reranker = TrainedInt8Reranker(artifact_path=str(tmp_path / "missing.onnx"), lightweight_path=str(artifact))
    results = reranker.search("workspace stipend", store.chunks, top_k=1)

    assert reranker.status.startswith("lightweight-int8 artifact")
    assert results[0].chunk.document_id == target.id
