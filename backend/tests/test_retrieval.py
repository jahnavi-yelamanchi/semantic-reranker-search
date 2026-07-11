from app.retrieval import BM25Retriever, EmbeddingRetriever
from app.store import SearchStore


def test_bm25_ranks_matching_chunk_first():
    store = SearchStore()
    doc, chunks = store.add_document("Support", "Password reset links expire after 30 minutes.")
    store.add_document("Other", "Invoices are available in billing settings.")

    results = BM25Retriever(store.chunks).search("password reset", top_k=1)

    assert results[0].chunk.document_id == doc.id
    assert results[0].score > 0


def test_embedding_retriever_returns_requested_count():
    store = SearchStore()
    store.add_document("One", "React frontend accessibility and design systems.")
    store.add_document("Two", "Docker deployment and FastAPI service health checks.")

    results = EmbeddingRetriever().search("frontend design", store.chunks, top_k=2)

    assert len(results) == 2

