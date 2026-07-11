from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .metrics import load_benchmark_rows
from .retrieval import BM25Retriever, EmbeddingRetriever, RankedChunk
from .reranker import TrainedInt8Reranker
from .schemas import ArtifactStatus, BenchmarkRow, DocumentIn, DocumentOut, RankingMode, SearchIn, SearchOut, SearchResult
from .store import SearchStore

app = FastAPI(title="Semantic Reranker Search", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = SearchStore()
store.reset_with_examples()
embedding_retriever = EmbeddingRetriever()
trained_reranker = TrainedInt8Reranker()


@app.get("/health")
def health() -> dict[str, str | int]:
    return {"status": "ok", "documents": len(store.documents), "chunks": len(store.chunks)}


@app.post("/documents", response_model=DocumentOut)
def add_document(payload: DocumentIn) -> DocumentOut:
    document, chunks = store.add_document(payload.title, payload.text)
    return DocumentOut(document_id=document.id, title=document.title, chunk_count=len(chunks))


@app.post("/documents/upload", response_model=DocumentOut)
async def upload_document(request: Request) -> DocumentOut:
    body = await request.body()
    text = body.decode("utf-8", errors="ignore")
    title = request.headers.get("x-document-title", "Uploaded document")
    document, chunks = store.add_document(title, text)
    return DocumentOut(document_id=document.id, title=document.title, chunk_count=len(chunks))


@app.post("/search", response_model=SearchOut)
def search(payload: SearchIn) -> SearchOut:
    if payload.mode == RankingMode.bm25:
        results = BM25Retriever(store.chunks).search(payload.query, payload.top_k)
        status = "bm25"
    elif payload.mode == RankingMode.base:
        results = embedding_retriever.search(payload.query, store.chunks, payload.top_k)
        status = embedding_retriever.status
    else:
        results = trained_reranker.search(payload.query, store.chunks, payload.top_k)
        status = trained_reranker.status

    return SearchOut(
        query=payload.query,
        mode=payload.mode,
        artifact_status=status,
        results=[to_result(index, result, payload.mode) for index, result in enumerate(results, start=1)],
    )


@app.get("/metrics", response_model=list[BenchmarkRow])
def metrics() -> list[BenchmarkRow]:
    return load_benchmark_rows()


@app.get("/artifacts", response_model=list[ArtifactStatus])
def artifacts() -> list[ArtifactStatus]:
    return [
        artifact_status("lightweight-int8", trained_reranker.lightweight_path),
        artifact_status("onnx-int8", trained_reranker.artifact_path),
    ]


def artifact_status(name: str, path: Path) -> ArtifactStatus:
    if not path.exists():
        return ArtifactStatus(name=name, path=str(path), present=False, size_mb=None)
    size = path.stat().st_size / (1024 * 1024)
    return ArtifactStatus(
        name=name,
        path=str(path),
        present=True,
        size_mb=round(size, 4) if size < 1 else round(size, 2),
    )


def to_result(rank: int, result: RankedChunk, mode: RankingMode) -> SearchResult:
    return SearchResult(
        rank=rank,
        document_id=result.chunk.document_id,
        chunk_id=result.chunk.id,
        title=store.title_for(result.chunk.document_id),
        snippet=result.chunk.text,
        score=round(result.score, 4),
        method=mode,
    )


static_dir = Path(__file__).resolve().parents[1] / "frontend_dist"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")
