from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class RankingMode(str, Enum):
    bm25 = "bm25"
    base = "base"
    finetuned = "finetuned"


class DocumentIn(BaseModel):
    title: str = Field(default="Untitled document", min_length=1, max_length=160)
    text: str = Field(min_length=1)


class DocumentOut(BaseModel):
    document_id: str
    title: str
    chunk_count: int


class SearchIn(BaseModel):
    query: str = Field(min_length=1)
    mode: RankingMode = RankingMode.bm25
    top_k: int = Field(default=5, ge=1, le=20)


class SearchResult(BaseModel):
    rank: int
    document_id: str
    chunk_id: str
    title: str
    snippet: str
    score: float
    method: RankingMode


class SearchOut(BaseModel):
    query: str
    mode: RankingMode
    artifact_status: str
    results: list[SearchResult]


class BenchmarkRow(BaseModel):
    model: str
    recall_at_5: float
    p95_latency_ms: float
    size_mb: float | None = None


class ArtifactStatus(BaseModel):
    name: str
    path: str
    present: bool
    size_mb: float | None = None
