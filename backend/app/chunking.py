from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    text: str
    position: int


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def chunk_document(document_id: str, text: str, max_words: int = 120, overlap: int = 24) -> list[Chunk]:
    words = normalize_text(text).split()
    if not words:
        return []

    chunks: list[Chunk] = []
    start = 0
    position = 0
    step = max(1, max_words - overlap)

    while start < len(words):
        end = min(len(words), start + max_words)
        chunk_text = " ".join(words[start:end])
        chunks.append(
            Chunk(
                id=f"{document_id}::chunk-{position}",
                document_id=document_id,
                text=chunk_text,
                position=position,
            )
        )
        if end == len(words):
            break
        start += step
        position += 1

    return chunks

