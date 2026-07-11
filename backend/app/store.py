from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from .chunking import Chunk, chunk_document


@dataclass
class Document:
    id: str
    title: str
    text: str


@dataclass
class SearchStore:
    documents: dict[str, Document] = field(default_factory=dict)
    chunks: list[Chunk] = field(default_factory=list)

    def add_document(self, title: str, text: str) -> tuple[Document, list[Chunk]]:
        document = Document(id=str(uuid.uuid4()), title=title.strip() or "Untitled document", text=text)
        chunks = chunk_document(document.id, text)
        self.documents[document.id] = document
        self.chunks.extend(chunks)
        return document, chunks

    def reset_with_examples(self) -> None:
        self.documents.clear()
        self.chunks.clear()
        examples = [
            (
                "CloudSync FAQ",
                "CloudSync backs up product photos, invoices, and support exports every 15 minutes. "
                "The starter plan supports 50 GB. Admins can restore deleted files for 30 days. "
                "Enterprise customers can enable SSO, audit logs, and region-specific storage.",
            ),
            (
                "SupportBot Product Guide",
                "SupportBot answers customer questions from uploaded help center articles. "
                "It supports escalation rules, confidence thresholds, and multilingual responses. "
                "Teams can review unresolved questions and add new FAQ entries from the dashboard.",
            ),
            (
                "ML Engineer Job Listing",
                "We are hiring a machine learning engineer to build retrieval and ranking systems. "
                "The role requires Python, FastAPI, vector search, model evaluation, and deployment experience. "
                "Experience with ONNX, quantization, and cloud GPU training is a plus.",
            ),
        ]
        for title, text in examples:
            self.add_document(title, text)

    def title_for(self, document_id: str) -> str:
        document = self.documents.get(document_id)
        return document.title if document else "Unknown document"

