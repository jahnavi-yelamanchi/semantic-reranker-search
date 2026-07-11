from app.chunking import chunk_document, tokenize


def test_tokenize_normalizes_words_and_numbers():
    assert tokenize("FastAPI, ONNX INT8!") == ["fastapi", "onnx", "int8"]


def test_chunk_document_uses_overlap():
    text = " ".join(f"word{i}" for i in range(20))
    chunks = chunk_document("doc-1", text, max_words=10, overlap=2)
    assert len(chunks) == 3
    assert chunks[0].id == "doc-1::chunk-0"
    assert chunks[1].text.startswith("word8")

