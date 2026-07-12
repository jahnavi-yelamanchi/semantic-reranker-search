# Semantic Reranker Search

A recruiter-facing semantic search project that compares keyword retrieval, base embedding search, and a trained INT8 reranker.

Users can paste or upload product docs, FAQs, or job listings, then query the corpus and compare ranked results across retrieval modes.

## What This Demonstrates

- **Train:** synthetic positive/negative query-document pairs plus a pairwise logistic reranker.
- **Optimize:** INT8-quantized trained weights with an ONNX/Modal path documented for heavier model export.
- **Deploy:** FastAPI, React, Docker, and Render.
- **Evaluate:** Recall@5, P95 latency, and model size benchmarks.
- **Integrate:** document chunking, search API, benchmark UI, and model artifact loading.

## Architecture

```mermaid
flowchart LR
  A[React workbench] --> B[FastAPI API]
  B --> C[Document chunk store]
  B --> D[BM25 retriever]
  B --> E[Base embedding retriever]
  B --> F[Trained INT8 reranker]
  G[Local lightweight training] --> H[INT8 JSON artifact]
  H --> F
  I[Modal training job] -. stretch .-> J[ONNX INT8 artifact]
  J -. optional .-> F
```

## Project Structure

```text
backend/app/          FastAPI app, retrieval, metrics, chunking
backend/tests/        Unit and API tests
frontend/src/         React search workbench
modal/train.py        Remote Modal training/export/quantization job
scripts/              Dataset generation and benchmark scripts
artifacts/            Downloaded model and benchmark artifacts
data/                 Generated training pairs
```

## Local Development

Install backend dependencies:

```bash
cd /Users/jahnaviyelamanchi/Documents/semantic-reranker-search
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

Run the API:

```bash
uvicorn app.main:app --app-dir backend --reload
```

Run the React app:

```bash
cd frontend
npm install
npm run dev
```

If the frontend is running separately from the API, set `VITE_API_BASE=http://localhost:8000` in `frontend/.env.local`.

Shortcut commands:

```bash
make api
make frontend
make test
make data
make train-lightweight
make benchmark
make modal-train
make modal-download
```

By default, the API starts with example documents so the UI can search immediately. The checked-in lightweight INT8 artifact powers `finetuned` mode without requiring Modal.

## Generate Data

```bash
python scripts/generate_dataset.py --count 1000 --out data/training_pairs.jsonl
```

The generated dataset uses product-doc, FAQ, and job-listing examples with positive and negative query-document pairs.

## Train

The fast local path trains a pairwise logistic reranker and quantizes its weights to INT8 in seconds:

```bash
make data
make train-lightweight
make benchmark
```

This creates `artifacts/lightweight-reranker-int8.json`, which the `finetuned` API mode uses immediately. The artifact stores 384 INT8 weights, a quantization scale, and a bias term trained from positive/negative query-document pairs.

Modal remains the heavier remote ONNX export path.

```bash
pip install modal
modal setup
make modal-train
```

The Modal job:

1. builds synthetic query-document pairs,
2. fine-tunes `sentence-transformers/all-MiniLM-L6-v2`,
3. saves model artifacts to a Modal Volume,
4. exports ONNX,
5. writes `model-int8.onnx` with dynamic INT8 quantization.

Download artifacts from the Modal Volume into `artifacts/` before benchmarking or deployment:

```bash
make modal-download
```

Expected artifact paths:

```text
artifacts/lightweight-reranker-int8.json
artifacts/model-int8.onnx
artifacts/metrics.json
```

## Benchmark

```bash
python scripts/generate_dataset.py --count 1000
python scripts/evaluate.py --pairs data/training_pairs.jsonl --out artifacts/metrics.json --limit 100
```

The benchmark script writes rows consumed by both the API and UI:

| Model | Recall@5 | P95 latency | Size |
| --- | ---: | ---: | ---: |
| BM25 | 0.200 | 0.49 ms | - |
| Base embedding model | 0.210 | 4.09 ms | - |
| Fine-tuned INT8 reranker | 0.240 | 5.35 ms | < 0.01 MB |

These are baseline numbers from `artifacts/metrics.json` after running the fast local lightweight training path.

## Test

```bash
pytest
```

Tests cover chunking, BM25 retrieval, embedding retrieval fallback, API health, document ingestion, search, and metrics shape.

## Docker

```bash
docker build -t semantic-reranker-search .
docker run --rm -p 8000:8000 semantic-reranker-search
```

The Docker image builds the React app, serves it from FastAPI, and exposes `/health` for Render.

## Render Deployment

1. Push this repo to GitHub.
2. Create a Render Blueprint from `render.yaml`, or create a Docker web service manually.
3. Set the health check path to `/health`.
4. The lightweight INT8 artifact is already included. Add `artifacts/model-int8.onnx` later if you complete the heavier ONNX export path.

## API

### `POST /documents`

```json
{
  "title": "Product FAQ",
  "text": "Refunds are available within 14 days..."
}
```

### `POST /search`

```json
{
  "query": "How do refunds work?",
  "mode": "bm25",
  "top_k": 5
}
```

Modes: `bm25`, `base`, `finetuned`.

### `GET /metrics`

Returns benchmark rows for the UI and README table.

### `GET /artifacts`

Returns model artifact availability and sizes for the lightweight INT8 and optional ONNX INT8 artifacts.

### `GET /health`

Render health check endpoint.

## Current Scope

This is a one-day MVP. It intentionally skips authentication, payments, teams, persistent multi-user storage, and complex ingestion pipelines.
