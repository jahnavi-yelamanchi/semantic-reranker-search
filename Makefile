.PHONY: api frontend test smoke data benchmark modal-train modal-download docker-build

api:
	uvicorn app.main:app --app-dir backend --reload

frontend:
	cd frontend && npm run dev

test:
	pytest

smoke:
	python scripts/smoke_api.py

data:
	python scripts/generate_dataset.py --count 1000 --out data/training_pairs.jsonl

benchmark:
	python scripts/evaluate.py --pairs data/training_pairs.jsonl --out artifacts/metrics.json --limit 100

modal-train:
	modal run modal/train.py --max-examples 1000 --epochs 1

modal-download:
	mkdir -p artifacts
	modal volume get semantic-reranker-artifacts /model-int8.onnx artifacts/model-int8.onnx
	modal volume get semantic-reranker-artifacts /modal_metrics.json artifacts/modal_metrics.json

docker-build:
	docker build -t semantic-reranker-search .
