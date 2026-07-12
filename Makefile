.PHONY: api frontend test smoke data train-lightweight benchmark modal-train modal-download docker-build

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

train-lightweight:
	python scripts/train_lightweight.py --pairs data/training_pairs.jsonl --out artifacts/lightweight-reranker-int8.json

benchmark:
	python scripts/evaluate.py --pairs data/training_pairs.jsonl --out artifacts/metrics.json --limit 100

modal-train:
	modal run modal/train.py --max-examples 2000 --epochs 2

modal-download:
	mkdir -p artifacts
	modal volume get semantic-reranker-artifacts /model-int8.onnx artifacts/model-int8.onnx
	mkdir -p artifacts/tokenizer
	modal volume get semantic-reranker-artifacts /tokenizer/vocab.txt artifacts/tokenizer/vocab.txt
	modal volume get semantic-reranker-artifacts /tokenizer/tokenizer_config.json artifacts/tokenizer/tokenizer_config.json
	modal volume get semantic-reranker-artifacts /tokenizer/tokenizer.json artifacts/tokenizer/tokenizer.json
	modal volume get semantic-reranker-artifacts /tokenizer/special_tokens_map.json artifacts/tokenizer/special_tokens_map.json
	modal volume get semantic-reranker-artifacts /modal_metrics.json artifacts/modal_metrics.json

docker-build:
	docker build -t semantic-reranker-search .
