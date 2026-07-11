.PHONY: api frontend test data benchmark docker-build

api:
	uvicorn app.main:app --app-dir backend --reload

frontend:
	cd frontend && npm run dev

test:
	pytest

data:
	python scripts/generate_dataset.py --count 1000 --out data/training_pairs.jsonl

benchmark:
	python scripts/evaluate.py --pairs data/training_pairs.jsonl --out artifacts/metrics.json --limit 100

docker-build:
	docker build -t semantic-reranker-search .
