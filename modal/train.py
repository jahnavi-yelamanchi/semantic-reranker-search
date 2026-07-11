from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("semantic-reranker-search")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "datasets==2.20.0",
        "sentence-transformers==3.0.1",
        "optimum[onnxruntime]==1.21.2",
        "onnxruntime==1.18.1",
        "scikit-learn==1.5.1",
    )
)

volume = modal.Volume.from_name("semantic-reranker-artifacts", create_if_missing=True)


@app.function(image=image, gpu="T4", timeout=7200, volumes={"/artifacts": volume})
def train_and_export(max_examples: int = 1000, epochs: int = 1) -> dict[str, str]:
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer, losses
    from sentence_transformers.evaluation import InformationRetrievalEvaluator
    from torch.utils.data import DataLoader

    rows = build_synthetic_pairs(max_examples)
    dataset = Dataset.from_list(rows)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    train_examples = []
    for row in dataset:
        from sentence_transformers import InputExample

        train_examples.append(InputExample(texts=[row["query"], row["positive"], row["negative"]]))

    dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    loss = losses.TripletLoss(model)
    model.fit(train_objectives=[(dataloader, loss)], epochs=epochs, warmup_steps=20, show_progress_bar=True)

    output_dir = Path("/artifacts/model")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(output_dir))

    metrics = evaluate_model(model, rows[: min(200, len(rows))])
    (Path("/artifacts") / "modal_metrics.json").write_text(json.dumps(metrics, indent=2))

    export_onnx(output_dir, Path("/artifacts/model.onnx"), Path("/artifacts/model-int8.onnx"))
    volume.commit()
    return {
        "model_dir": "/artifacts/model",
        "onnx": "/artifacts/model.onnx",
        "int8": "/artifacts/model-int8.onnx",
        "metrics": "/artifacts/modal_metrics.json",
    }


@app.local_entrypoint()
def main(max_examples: int = 1000, epochs: int = 1) -> None:
    result = train_and_export.remote(max_examples=max_examples, epochs=epochs)
    print(result)


def build_synthetic_pairs(count: int) -> list[dict[str, str]]:
    topics = [
        ("backup", "CloudSync backs up files every 15 minutes and keeps deleted files for 30 days."),
        ("sso", "Enterprise admins can enable SAML SSO, audit logs, and role-based access controls."),
        ("billing", "Invoices are available from the billing settings page after each renewal."),
        ("refunds", "Refund requests are accepted within 14 days when usage is below the plan limit."),
        ("ml engineer", "The ML engineer role requires Python, retrieval systems, FastAPI, and model evaluation."),
        ("frontend", "The frontend engineer role requires React, TypeScript, accessibility, and design systems."),
    ]
    rows = []
    for index in range(count):
        topic, positive = topics[index % len(topics)]
        _, negative = topics[(index + 2) % len(topics)]
        rows.append(
            {
                "query": f"Which document has information about {topic}?",
                "positive": positive,
                "negative": negative,
            }
        )
    return rows


def evaluate_model(model, rows: list[dict[str, str]]) -> dict[str, float]:
    queries = {str(index): row["query"] for index, row in enumerate(rows)}
    corpus = {str(index): row["positive"] for index, row in enumerate(rows)}
    relevant_docs = {str(index): {str(index)} for index in range(len(rows))}
    evaluator = InformationRetrievalEvaluator(queries, corpus, relevant_docs, show_progress_bar=False)
    scores = evaluator(model)
    return {key: float(value) for key, value in scores.items() if isinstance(value, (float, int))}


def export_onnx(model_dir: Path, onnx_path: Path, int8_path: Path) -> None:
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    ort_model = ORTModelForFeatureExtraction.from_pretrained(model_dir, export=True)
    export_dir = onnx_path.parent / "onnx_export"
    export_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(export_dir)
    ort_model.save_pretrained(export_dir)

    exported = export_dir / "model.onnx"
    exported.replace(onnx_path)
    quantize_dynamic(str(onnx_path), str(int8_path), weight_type=QuantType.QInt8)
