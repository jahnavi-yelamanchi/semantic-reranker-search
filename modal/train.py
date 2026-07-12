from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("semantic-reranker-search")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "datasets==2.20.0",
        "accelerate==0.33.0",
        "sentence-transformers==3.0.1",
        "optimum[onnxruntime]==1.21.2",
        "onnxscript==0.7.1",
        "onnx-ir==0.2.1",
        "onnxruntime==1.18.1",
        "scikit-learn==1.5.1",
    )
)

volume = modal.Volume.from_name("semantic-reranker-artifacts", create_if_missing=True)


@app.function(image=image, gpu="T4", timeout=7200, volumes={"/artifacts": volume})
def train_and_export(max_examples: int = 1000, epochs: int = 1) -> dict[str, str]:
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer, losses
    from torch.utils.data import DataLoader

    rows = build_synthetic_pairs(max_examples)
    dataset = Dataset.from_list(rows)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    train_examples = []
    for row in dataset:
        from sentence_transformers import InputExample

        train_examples.append(InputExample(texts=[row["query"], row["positive"]], label=1.0))
        train_examples.append(InputExample(texts=[row["query"], row["negative"]], label=0.0))

    dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
    loss = losses.CosineSimilarityLoss(model)
    model.fit(train_objectives=[(dataloader, loss)], epochs=epochs, warmup_steps=20, show_progress_bar=True)

    output_dir = Path("/artifacts/model")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(output_dir))

    metrics = evaluate_model(model, rows[: min(200, len(rows))])
    (Path("/artifacts") / "modal_metrics.json").write_text(json.dumps(metrics, indent=2))

    export_onnx(model, Path("/artifacts/model.onnx"), Path("/artifacts/model-int8.onnx"), Path("/artifacts/tokenizer"))
    volume.commit()
    return {
        "model_dir": "/artifacts/model",
        "tokenizer": "/artifacts/tokenizer",
        "onnx": "/artifacts/model.onnx",
        "int8": "/artifacts/model-int8.onnx",
        "metrics": "/artifacts/modal_metrics.json",
    }


@app.local_entrypoint()
def main(max_examples: int = 1000, epochs: int = 1) -> None:
    result = train_and_export.remote(max_examples=max_examples, epochs=epochs)
    print(result)


def build_synthetic_pairs(count: int) -> list[dict[str, str]]:
    import random

    topics = [
        ("product_docs", "backup", "CloudSync backs up files every 15 minutes and keeps deleted files for 30 days."),
        ("product_docs", "sso", "Enterprise admins can enable SAML SSO, audit logs, and role-based access controls."),
        ("product_docs", "billing", "Invoices are available from the billing settings page after each renewal."),
        ("product_docs", "permissions", "Workspace owners can grant viewer, editor, or admin permissions to teammates."),
        ("faqs", "refunds", "Refund requests are accepted within 14 days when usage is below the plan limit."),
        ("faqs", "password", "Users can reset a password from the sign-in page using a verified email address."),
        ("faqs", "uploads", "PDF, TXT, and Markdown files can be uploaded from the knowledge base screen."),
        ("faqs", "support", "Premium plans include priority email support with a four-hour response target."),
        ("jobs", "ml engineer", "The ML engineer role requires Python, retrieval systems, FastAPI, and model evaluation."),
        ("jobs", "frontend", "The frontend engineer role requires React, TypeScript, accessibility, and design systems."),
        ("jobs", "data", "The data analyst role requires SQL, dashboards, experimentation, and stakeholder communication."),
        ("jobs", "devops", "The platform engineer role requires Docker, observability, CI/CD, and cloud deployment."),
    ]
    query_templates = [
        "How do I find information about {topic}?",
        "Which document explains {topic}?",
        "What does the company say about {topic}?",
        "Show me details for {topic}.",
    ]
    random.seed(7)
    rows = []
    for index in range(count):
        domain, topic, positive = random.choice(topics)
        negative_pool = [item for item in topics if item[1] != topic]
        _, _, negative = random.choice(negative_pool)
        query = random.choice(query_templates).format(topic=topic)
        rows.append(
            {
                "domain": domain,
                "query": query,
                "positive": positive,
                "negative": negative,
            }
        )
    return rows


def evaluate_model(model, rows: list[dict[str, str]]) -> dict[str, float]:
    from sentence_transformers.evaluation import InformationRetrievalEvaluator

    queries = {str(index): row["query"] for index, row in enumerate(rows)}
    corpus = {str(index): row["positive"] for index, row in enumerate(rows)}
    relevant_docs = {str(index): {str(index)} for index in range(len(rows))}
    evaluator = InformationRetrievalEvaluator(queries, corpus, relevant_docs, show_progress_bar=False)
    scores = evaluator(model)
    return {key: float(value) for key, value in scores.items() if isinstance(value, (float, int))}


def export_onnx(sentence_model, onnx_path: Path, int8_path: Path, tokenizer_dir: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoTokenizer
    import torch

    transformer = sentence_model._first_module()
    hf_model_dir = onnx_path.parent / "hf_model"
    hf_model_dir.mkdir(parents=True, exist_ok=True)
    transformer.auto_model.save_pretrained(hf_model_dir)
    transformer.tokenizer.save_pretrained(hf_model_dir)

    tokenizer = AutoTokenizer.from_pretrained(str(hf_model_dir))
    export_dir = onnx_path.parent / "onnx_export"
    export_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(export_dir)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_dir)

    auto_model = transformer.auto_model.eval().cpu()
    dummy = tokenizer(
        ["Which document has information about billing?"],
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    input_names = [name for name in ["input_ids", "attention_mask", "token_type_ids"] if name in dummy]

    class FeatureExtractionWrapper(torch.nn.Module):
        def __init__(self, wrapped_model):
            super().__init__()
            self.wrapped_model = wrapped_model

        def forward(self, *inputs):
            kwargs = dict(zip(input_names, inputs))
            return self.wrapped_model(**kwargs).last_hidden_state

    dynamic_axes = {
        name: {0: "batch_size", 1: "sequence_length"}
        for name in input_names
    }
    dynamic_axes["last_hidden_state"] = {0: "batch_size", 1: "sequence_length"}
    wrapper = FeatureExtractionWrapper(auto_model).eval()
    torch.onnx.export(
        wrapper,
        tuple(dummy[name] for name in input_names),
        str(onnx_path),
        input_names=input_names,
        output_names=["last_hidden_state"],
        dynamic_axes=dynamic_axes,
        opset_version=14,
        dynamo=False,
    )
    quantize_dynamic(str(onnx_path), str(int8_path), weight_type=QuantType.QInt8)
