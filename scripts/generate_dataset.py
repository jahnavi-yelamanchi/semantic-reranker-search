from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


DOMAINS = {
    "product_docs": [
        ("backup", "CloudSync backs up files every 15 minutes and keeps deleted files for 30 days."),
        ("sso", "Enterprise admins can enable SAML SSO, audit logs, and role-based access controls."),
        ("billing", "Invoices are available from the billing settings page after each renewal."),
        ("permissions", "Workspace owners can grant viewer, editor, or admin permissions to teammates."),
    ],
    "faqs": [
        ("refunds", "Refund requests are accepted within 14 days when usage is below the plan limit."),
        ("password", "Users can reset a password from the sign-in page using a verified email address."),
        ("uploads", "PDF, TXT, and Markdown files can be uploaded from the knowledge base screen."),
        ("support", "Premium plans include priority email support with a four-hour response target."),
    ],
    "jobs": [
        ("ml engineer", "The ML engineer role requires Python, retrieval systems, FastAPI, and model evaluation."),
        ("frontend", "The frontend engineer role requires React, TypeScript, accessibility, and design systems."),
        ("data", "The data analyst role requires SQL, dashboards, experimentation, and stakeholder communication."),
        ("devops", "The platform engineer role requires Docker, observability, CI/CD, and cloud deployment."),
    ],
}

QUERY_TEMPLATES = [
    "How do I find information about {topic}?",
    "Which document explains {topic}?",
    "What does the company say about {topic}?",
    "Show me details for {topic}.",
]


def build_examples(count: int, seed: int) -> list[dict[str, str]]:
    random.seed(seed)
    all_docs = [(domain, topic, text) for domain, entries in DOMAINS.items() for topic, text in entries]
    rows = []
    for index in range(count):
        domain, topic, positive = random.choice(all_docs)
        negative_pool = [doc for doc in all_docs if doc[1] != topic]
        _, _, negative = random.choice(negative_pool)
        query = random.choice(QUERY_TEMPLATES).format(topic=topic)
        rows.append(
            {
                "id": f"pair-{index}",
                "domain": domain,
                "query": query,
                "positive": positive,
                "negative": negative,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="data/training_pairs.jsonl")
    args = parser.parse_args()

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = build_examples(args.count, args.seed)
    output.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    print(f"wrote {len(rows)} examples to {output}")


if __name__ == "__main__":
    main()

