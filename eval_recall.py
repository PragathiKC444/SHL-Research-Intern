from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import requests


def normalize_name(value: str) -> str:
    return " ".join(value.lower().split())


def extract_expected_names(trace: dict[str, Any]) -> set[str]:
    candidates = [
        "expected_shortlist",
        "expected_assessments",
        "relevant_assessments",
        "ground_truth",
    ]
    for key in candidates:
        value = trace.get(key)
        if isinstance(value, list):
            names = {normalize_name(item) for item in value if isinstance(item, str) and item.strip()}
            if names:
                return names
    return set()


def compute_recall_at_k(expected: set[str], predicted: list[str], k: int = 10) -> float:
    if not expected:
        return 0.0
    top_k = {normalize_name(item) for item in predicted[:k]}
    hits = len(expected & top_k)
    return hits / len(expected)


def call_chat(base_url: str, messages: list[dict[str, str]]) -> dict[str, Any]:
    response = requests.post(f"{base_url.rstrip('/')}/chat", json={"messages": messages}, timeout=30)
    response.raise_for_status()
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute mean Recall@10 for conversation traces.")
    parser.add_argument("--base-url", required=True, help="Public API base URL, e.g. https://...onrender.com")
    parser.add_argument("--traces", required=True, help="Path to JSON file with traces")
    args = parser.parse_args()

    traces_path = Path(args.traces)
    traces = json.loads(traces_path.read_text(encoding="utf-8"))
    if not isinstance(traces, list):
        raise ValueError("Traces file must contain a JSON array.")

    recalls: list[float] = []
    evaluated = 0

    for index, trace in enumerate(traces, start=1):
        if not isinstance(trace, dict):
            continue

        messages = trace.get("messages", [])
        expected = extract_expected_names(trace)

        if not isinstance(messages, list) or not expected:
            continue

        result = call_chat(args.base_url, messages)
        rec_names = [item.get("name", "") for item in result.get("recommendations", []) if isinstance(item, dict)]
        recall = compute_recall_at_k(expected, rec_names, k=10)
        recalls.append(recall)
        evaluated += 1
        print(f"Trace {index}: Recall@10 = {recall:.3f}")

    mean_recall = sum(recalls) / len(recalls) if recalls else 0.0
    print(f"Evaluated traces: {evaluated}")
    print(f"Mean Recall@10: {mean_recall:.3f}")


if __name__ == "__main__":
    main()
