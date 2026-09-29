import os
import sys
import json

from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from model import GroqModel
from src.retriever import build_retriever

load_dotenv()

GOLDEN_PATH = "goldens/retreiever_golden.json"
THRESHOLD = 0.7
JUDGE_MODEL = GroqModel()


def load_goldens(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_by_metric(result):
    summary = {}

    for metric in result:
        name = metric.metric_name

        if name not in summary:
            summary[name] = []

        summary[name].append(metric.score)

    return {
        name: sum(scores) / len(scores)
        for name, scores in summary.items()
    }


def run(retriever):
    goldens = load_goldens(GOLDEN_PATH)

    test_cases = []

    for golden in goldens:
        retrieved = retriever.invoke(golden["query"])

        retrieval_context = [
            doc.page_content
            for doc in retrieved
        ]

        test_cases.append(
            LLMTestCase(
                input=golden["query"],
                expected_output=golden["ideal_answer"],
                actual_output=golden["ideal_answer"],
                retrieval_context=retrieval_context,
            )
        )

    metrics = [
        ContextualRecallMetric(
            threshold=THRESHOLD,
            model=JUDGE_MODEL,
            include_reason=True,
        ),
        ContextualPrecisionMetric(
            threshold=THRESHOLD,
            model=JUDGE_MODEL,
            include_reason=True,
        ),
    ]

    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        hyperparameters={
            "retriever": "reranker",
            "embedding_model": "text-embedding-3-large",
            "chunk_size": 1000,
            "chunk_overlap": 150,
            "top_k": 3,
            "judge_model": "GroqModel",
            "golden_set": GOLDEN_PATH,
        },
    )

    return summarize_by_metric(result)


def run_local():
    retriever = build_retriever()
    return run(retriever)


def print_summary(name, summary):
    print(f"\n{name} evaluation")
    print("-" * 40)

    for metric, score in summary.items():
        print(f"{metric}: {score:.4f}")


if __name__ == "__main__":
    print_summary(
        "retriever",
        run_local()
    )