import sys
from pathlib import Path

from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.models import OllamaModel

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.generator import generate
from evals.harness import load_goldens, summarize_by_metric, print_summary

load_dotenv()

GOLDEN_PATH = ROOT_DIR / "goldens" / "faithfullness_dataset.json"
THRESHOLD = 0.7

model = OllamaModel(
    model="qwen3:8b",
)

def run():
    goldens = load_goldens(GOLDEN_PATH)

    test_cases = [
        LLMTestCase(
            input=g["query"],
            actual_output=generate(g["query"], g["ideal_context"]),
            retrieval_context=g["ideal_context"],
        )
        for g in goldens
    ]

    metrics = [
        FaithfulnessMetric(
            threshold=THRESHOLD,
            model=model,
            include_reason=True,
        ),
        AnswerRelevancyMetric(
            threshold=THRESHOLD,
            model=model,
            include_reason=True,
        ),
    ]

    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
    )

    return summarize_by_metric(result)


if __name__ == "__main__":
    print_summary("generator", run())

