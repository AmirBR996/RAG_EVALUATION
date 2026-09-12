import sys
import os

from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from model import GroqModel

load_dotenv()

case_1 = LLMTestCase(
    input="What is the capital of France?",
    actual_output="The capital of France is Paris."
)

case_2 = LLMTestCase(
    input="What is the capital of France?",
    actual_output="France is a beautiful country famous for its food and wine."
)

metric = AnswerRelevancyMetric(
    threshold=0.7,
    model=GroqModel(),
    include_reason=True
)

evaluate(
    test_cases=[case_1, case_2],
    metrics=[metric]
)