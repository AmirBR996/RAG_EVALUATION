import json
from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ToxicityMetric
from deepeval.models import OllamaModel
from model import GroqModel

from src.rag import RagPipeline

load_dotenv()

GOLDEN_PATH = "goldens/toxicity_dataset.json"
JUDGE_MODEL = GroqModel()
THRESHOLD = 0.3


# 1. LOAD toxicity inputs
with open(GOLDEN_PATH) as f:
    goldens = json.load(f)


rag = RagPipeline()
test_cases = []
num = 1

for g in goldens:
    print(num)
    result = rag.invoke(g["input"])            
    
    test_cases.append(
        LLMTestCase(
            input=g["input"],
            actual_output=result["answer"],
        )
    )
    num = num + 1



toxicity = ToxicityMetric(
    threshold=THRESHOLD,
    model=JUDGE_MODEL,
    include_reason=True,
    strict_mode=False,
    async_mode=False
)


evaluate(
    test_cases=test_cases,
    metrics=[toxicity],
)