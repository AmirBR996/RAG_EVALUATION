import os
import glob
import json
import random

from deepeval.synthesizer import Synthesizer
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from model import GroqModel


def load_chunks():
    texts = []

    for path in glob.glob("data/*.vtt"):
        with open(path, encoding="utf-8") as f:
            lines = [
                line.strip()
                for line in f
                if line.strip()
                and line.strip() != "WEBVTT"
                and "-->" not in line
            ]

        texts.append(" ".join(lines))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    return splitter.split_text("\n\n".join(texts))


chunks = load_chunks()

if not chunks:
    raise ValueError("No transcript chunks found in data/*.vtt")

sample = random.sample(
    chunks,
    min(15, len(chunks))
)

contexts = [[chunk] for chunk in sample]

groq_model = GroqModel()

synthesizer = Synthesizer(
    model=groq_model
)

goldens = synthesizer.generate_goldens_from_contexts(
    contexts=contexts,
    include_expected_output=True,
    max_goldens_per_context=1
)

rows = []

for i, golden in enumerate(goldens, 1):
    rows.append({
        "id": f"g{i:03d}",
        "query": golden.input,
        "ideal_answer": golden.expected_output,
        "context": golden.context,
        "source": "TODO-verify"
    })

os.makedirs("goldens", exist_ok=True)

output_path = "goldens/retriever_deepeval_goldens.json"

with open(
    output_path,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        rows,
        f,
        indent=2,
        ensure_ascii=False
    )

print(f"Generated {len(rows)} goldens.")
print(f"Saved to {output_path}")