from langsmith import traceable

from src.retriever import build_retriever
from src.generator import generate


class RagPipeline:
    def __init__(self):
        print("Creating retriever...")
        self.retriever = build_retriever()
        print("Retriever ready")

    @traceable(run_type="chain", name="RagPipeline")
    def invoke(self, query: str) -> dict:
        print("Starting retrieval...")

        docs = self.retriever.invoke(query)

        print(f"Retrieved {len(docs)} documents")
        print("Starting generation...")

        context = [doc.page_content for doc in docs]
        answer = generate(query, context)

        print("Generation finished")

        return {
            "query": query,
            "context": context,
            "answer": answer,
        }


if __name__ == "__main__":
    print("Starting RAG pipeline...")

    rag = RagPipeline()

    print("Running query...")

    result = rag.invoke("Why do we need golden datasets?")

    print("QUERY:", result["query"])
    print("ANSWER:", result["answer"])
    print("\nCONTEXT CHUNKS:")

    for i, chunk in enumerate(result["context"]):
        print(f"[{i}] {chunk[:120]}...")
