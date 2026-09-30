from langsmith import traceable
from src.retriever import build_retriever
from src.generator import generate


class RagPipeline:
    def __init__(self):
        self.retriever = build_retriever()

    @traceable(run_type="chain", name="RagPipeline")
    def invoke(self, query: str) -> dict:
        docs = self.retriever.invoke(query)


        context = [doc.page_content for doc in docs]
        answer = generate(query, context)
    

        return {
            "query": query,
            "context": context,
            "answer": answer,
        }


if __name__ == "__main__":
    rag = RagPipeline()
    result = rag.invoke("What is recall and precision")
    print(f"QUERY = ", result["query"])
    print(f"answer = ", result["answer"])


