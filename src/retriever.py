import os
import re
import glob
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "chroma_store"


def load_transcripts():
    docs = []

    for path in glob.glob(f"{DATA_DIR}/*.vtt"):
        lines = []

        with open(path, encoding="utf-8") as file:
            for line in file:
                line = line.strip()

                if not line or line == "WEBVTT" or "-->" in line:
                    continue

                lines.append(line)

        text = " ".join(lines)

        match = re.search(r"Session[ _]*(\d+)", path)
        session = match.group(1) if match else os.path.basename(path)

        docs.append(
            Document(
                page_content=text,
                metadata={"session": session},
            )
        )

    return docs


def load_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if DB_DIR.exists() and any(DB_DIR.iterdir()):
        return Chroma(
            persist_directory=str(DB_DIR),
            embedding_function=embeddings,
        )

    docs = load_transcripts()

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    ).split_documents(docs)

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(DB_DIR),
    )


def build_retriever():
    return load_store().as_retriever(
        search_kwargs={"k": 5}
    )


if __name__ == "__main__":
    retriever = build_retriever()
    results = retriever.invoke("what is llm evaluation?")

    for result in results:
        print(
            f"[Session {result.metadata['session']}] "
            f"{result.page_content[:150]}...\n"
        )

