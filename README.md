# RAG Evaluation Framework

A comprehensive framework for building and evaluating a Retrieval-Augmented Generation (RAG) pipeline, specifically designed for a course on LLM evaluations.

## 🚀 Overview

This repository implements a complete RAG pipeline that leverages a vector database for context retrieval and an LLM for grounded answer generation. It includes a rigorous evaluation suite to measure performance across multiple dimensions including correctness, faithfulness, safety, and operational metrics.

## 🏗️ Architecture

The system consists of three primary components:

1.  **Retriever (`src/retriever.py`)**: 
    - Loads course transcripts (`.vtt` files) from the `data/` directory.
    - Uses `RecursiveCharacterTextSplitter` for document chunking.
    - Employs `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`) and `ChromaDB` for efficient vector storage and similarity search.
2.  **Generator (`src/generator.py`)**:
    - Uses `ChatOllama` with the `gemma4:31b-cloud` model.
    - Implements a strict system prompt to ensure answers are grounded **only** in the provided context (preventing hallucinations).
    - Includes safety guardrails and pedagogical guidelines for a teaching assistant persona.
3.  **Pipeline (`src/rag.py`)**: 
    - Orchestrates the flow from query $\rightarrow$ retrieval $\rightarrow$ generation.
    - Integrated with **LangSmith** for tracing and observability.

## 🧪 Evaluation Suite

The core of this repository is its extensive evaluation capabilities located in `evals/` and `regression_eval/`.

### Evaluation Dimensions
- **Correctness**: Measuring the accuracy of the generated answers against goldens.
- **Faithfulness**: Ensuring the answer is derived solely from the retrieved context.
- **Safety**: Testing for toxicity, leakage of system prompts, and sensitive data exposure.
- **Operational**: Tracking latency and cost.
- **Retrieval**: Evaluating the quality of retrieved documents.

### Regression Testing
The `regression_eval/` directory provides tools to run comprehensive test suites and compare results across different pipeline versions, ensuring that improvements in one area don't cause regressions in another.

## 🛠️ Getting Started

### Prerequisites
- Python 3.10+
- Ollama (installed and running with `gemma4:31b-cloud` pulled)
- A `.env` file with necessary API keys (e.g., LangSmith)

### Installation
```bash
pip install -r requirement.txt
```

### Running the Pipeline
You can run the RAG pipeline directly:
```bash
python src/rag.py
```

### Running Evaluations
Explore the `evals/` directory to run specific tests, or use the regression suite:
```bash
python regression_eval/run_suite.py
```

## 📁 Project Structure
- `src/`: Core RAG implementation.
- `data/`: Course transcripts used as the knowledge base.
- `goldens/`: Ground-truth datasets for evaluation.
- `evals/`: Metric definitions and evaluation harness.
- `regression_eval/`: Tools for running and comparing evaluation suites.
- `chroma_store/`: Persisted vector database.
