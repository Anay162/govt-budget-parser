# SLED Intelligence Engine (Local RAG)

An enterprise-grade, locally hosted Retrieval-Augmented Generation (RAG) pipeline designed to ingest, parse, and query complex unstructured government documents (e.g., Municipal Budgets, City Council Minutes, ACFRs).

## The Problem This Solves
Standard RAG architectures fail when applied to government contracting and SLED (State, Local, and Education) data due to three critical flaws:
1. **Financial Table Destruction:** Naive text chunking destroys the row/column relationships in budget tables.
2. **Hallucinations:** Generic LLMs often guess numerical values when unsure, which is catastrophic for financial intelligence.
3. **Cross-Contamination:** Searching across a unified vector database without hard routing leads to pulling data from the wrong municipality or fiscal year.

## The Architecture
This pipeline solves these issues through a strictly controlled, multi-stage architecture:

* **Layout-Aware Parsing (`unstructured` + YOLOX):** Instead of flattening the PDFs, this engine uses a vision model to identify bounding boxes around financial tables, extracting and preserving them as raw HTML.
* **Semantic Chunking:** Text is chunked logically by section headers, not arbitrary character counts, ensuring context is never severed from its source.
* **Hard Metadata Routing (ChromaDB):** During ingestion, documents are tagged with immutable metadata (`municipality`, `year`, `document_type`). The retriever uses `$and` and `$in` operators to strictly filter the vector space *before* running cosine similarity, preventing multi-tenant data bleed.
* **Strict-Prompt Generation (Local Llama 3):** The 8B LLM is prompted to read HTML table coordinates mathematically and is explicitly instructed to refuse to answer rather than hallucinate if data is missing.

## Tech Stack
* **Frontend:** Streamlit
* **Parsing/Chunking:** Unstructured.io (YOLOX Vision Model)
* **Vector Database:** ChromaDB
* **Embeddings:** HuggingFace (`BAAI/bge-small-en-v1.5`)
* **LLM Engine:** Ollama (`Llama 3 8B`)
* **Orchestration:** LangChain

## How to Run Locally

**1. Clone the repo and set up the environment:**
```bash
git clone <https://github.com/Anay162/govt-budget-parser>
cd govt-budget-parser
python -m venv budget_env
source budget_env/bin/activate
pip install -r requirements.txt
```

**2. Ensure Ollama is running:**
Download Ollama and pull the Llama 3 model:

    ollama run llama3

**3. Ingest a document:**
Place any government PDF into the `data/raw/` folder, then run the dynamic ingestion pipeline:

    python src/ingest.py annarbor-fy25.pdf --municipality "Ann Arbor" --type "Budget" --year 2025

**4. Launch the UI:**

    python -m streamlit run app.py

## 🔍 Key Features to Test
* **HTML Table Reconstruction:** Ask a specific row/column financial question and check the "Inspected Sources" expander to see the rendered HTML.
* **Metadata Siloing:** Attempt to search for a Detroit initiative while the sidebar filter is set to Ann Arbor to watch the system safely block the search.
