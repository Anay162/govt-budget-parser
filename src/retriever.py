import os
from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / "data" / "chroma_db"

def get_retriever():
    """
    Initializes the embedding model and connects to the local ChromaDB.
    Returns the vector store object so we can query it.
    """
    if not DB_DIR.exists():
        raise FileNotFoundError(f"Database not found at {DB_DIR}. Run embed_store.py first.")

    # We MUST use the exact same embedding model we used to create the database
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'}, 
        encode_kwargs={'normalize_embeddings': True}
    )

    # Load the persisted database
    vector_store = Chroma(
        collection_name="pursuit_budget_docs",
        persist_directory=str(DB_DIR),
        embedding_function=embedding_model
    )
    
    return vector_store

def search_documents(query: str, metadata_filters: dict = None, top_k: int = 3):
    """
    Executes a metadata-filtered vector search.
    """
    print(f"\n--- Searching for: '{query}' ---")
    if metadata_filters:
        print(f"Applying pre-filters: {metadata_filters}")
        
    vector_store = get_retriever()
    
    # we pass the filters directly into the similarity search.
    results = vector_store.similarity_search(
        query=query, 
        k=top_k, 
        filter=metadata_filters
    )
    
    if not results:
        print("No results found matching the query and filters.")
        return []

    print(f"Found {len(results)} relevant chunks. Top result preview:")

    top_doc = results[0]
    print(f"\n[Relevance 1] Section: {top_doc.metadata.get('section_header', 'Unknown')}")
    print(f"Page: {top_doc.metadata.get('page_number', 'Unknown')}")

    if top_doc.metadata.get("is_table"):
        print(">> Type: Data Table (HTML layout preserved in metadata)")

    preview_text = top_doc.page_content.replace('\n', ' ')[:200]
    print(f"Preview: {preview_text}...\n")
    
    return results

if __name__ == "__main__":
    # Test 1: A general semantic search without filters
    q1 = "What is the relationship between political polarization and democratic backsliding?"
    search_documents(query=q1, top_k=2)
    
    # Test 2: A highly specific search using the hard metadata filters we injected
    q2 = "What are the ARIMA model parameters used?"
    strict_filters = {
        "$and": [
            {"year": 2024},
            {"document_type": "Academic Report"}
        ]
    }
    search_documents(query=q2, metadata_filters=strict_filters, top_k=1)