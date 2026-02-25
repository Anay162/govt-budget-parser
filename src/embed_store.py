import os
import json
from pathlib import Path
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
CHUNKS_DIR = BASE_DIR / "data" / "chunks"
DB_DIR = BASE_DIR / "data" / "chroma_db"

def create_vector_database(input_filename: str, collection_name: str = "pursuit_budget_docs"):
    """
    Reads semantic chunks, generates open-source vector embeddings, 
    and stores them in a local ChromaDB vector store with metadata.
    """
    input_path = CHUNKS_DIR / input_filename
    
    if not input_path.exists():
        print(f"Error: Could not find {input_path}")
        return

    print(f"Loading chunks from {input_filename}...")
    with open(input_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Convert JSON dictionaries into LangChain Document objects
    documents = []
    for chunk in chunks:

        doc = Document(
            page_content=chunk["text"],
            metadata=chunk["metadata"]
        )
        # If it's a table, ensure the HTML representation is stored in metadata
        # so the LLM can render it perfectly later, even though we embed the raw text.
        if chunk.get("text_as_html"):
            doc.metadata["text_as_html"] = chunk["text_as_html"]
            
        documents.append(doc)

    print(f"Prepared {len(documents)} documents for vectorization.")

    # Initialize the embedding model. 
    # BAAI/bge-small-en-v1.5 is an incredibly fast, highly-rated open source retrieval model.
    # For a production gov-tech environment, you might scale this up to BGE-m3.
    print("Initializing HuggingFace embedding model (this may download model weights the first time)...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'}, # Change to 'mps' or 'cuda' if running on a GPU
        encode_kwargs={'normalize_embeddings': True} # Normalization improves cosine similarity search
    )

    # Create and persist the Chroma vector database
    print(f"Embedding documents and building ChromaDB at {DB_DIR}...")
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        collection_name=collection_name,
        persist_directory=str(DB_DIR)
    )

    print("\nSuccess! Vector database populated and saved to disk.")
    print("You are now ready to perform semantic and metadata-filtered searches.")

if __name__ == "__main__":
    # Example usage:
    # create_vector_database("chunks.json")
    pass