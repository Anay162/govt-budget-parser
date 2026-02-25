import os
import argparse
from pathlib import Path
import warnings
import logging

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Mute pdfminer color space warnings
logging.getLogger("pdfminer").setLevel(logging.ERROR)

# Import our core pipeline functions
from parser import parse_budget_pdf
from chunker import create_semantic_chunks
from embed_store import create_vector_database

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"

def run_ingestion_pipeline(pdf_filename: str, municipality: str, doc_type: str, year: int):
    """Runs a document through the entire RAG backend pipeline."""
    pdf_path = RAW_DATA_DIR / pdf_filename
    
    if not pdf_path.exists():
        print(f" Error: Could not find {pdf_path}")
        return

    # Define dynamic filenames based on the input PDF
    base_name = pdf_path.stem
    parsed_json_name = f"{base_name}_parsed.json"
    chunked_json_name = f"{base_name}_chunks.json"

    # Define the dynamic metadata dictionary
    global_metadata = {
        "municipality": municipality,
        "document_type": doc_type,
        "year": year
    }

    print(f"\n Starting Ingestion Pipeline for: {pdf_filename}")
    print(f"Attached Metadata: {global_metadata}\n")

    # Step 1: Layout-Aware Parsing
    print("--- [1/3] Parsing PDF (This may take a while for large budgets) ---")
    parse_budget_pdf(pdf_path, parsed_json_name)

    # Step 2: Semantic Chunking
    print("\n--- [2/3] Chunking and Injecting Metadata ---")
    create_semantic_chunks(parsed_json_name, chunked_json_name, global_metadata)

    # Step 3: Vectorization & Storage
    print("\n--- [3/3] Embedding into ChromaDB ---")
    create_vector_database(chunked_json_name)

    print(f"\n Successfully ingested {pdf_filename} into the vector database!")

if __name__ == "__main__":
    # Set up the CLI argument parser
    parser = argparse.ArgumentParser(description="Ingest a government PDF into the Pursuit RAG database.")
    parser.add_argument("filename", help="Name of the PDF file in data/raw/ (e.g., ann_arbor_budget.pdf)")
    parser.add_argument("--municipality", required=True, help="E.g., 'Ann Arbor', 'Detroit', 'Federal'")
    parser.add_argument("--type", required=True, help="E.g., 'Budget', 'Meeting Minutes', 'ACFR'")
    parser.add_argument("--year", required=True, type=int, help="Document year (e.g., 2025)")

    args = parser.parse_args()

    run_ingestion_pipeline(args.filename, args.municipality, args.type, args.year)