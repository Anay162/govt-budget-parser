import os
import json
from pathlib import Path

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
CHUNKS_DIR = BASE_DIR / "data" / "chunks"

def create_semantic_chunks(input_filename: str, output_filename: str, global_metadata: dict):
    """
    Reads parsed JSON elements, groups text under section headers, 
    isolates tables, and injects global filtering metadata.
    """
    input_path = PROCESSED_DATA_DIR / input_filename
    
    if not input_path.exists():
        print(f"Error: Could not find {input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        elements = json.load(f)

    chunks = []
    current_section = "General Document Start"
    current_text_block = []
    current_page = 1

    def save_chunk(text_content, section, page, is_table=False, html_content=""):
        """Helper to format and save a chunk with injected metadata."""
        if not text_content and not html_content:
            return
            
        # Merge the global metadata (e.g., city, year) with chunk-specific metadata
        meta = global_metadata.copy()
        meta.update({
            "section_header": section,
            "page_number": page,
            "is_table": is_table
        })
        
        chunk = {
            "text": text_content,
            "metadata": meta
        }
        
        # If it's a table, we attach the HTML layout for the LLM to read later
        if is_table and html_content:
            chunk["text_as_html"] = html_content
            
        chunks.append(chunk)

    print(f"Chunking {len(elements)} elements...")

    for el in elements:
        el_type = el.get("type")
        text = el.get("text", "")
        meta = el.get("metadata", {})
        
        # Track the page number as we move through the document
        page = meta.get("page_number", current_page)
        current_page = page

        if el_type == "Title":
            # We hit a new section header. Save everything we accumulated under the old header.
            if current_text_block:
                save_chunk("\n".join(current_text_block), current_section, page)
                current_text_block = [] # Reset for the new section
            
            # Update the active section header
            current_section = text
            
        elif el_type == "Table":
            # Save any text that came right before the table
            if current_text_block:
                save_chunk("\n".join(current_text_block), current_section, page)
                current_text_block = []
                
            # Save the table as its own highly concentrated chunk
            html_data = meta.get("text_as_html", "")
            save_chunk(text, current_section, page, is_table=True, html_content=html_data)
            
        elif el_type in ["NarrativeText", "ListItem", "Formula", "UncategorizedText"]:
            # Accumulate text paragraphs and bullet points together
            current_text_block.append(text)
            
        # Note: We deliberately ignore "Header", "Footer", and "FigureCaption" 
        # to keep the vector embeddings clean from page numbers and noise.

    # Catch the final block of text at the end of the document
    if current_text_block:
        save_chunk("\n".join(current_text_block), current_section, current_page)

    # Save the final chunks to disk
    output_path = CHUNKS_DIR / output_filename
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=4)
        
    print(f"Successfully created {len(chunks)} semantic chunks.")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    # Example usage:
    # mock_metadata = {"municipality": "Ann Arbor", "year": 2025}
    # create_semantic_chunks("parsed.json", "chunks.json", mock_metadata)
    pass