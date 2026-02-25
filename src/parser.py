import os
import json
from pathlib import Path
from unstructured.partition.pdf import partition_pdf

# Define paths based on our architecture
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

def parse_budget_pdf(pdf_path: Path, output_filename: str):
    """
    Parses a complex government PDF budget using layout-aware vision models.
    Extracts tables as HTML to preserve row/column semantics.
    """
    print(f"Starting high-resolution parsing for: {pdf_path.name}...")
    print("Note: This uses vision models (YOLOX) and can be computationally heavy.")

    #hi_res strategy identifies tables vs text
    elements = partition_pdf(
        filename=str(pdf_path),
        strategy="hi_res",
        infer_table_structure=True,
        extract_images_in_pdf=False,
        languages=["eng"]
    )

    parsed_data = []
    
    for element in elements:
        element_dict = element.to_dict()
        
        # Clean up the output to only keep what we need for the vector DB
        clean_element = {
            "type": element_dict.get("type"),
            "text": element_dict.get("text"),
            "metadata": {
                "source": pdf_path.name,
                "page_number": element_dict.get("metadata", {}).get("page_number"),
            }
        }

        # If it's a table, unstructured extracts the raw HTML layout
        if element_dict.get("type") == "Table":
            clean_element["metadata"]["text_as_html"] = element_dict.get("metadata", {}).get("text_as_html")

        parsed_data.append(clean_element)

    # Save the structured data as JSON for the chunker
    output_path = PROCESSED_DATA_DIR / output_filename
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(parsed_data, f, indent=4)
        
    print(f"Successfully parsed and saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    # Example usage:
    # parse_budget_pdf(RAW_DATA_DIR / "sample.pdf", "sample_parsed.json")
    pass