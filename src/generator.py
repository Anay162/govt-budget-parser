import os
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# Import the search function from the script we wrote previously
from retriever import search_documents

# Define a strict prompt to prevent the LLM from hallucinating
PROMPT_TEMPLATE = """
You are an expert data analyst specializing in parsing complex government and academic documents.
Answer the question based STRICTLY on the following provided context. 

If the exact answer cannot be found in the context, say: "I cannot answer this based on the retrieved documents."
Do not invent or hallucinate external information. If the context contains an HTML table, read the rows and columns carefully to extract the precise data.

====================
CONTEXT:
{context}
====================

QUESTION: 
{question}

EXPERT ANSWER:
"""

def generate_answer(query: str, metadata_filters: dict = None):
    """
    Executes the full RAG pipeline: Retrieval -> Context Formatting -> Generation.
    """
    # 1. Retrieve the top relevant chunks using your existing architecture
    print("\n[1/3] Retrieving relevant documents from ChromaDB...")
    results = search_documents(query, metadata_filters=metadata_filters, top_k=5)
    
    if not results:
        print("No relevant documents found. Aborting generation.")
        return "No relevant documents found to answer the question."

    # 2. Format the retrieved chunks into a single readable context block
    print("\n[2/3] Formatting context for Llama 3...")
    context_parts = []
    for i, doc in enumerate(results):
        # The Flex: If it's a table, feed the LLM the raw HTML. Otherwise, use the text.
        content = doc.metadata.get("text_as_html", doc.page_content)
        source_section = doc.metadata.get("section_header", "Unknown Section")
        
        formatted_chunk = f"--- Document {i+1} (Source Section: {source_section}) ---\n{content}\n"
        context_parts.append(formatted_chunk)
        
    full_context_string = "\n".join(context_parts)

    print("\n--- DEBUG: Context being sent to LLM ---")
    # Only printing the first 1000 characters to keep the terminal clean
    print(full_context_string[:1000] + "\n...[truncated]...") 
    print("----------------------------------------\n")

    # 3. Build the final prompt
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    final_prompt = prompt.format(context=full_context_string, question=query)

    # 4. Initialize the local Llama model via Ollama
    print("\n[3/3] Querying local Llama 3 model... (This may take a few seconds)")
    llm = OllamaLLM(model="llama3") 
    
    # 5. Generate the response
    response = llm.invoke(final_prompt)
    
    print("\n" + "="*40)
    print("FINAL GENERATED ANSWER:")
    print("="*40)
    print(response.strip())
    print("="*40 + "\n")
    
    return response

if __name__ == "__main__":

    test_query = "Based on the ARIMA Modeling section, what is the exact AIC value for the AR2 and MA2 model?"
    
    strict_filters = {
        "$and": [
            {"year": 2024},
            {"document_type": "Academic Report"}
        ]
    }
    
    generate_answer(query=test_query, metadata_filters=strict_filters)