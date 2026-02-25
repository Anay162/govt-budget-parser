import sys
import os
from pathlib import Path
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# Add the src directory to the Python path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from retriever import search_documents

# --- UI Configuration ---
st.set_page_config(page_title="Government Document Intelligence", page_icon="🏛️", layout="wide")
st.title("🏛️ Government Document Intelligence")
st.markdown("Advanced RAG pipeline with layout-aware table extraction and metadata filtering.")

# --- Sidebar: Metadata Filters ---
with st.sidebar:
    st.header("🔍 Metadata Routing")
    st.markdown("Select multiple filters to compare documents. Leave blank to search all.")
    
    filter_municipality = st.text_input("Municipality (e.g., Ann Arbor)", value="")
    
    # Multiselect allows 0, 1, or multiple choices
    filter_types = st.multiselect(
        "Document Type", 
        options=["Academic Report", "Budget", "City Council Minutes"],
        default=[]
    )
    
    filter_years = st.multiselect(
        "Document Year", 
        options=[2023, 2024, 2025, 2026],
        default=[]
    )
    
    st.markdown("---")
    st.caption("Powered by Local Llama 3 & BAAI/bge-small")

# --- Chat State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome. What public sector or academic data are you looking for today?"}
    ]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Core RAG Logic ---
prompt_template = """
You are an expert data analyst, specializing in parsing complex government and academic documents.
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

# Accept user input
if prompt := st.chat_input("E.g., What is the exact AIC value for the AR2 and MA2 model?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Create a placeholder for the loading spinner
        with st.spinner("Retrieving relevant documents and applying filters..."):
            
            # 1. Build the metadata filter dynamically from the sidebar
            filter_conditions = []
            
            if filter_municipality:
                filter_conditions.append({"municipality": filter_municipality.strip()})
                
            if filter_types:
                if len(filter_types) == 1:
                    filter_conditions.append({"document_type": filter_types[0]})
                else:
                    # Use ChromaDB's $in operator for multiple selections
                    filter_conditions.append({"document_type": {"$in": filter_types}})
                    
            if filter_years:
                if len(filter_years) == 1:
                    filter_conditions.append({"year": filter_years[0]})
                else:
                    # Use ChromaDB's $in operator for multiple selections
                    filter_conditions.append({"year": {"$in": filter_years}})
                
            # Assemble the final dictionary based on how many conditions we have
            strict_filters = None
            if len(filter_conditions) == 1:
                strict_filters = filter_conditions[0]
            elif len(filter_conditions) > 1:
                strict_filters = {"$and": filter_conditions}
            
            # 2. Retrieve documents (top_k=5 to ensure we catch orphaned tables)
            results = search_documents(prompt, metadata_filters=strict_filters, top_k=5)
            
            if not results:
                st.warning("No relevant documents found matching those filters.")
                st.session_state.messages.append({"role": "assistant", "content": "No relevant documents found."})
            else:
                # 3. Format the context and show the sources in an expander
                context_parts = []
                with st.expander(f"🔍 Inspected {len(results)} source chunks"):
                    for i, doc in enumerate(results):
                        content = doc.metadata.get("text_as_html", doc.page_content)
                        section = doc.metadata.get("section_header", "Unknown Section")
                        page = doc.metadata.get("page_number", "Unknown")
                        
                        st.markdown(f"**Chunk {i+1}:** {section} (Page {page})")
                        if doc.metadata.get("is_table"):
                            st.caption("Type: Data Table (HTML Extracted)")
                            st.html(content) # Renders the actual HTML table in the UI!
                        else:
                            st.text(content[:300] + "...") # Preview the text
                            
                        context_parts.append(f"--- Document {i+1} (Source Section: {section}) ---\n{content}\n")
                        
                full_context_string = "\n".join(context_parts)
                
                # 4. Generate the answer
                llm = OllamaLLM(model="llama3")
                final_prompt = PromptTemplate(
                    template=prompt_template,
                    input_variables=["context", "question"]
                ).format(context=full_context_string, question=prompt)
                
                with st.spinner("Synthesizing answer via Llama 3..."):
                    response = llm.invoke(final_prompt)
                    
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})