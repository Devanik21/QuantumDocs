import streamlit as st
import google.generativeai as genai
import os
import PyPDF2
from docx import Document

# Initialize Streamlit app
st.set_page_config(page_title="QuantumDocs AI: Entangle with Your Documents", page_icon="📄")
st.title("📄 QuantumDocs AI: Entangle with Your Documents")

# Sidebar for API Key input
with st.sidebar:
    st.header("🔑 API Configuration")
    api_key = st.text_input("Enter Gemini API Key:", type="password")

# Function to extract text from large PDF in chunks
def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text_chunks = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            text_chunks.append(text)
    return text_chunks  # Return as a list of chunks

# Function to extract text from large DOCX in chunks
def extract_text_from_docx(uploaded_file):
    doc = Document(uploaded_file)
    text_chunks = [para.text for para in doc.paragraphs if para.text]
    return text_chunks  # Return as a list of chunks

# Function to extract text from large TXT files in chunks
def extract_text_from_txt(uploaded_file):
    text = uploaded_file.read().decode("utf-8")
    return text.split("\n\n")  # Split into paragraphs for better chunking

# File upload
uploaded_files = st.file_uploader("Upload Documents (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)

# Extract and store text
corpus_chunks = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file.type == "application/pdf":
            corpus_chunks.extend(extract_text_from_pdf(uploaded_file))
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            corpus_chunks.extend(extract_text_from_docx(uploaded_file))
        elif uploaded_file.type == "text/plain":
            corpus_chunks.extend(extract_text_from_txt(uploaded_file))
    st.success(f"✅ {len(corpus_chunks)} document sections processed successfully!")

# User query
query = st.text_input("Ask a question about the documents:")

# Function to call Gemini API with chunked context
def query_gemini_rag(query, context_chunks, api_key):
    if not api_key:
        return "❌ API key is required."

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    # Build prompt dynamically to ensure long, detailed responses
    prompt = f"Provide a detailed, structured, and long-form response (3000+ words) based on the following document excerpts:\n\n"
    for chunk in context_chunks[:10]:  # Process first 10 chunks to avoid exceeding token limits
        prompt += f"- {chunk[:2000]}\n\n"  # Limit each chunk to 2000 characters

    prompt += f"\n\nNow, answer the user's question comprehensively:\n{query}"

    response = model.generate_content(
        prompt,
        generation_config={"temperature": 0.7, "top_p": 0.9, "max_output_tokens": 8192}  # Max output for long responses
    )
    return response.text

# Generate response
if query and corpus_chunks and api_key:
    with st.spinner("🔍 Analyzing documents and generating a detailed response..."):
        response = query_gemini_rag(query, corpus_chunks, api_key)
        st.subheader("💡 AI Response:")
        st.write(response)
