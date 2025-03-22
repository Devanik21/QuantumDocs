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

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

# Function to extract text from DOCX
def extract_text_from_docx(uploaded_file):
    doc = Document(uploaded_file)
    return "\n".join([para.text for para in doc.paragraphs])

# Function to extract text from TXT
def extract_text_from_txt(uploaded_file):
    return uploaded_file.read().decode("utf-8")

# File upload
uploaded_files = st.file_uploader("Upload Documents (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)

# Extract and store text
corpus = ""
if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file.type == "application/pdf":
            corpus += extract_text_from_pdf(uploaded_file) + "\n\n"
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            corpus += extract_text_from_docx(uploaded_file) + "\n\n"
        elif uploaded_file.type == "text/plain":
            corpus += extract_text_from_txt(uploaded_file) + "\n\n"
    st.success("Documents processed successfully!")

# User query
query = st.text_input("Ask a question about the documents:")

# Function to call Gemini API
def query_gemini_rag(query, context, api_key):
    if not api_key:
        return "❌ API key is required."
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")  # Updated model
    response = model.generate_content(f"Context: {context}\n\nQuestion: {query}")
    return response.text

# Generate response
if query and corpus and api_key:
    with st.spinner("Analyzing documents..."):
        response = query_gemini_rag(query, corpus, api_key)
        st.subheader("💡 AI Response:")
        st.write(response)
