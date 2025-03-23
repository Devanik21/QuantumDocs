import streamlit as st
import google.generativeai as genai
import os
import PyPDF2
from docx import Document
import json
import pandas as pd
from pptx import Presentation
from bs4 import BeautifulSoup
import ebooklib
from ebooklib import epub
import zipfile
import io

# Initialize Streamlit app
st.set_page_config(page_title="QuantumDocs AI: Entangle with Your Documents", page_icon="📝")
st.title("📝 QuantumDocs AI: Entangle with Your Documents")

# Sidebar for API Key input
with st.sidebar:
    st.header("🔑 API Configuration")
    api_key = st.text_input("Enter Gemini API Key:", type="password")

# Increase file upload limit (1GB)
st.session_state["max_upload_size"] = 1 * 1024 * 1024 * 1024  # 1GB

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    return [page.extract_text() for page in reader.pages if page.extract_text()]

# Function to extract text from DOCX
def extract_text_from_docx(uploaded_file):
    doc = Document(uploaded_file)
    return [para.text for para in doc.paragraphs if para.text]

# Function to extract text from TXT
def extract_text_from_txt(uploaded_file):
    return uploaded_file.read().decode("utf-8").split("\n\n")

# Function to extract text from CSV
def extract_text_from_csv(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df.astype(str).apply(lambda x: " ".join(x), axis=1).tolist()

# Function to extract text from JSON
def extract_text_from_json(uploaded_file):
    return [json.dumps(json.load(uploaded_file), indent=2)]

# Function to extract text from Markdown
def extract_text_from_md(uploaded_file):
    return uploaded_file.read().decode("utf-8").split("\n\n")

# Function to extract text from PowerPoint (PPTX)
def extract_text_from_pptx(uploaded_file):
    presentation = Presentation(uploaded_file)
    return [shape.text for slide in presentation.slides for shape in slide.shapes if hasattr(shape, "text") and shape.text]

# Function to extract text from Excel (XLSX)
def extract_text_from_xlsx(uploaded_file):
    df = pd.read_excel(uploaded_file)
    return df.astype(str).apply(lambda x: " ".join(x), axis=1).tolist()

# Function to extract text from HTML
def extract_text_from_html(uploaded_file):
    return [BeautifulSoup(uploaded_file.read(), "html.parser").get_text()]

# Function to extract text from EPUB
def extract_text_from_epub(uploaded_file):
    book = epub.read_epub(uploaded_file)
    return [BeautifulSoup(item.content, "html.parser").get_text() for item in book.get_items() if item.get_type() == ebooklib.ITEM_DOCUMENT]




# File upload
uploaded_files = st.file_uploader(
    "Upload Documents (PDF, DOCX, TXT, CSV, JSON, MD, PPTX, XLSX, HTML, EPUB)", 
    type=["pdf", "docx", "txt", "csv", "json", "md", "pptx", "xlsx", "html", "epub"], 
    accept_multiple_files=True
)

# Extract and store text
corpus_chunks = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_ext = uploaded_file.name.split(".")[-1].lower()
        if file_ext in ["pdf", "docx", "txt", "csv", "json", "md", "pptx", "xlsx", "html", "epub"]:
            extract_func = globals()[f"extract_text_from_{file_ext}"]
            corpus_chunks.extend(extract_func(uploaded_file))
    st.success(f"✅ {len(corpus_chunks)} document sections processed successfully!")

# User query
query = st.text_input("Ask a question about the documents:")

# Function to call Gemini API
def query_gemini_rag(query, context_chunks, api_key):
    if not api_key:
        return "❌ API key is required."
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    prompt = "Provide a detailed, structured, and long-form response (3000+ words) based on the following document excerpts:\n\n"
    for chunk in context_chunks[:10]:
        prompt += f"- {chunk[:2000]}\n\n"
    prompt += f"\n\nNow, answer the user's question comprehensively:\n{query}"
    
    response = model.generate_content(prompt, generation_config={"temperature": 0.7, "top_p": 0.9, "max_output_tokens": 8192})
    return response.text

# Generate response
if query and corpus_chunks and api_key:
    with st.spinner("🔍 Analyzing documents and generating a detailed response..."):
        response = query_gemini_rag(query, corpus_chunks, api_key)
        st.subheader("💡 AI Response:")
        st.write(response)
