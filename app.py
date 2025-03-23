import streamlit as st
import google.generativeai as genai
import os
import PyPDF2
from docx import Document
import json
import pandas as pd

# Initialize Streamlit app
st.set_page_config(page_title="QuantumDocs AI: Entangle with Your Documents", page_icon="📄")
st.title("📄 QuantumDocs AI: Entangle with Your Documents")

# Sidebar for API Key input
with st.sidebar:
    st.header("🔑 API Configuration")
    api_key = st.text_input("Enter Gemini API Key:", type="password")

# Increase file upload limit (1GB)
st.session_state["max_upload_size"] = 1 * 1024 * 1024 * 1024  # 1GB

# Function to extract text from large PDF in chunks
def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text_chunks = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            text_chunks.append(text)
    return text_chunks  # Return as a list of chunks

# Function to extract text from DOCX
def extract_text_from_docx(uploaded_file):
    doc = Document(uploaded_file)
    text_chunks = [para.text for para in doc.paragraphs if para.text]
    return text_chunks  # Return as a list of chunks

# Function to extract text from TXT
def extract_text_from_txt(uploaded_file):
    text = uploaded_file.read().decode("utf-8")
    return text.split("\n\n")  # Split into paragraphs for better chunking

# Function to extract text from CSV
def extract_text_from_csv(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df.astype(str).apply(lambda x: " ".join(x), axis=1).tolist()

# Function to extract text from JSON
def extract_text_from_json(uploaded_file):
    data = json.load(uploaded_file)
    return [json.dumps(data, indent=2)]  # Convert JSON to a string

# Function to extract text from Markdown
def extract_text_from_md(uploaded_file):
    text = uploaded_file.read().decode("utf-8")
    return text.split("\n\n")  # Split into paragraphs

from pptx import Presentation

def extract_text_from_pptx(uploaded_file):
    presentation = Presentation(uploaded_file)
    text_chunks = []
    for slide in presentation.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text:
                text_chunks.append(shape.text)
    return text_chunks

def extract_text_from_xlsx(uploaded_file):
    df = pd.read_excel(uploaded_file)
    return df.astype(str).apply(lambda x: " ".join(x), axis=1).tolist()

from bs4 import BeautifulSoup

def extract_text_from_html(uploaded_file):
    soup = BeautifulSoup(uploaded_file.read(), "html.parser")
    return [soup.get_text()]

import ebooklib
from ebooklib import epub

def extract_text_from_epub(uploaded_file):
    book = epub.read_epub(uploaded_file)
    text_chunks = []
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.content, "html.parser")
            text_chunks.append(soup.get_text())
    return text_chunks

import zipfile
import io

def extract_text_from_zip(uploaded_file):
    text_chunks = []
    with zipfile.ZipFile(uploaded_file, 'r') as z:
        for file_name in z.namelist():
            with z.open(file_name) as file:
                ext = file_name.split('.')[-1].lower()
                if ext == "txt":
                    text_chunks.extend(extract_text_from_txt(io.BytesIO(file.read())))
                elif ext == "csv":
                    text_chunks.extend(extract_text_from_csv(io.BytesIO(file.read())))
                elif ext == "json":
                    text_chunks.extend(extract_text_from_json(io.BytesIO(file.read())))
                elif ext == "md":
                    text_chunks.extend(extract_text_from_md(io.BytesIO(file.read())))
                elif ext == "docx":
                    text_chunks.extend(extract_text_from_docx(io.BytesIO(file.read())))
                elif ext == "pdf":
                    text_chunks.extend(extract_text_from_pdf(io.BytesIO(file.read())))
    return text_chunks

# File upload
uploaded_files = st.file_uploader(
    "Upload Documents (PDF, DOCX, TXT, CSV, JSON, MD, PPTX, XLSX, HTML, EPUB, ZIP)", 
    type=["pdf", "docx", "txt", "csv", "json", "md", "pptx", "xlsx", "html", "epub", "zip"], 
    accept_multiple_files=True
)


# Extract and store text
corpus_chunks = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_type = uploaded_file.type

        if file_type == "application/pdf":
            corpus_chunks.extend(extract_text_from_pdf(uploaded_file))
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            corpus_chunks.extend(extract_text_from_docx(uploaded_file))
        elif file_type == "text/plain":
            corpus_chunks.extend(extract_text_from_txt(uploaded_file))
        elif file_type == "text/csv":
            corpus_chunks.extend(extract_text_from_csv(uploaded_file))
        elif file_type == "application/json":
            corpus_chunks.extend(extract_text_from_json(uploaded_file))
        elif file_type == "text/markdown":
            corpus_chunks.extend(extract_text_from_md(uploaded_file))

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
