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

# File Extraction Functions
def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    return [page.extract_text() for page in reader.pages if page.extract_text()]

def extract_text_from_docx(uploaded_file):
    doc = Document(uploaded_file)
    return [para.text for para in doc.paragraphs if para.text]

def extract_text_from_txt(uploaded_file):
    return uploaded_file.read().decode("utf-8").split("\n\n")

def extract_text_from_csv(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df.astype(str).apply(lambda x: " ".join(x), axis=1).tolist()

def extract_text_from_json(uploaded_file):
    data = json.load(uploaded_file)
    return [json.dumps(data, indent=2)]

def extract_text_from_md(uploaded_file):
    return uploaded_file.read().decode("utf-8").split("\n\n")

def extract_text_from_pptx(uploaded_file):
    presentation = Presentation(uploaded_file)
    return [shape.text for slide in presentation.slides for shape in slide.shapes if hasattr(shape, "text") and shape.text]

def extract_text_from_xlsx(uploaded_file):
    df = pd.read_excel(uploaded_file)
    return df.astype(str).apply(lambda x: " ".join(x), axis=1).tolist()

def extract_text_from_html(uploaded_file):
    return [BeautifulSoup(uploaded_file.read(), "html.parser").get_text()]

def extract_text_from_epub(uploaded_file):
    book = epub.read_epub(uploaded_file)
    return [BeautifulSoup(item.content, "html.parser").get_text() for item in book.get_items() if item.get_type() == ebooklib.ITEM_DOCUMENT]

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

# File Upload Section
uploaded_files = st.file_uploader(
    "Upload Documents (PDF, DOCX, TXT, CSV, JSON, MD, PPTX, XLSX, HTML, EPUB, ZIP)", 
    type=["pdf", "docx", "txt", "csv", "json", "md", "pptx", "xlsx", "html", "epub", "zip"], 
    accept_multiple_files=True
)

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

# User Query Input
query = st.text_input("Ask a question about the documents:")

def query_gemini_rag(query, context_chunks, api_key):
    if not api_key:
        return "❌ API key is required."
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    prompt = "Provide a detailed, structured, and long-form response based on the following document excerpts:\n\n"
    for chunk in context_chunks[:10]:
        prompt += f"- {chunk[:2000]}\n\n"
    prompt += f"\nNow, answer the user's question comprehensively:\n{query}"
    
    response = model.generate_content(
        prompt,
        generation_config={"temperature": 0.7, "top_p": 0.9, "max_output_tokens": 8192}
    )
    return response.text

if query and corpus_chunks and api_key:
    with st.spinner("🔍 Analyzing documents and generating a detailed response..."):
        response = query_gemini_rag(query, corpus_chunks, api_key)
        st.subheader("💡 AI Response:")
        st.write(response)
