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

# Sidebar for API Key and Name input
with st.sidebar:
    st.header("🔑 API Configuration")
    api_key = st.text_input("Enter Gemini API Key:", type="password")
    user_name = st.text_input("Enter your name:", key="user_name")

# Increase upload limit (1GB)
st.session_state["max_upload_size"] = 1 * 1024 * 1024 * 1024  # 1GB

# Initialize chat memory in session state
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "user_name" not in st.session_state:
    st.session_state["user_name"] = user_name or "User"

# Function to extract text from different document types
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
    return [json.dumps(json.load(uploaded_file), indent=2)]

def extract_text_from_md(uploaded_file):
    return uploaded_file.read().decode("utf-8").split("\n\n")

# File upload
uploaded_files = st.file_uploader(
    "Upload Documents (PDF, DOCX, TXT, CSV, JSON, MD)", 
    type=["pdf", "docx", "txt", "csv", "json", "md"], 
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

# Function to query Gemini API with chat memory
def query_gemini_rag(query, context_chunks, api_key):
    if not api_key:
        return "❌ API key is required."

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    # Build prompt dynamically to ensure context retention
    chat_history_text = "\n".join(st.session_state["chat_history"][-5:])  # Last 5 messages for context
    prompt = f"User: {st.session_state['user_name']}\nChat History:\n{chat_history_text}\n\n"

    # Include document excerpts
    for chunk in context_chunks[:10]:  # Process first 10 chunks
        prompt += f"- {chunk[:2000]}\n\n"

    prompt += f"\n\nNow, respond to the latest question:\n{query}"

    response = model.generate_content(
        prompt,
        generation_config={"temperature": 0.7, "top_p": 0.9, "max_output_tokens": 8192}
    )

    return response.text

# Chat Interface
st.subheader("💬 Chat with Your Documents")
query = st.text_input("Ask something:")

if query and api_key:
    with st.spinner("🔍 Thinking..."):
        response = query_gemini_rag(query, corpus_chunks, api_key)
        
        # Store chat history
        st.session_state["chat_history"].append(f"User: {query}")
        st.session_state["chat_history"].append(f"AI: {response}")

        # Display chat history
        for message in st.session_state["chat_history"][-10:]:  # Show last 10 messages
            st.write(message)
