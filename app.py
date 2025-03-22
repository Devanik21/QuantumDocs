import streamlit as st
import google.generativeai as genai
import io
import base64
from datetime import datetime

# Function to load prompt templates
@st.cache_data
def load_prompt_templates():
    templates = {
        "document_chat": "Analyze the uploaded document and answer: {prompt}. Keep responses concise and relevant.",
        "ai_generation": "Generate content using AI for: {prompt}. Ensure high-quality output."
    }
    return templates

# Function to generate content with Gemini API
def generate_ai_content(prompt, api_key, model_name):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        with st.spinner("🔮 AI is working its magic..."):
            generation_config = {"temperature": 0.7, "top_p": 0.95, "top_k": 40, "max_output_tokens": 32768}  # Increased output token limit
            response = model.generate_content(prompt, generation_config=generation_config)
            return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# Function to save content to history
def save_to_history(tool_name, prompt, output):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.history.insert(0, {"timestamp": timestamp, "tool": tool_name, "prompt": prompt, "output": output})
    if len(st.session_state.history) > 20:
        st.session_state.history = st.session_state.history[:20]

# Streamlit UI
st.title("🌌 QuantumDocs AI: Entangle with Your Documents")
st.write("Upload your documents and interact with them on a quantum level!")

api_key = st.text_input("Enter Google Gemini API Key", type="password")

uploaded_files = st.file_uploader("Upload PDF/DOCX/TXT files", accept_multiple_files=True)
user_query = st.text_input("Ask something about your documents")

if api_key and uploaded_files and user_query:
    st.write("Processing...")
    response = generate_ai_content(user_query, api_key, "gemini-model")
    st.write(response)
    save_to_history("document_chat", user_query, response)
