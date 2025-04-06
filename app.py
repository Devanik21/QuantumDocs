import streamlit as st
import google.generativeai as genai
import os
import PyPDF2
from docx import Document
import json
import pandas as pd
from io import BytesIO
import time
import base64
import uuid
import re

# Initialize Streamlit app
st.set_page_config(page_title="Claude-Like Chat Interface", page_icon="🤖", layout="wide")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
    
if "file_contents" not in st.session_state:
    st.session_state.file_contents = {}

if "input_history" not in st.session_state:
    st.session_state.input_history = []
    
if "history_index" not in st.session_state:
    st.session_state.history_index = -1

# Custom CSS for better UI resembling Claude
st.markdown("""
<style>
.chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
}
.chat-message.user {
    background-color: #f0f2f6;
}
.chat-message.assistant {
    background-color: #ffffff;
    border: 1px solid #e6e6e6;
}
.file-upload-area {
    border: 2px dashed #aaa;
    border-radius: 5px;
    padding: 20px;
    text-align: center;
    margin-bottom: 20px;
    cursor: pointer;
}
.file-badge {
    background-color: #f0f2f6;
    border-radius: 15px;
    padding: 5px 10px;
    margin-right: 5px;
    margin-bottom: 5px;
    display: inline-block;
    font-size: 0.8rem;
}
.stButton>button {
    width: 100%;
}
.thinking-animation {
    display: flex;
    align-items: center;
    margin-top: 10px;
}
.thinking-dot {
    width: 8px;
    height: 8px;
    background-color: #888;
    border-radius: 50%;
    margin-right: 5px;
    animation: thinking 1.4s infinite ease-in-out both;
}
.thinking-dot:nth-child(1) { animation-delay: -0.32s; }
.thinking-dot:nth-child(2) { animation-delay: -0.16s; }
@keyframes thinking {
    0%, 80%, 100% { transform: scale(0); }
    40% { transform: scale(1); }
}
</style>
""", unsafe_allow_html=True)

# Sidebar for API Key and settings
with st.sidebar:
    st.header("🔑 API Configuration")
    api_key = st.text_input("Enter API Key:", type="password")
    
    st.header("⚙️ Model Settings")
    model_option = st.selectbox(
        "Model Selection:", 
        ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash"]
    )
    
    temperature = st.slider("Temperature:", 0.0, 1.0, 0.7, 0.1)
    
    # File upload section with drag and drop visualization
    st.header("📂 Upload Files")
    
    # Custom file upload area with "+" button visual
    st.markdown("""
    <div class="file-upload-area">
        <div style="font-size: 24px; margin-bottom: 10px;">+</div>
        <div>Click or drag files here</div>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Upload files (hidden label)", 
        type=["pdf", "docx", "txt", "csv", "json", "jpg", "jpeg", "png"], 
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    # Add files button with enhanced styling
    if uploaded_files:
        if st.button("➕ Add Files to Chat", key="add_files"):
            # Process and add files to session state
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in [f["name"] for f in st.session_state.uploaded_files]:
                    file_type = uploaded_file.name.split('.')[-1].lower()
                    file_id = str(uuid.uuid4())
                    
                    # Extract text from documents
                    if file_type == 'pdf':
                        text_content = extract_text_from_pdf(uploaded_file)
                    elif file_type == 'docx':
                        text_content = extract_text_from_docx(uploaded_file)
                    elif file_type == 'txt':
                        text_content = uploaded_file.getvalue().decode('utf-8')
                    elif file_type == 'csv':
                        text_content = f"CSV file with {len(pd.read_csv(uploaded_file))} rows"
                    elif file_type == 'json':
                        text_content = json.loads(uploaded_file.getvalue())
                    elif file_type in ['jpg', 'jpeg', 'png']:
                        # For images, store the binary data
                        text_content = "Image file"
                        
                    # Store file info and content
                    st.session_state.uploaded_files.append({
                        "id": file_id,
                        "name": uploaded_file.name,
                        "type": file_type,
                        "size": uploaded_file.size
                    })
                    
                    st.session_state.file_contents[file_id] = {
                        "name": uploaded_file.name,
                        "content": uploaded_file.getvalue(),
                        "extracted_text": text_content
                    }
                    
            # Add system message about files
            file_names = [f["name"] for f in st.session_state.uploaded_files]
            st.session_state.messages.append({
                "role": "system",
                "content": f"Files added to chat: {', '.join(file_names)}"
            })
            st.rerun()
    
    # Display current files in chat with better styling
    if st.session_state.uploaded_files:
        st.header("Current Files in Chat")
        st.markdown('<div style="display: flex; flex-wrap: wrap;">', unsafe_allow_html=True)
        
        for idx, file in enumerate(st.session_state.uploaded_files):
            # Get file icon based on type
            file_type = file['type'].lower()
            if file_type == 'pdf':
                icon = "📄"
            elif file_type in ['jpg', 'jpeg', 'png']:
                icon = "🖼️"
            elif file_type in ['docx', 'txt']:
                icon = "📝"
            elif file_type == 'csv':
                icon = "📊"
            else:
                icon = "📎"
                
            st.markdown(f"""
            <div class="file-badge">
                {icon} {file['name']} ({file['size']/1024:.1f} KB)
                <button onclick="alert('Remove functionality requires JavaScript integration')">❌</button>
            </div>
            """, unsafe_allow_html=True)
            
            # Add regular button as fallback for removal
            if st.button(f"Remove {file['name']}", key=f"remove_{file['id']}"):
                st.session_state.uploaded_files.pop(idx)
                del st.session_state.file_contents[file['id']]
                st.rerun()
                
        st.markdown('</div>', unsafe_allow_html=True)

    # Clear chat button
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.input_history = []
            st.session_state.history_index = -1
            st.rerun()

# Text extraction functions
def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n\n"
    return text

def extract_text_from_docx(uploaded_file):
    doc = Document(uploaded_file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n\n"
    return text

# Function to encode image to base64
def get_image_base64(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

# Main chat interface
st.title("🤖 Claude-Like Chat Interface")

# Display chat messages with better styling
for message in st.session_state.messages:
    if message["role"] == "system":
        st.info(message["content"])
    elif message["role"] == "user":
        with st.chat_message("user"):
            st.write(message["content"])
            # Check if message contains file references
            matches = re.findall(r'\[file:([a-zA-Z0-9-]+)\]', message.get("content", ""))
            for file_id in matches:
                if file_id in st.session_state.file_contents:
                    file_info = st.session_state.file_contents[file_id]
                    file_type = file_info["name"].split('.')[-1].lower()
                    if file_type in ['jpg', 'jpeg', 'png']:
                        # Display image
                        img_b64 = get_image_base64(file_info["content"])
                        st.markdown(f'<img src="data:image/{file_type};base64,{img_b64}" width="200">', unsafe_allow_html=True)
    else:
        with st.chat_message("assistant"):
            st.write(message["content"])

# Generate response function
def generate_response(prompt, api_key, model, temp):
    # Configure the API
    if not api_key:
        return "❌ API key is required."
    
    genai.configure(api_key=api_key)
    model_instance = genai.GenerativeModel(model)
    
    # Get context from uploaded files
    context = ""
    if st.session_state.uploaded_files:
        context += "Here's information from the uploaded files:\n\n"
        for file in st.session_state.uploaded_files:
            file_content = st.session_state.file_contents[file["id"]]
            if isinstance(file_content["extracted_text"], str):
                # Truncate long text to avoid context window issues
                context += f"From {file['name']}:\n{file_content['extracted_text'][:2000]}...\n\n"
    
    # Get conversation history
    history = ""
    for msg in st.session_state.messages:
        if msg["role"] != "system":
            history += f"{msg['role'].capitalize()}: {msg['content']}\n\n"
    
    # Create the full prompt with context and history
    full_prompt = f"{context}\n\nConversation history:\n{history}\n\nUser: {prompt}\n\nAssistant:"
    
    try:
        response = model_instance.generate_content(
            full_prompt, 
            generation_config={
                "temperature": temp,
                "max_output_tokens": 2048
            }
        )
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Custom input area with up arrow functionality
st.markdown("### 💬 Message")
col1, col2 = st.columns([6, 1])

with col1:
    user_input = st.text_area("Type your message...", height=100, label_visibility="collapsed", key="user_input")

with col2:
    send_button = st.button("⬆️ Send", use_container_width=True)
    up_arrow = st.button("↑ History", help="Press to cycle through input history")
    
    # Handle up arrow functionality
    if up_arrow and st.session_state.input_history:
        st.session_state.history_index = (st.session_state.history_index + 1) % len(st.session_state.input_history)
        st.session_state.user_input = st.session_state.input_history[st.session_state.history_index]
        st.rerun()

# Process input when send button is clicked
if send_button and user_input:
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Add to input history
    st.session_state.input_history.insert(0, user_input)
    st.session_state.history_index = -1
    
    # Reset input field
    st.session_state.user_input = ""
    
    # Generate AI response
    with st.spinner("Thinking..."):
        # Display thinking animation
        with st.chat_message("assistant"):
            st.markdown("""
            <div class="thinking-animation">
                <div class="thinking-dot"></div>
                <div class="thinking-dot"></div>
                <div class="thinking-dot"></div>
            </div>
            """, unsafe_allow_html=True)
            
        # Generate response
        response = generate_response(user_input, api_key, model_option, temperature)
    
    # Add AI response to chat
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

# Add a note about features
st.caption("""
💡 **Tips**: 
- Click the + icon to upload files and add them to the conversation
- Use the up arrow button to cycle through previous messages
- Chat history is maintained within the session
""")
