import streamlit as st
import tempfile

def upload_pdfs():
    with st.sidebar:
        st.header("📁 Upload RAG PDFs")
        uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=False)
        submit = st.button(" Submit to DB")
    return uploaded_files, submit

def save_uploaded_files(uploaded_files):
    file_paths = []
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            file_paths.append(tmp.name)
    return file_paths