import streamlit as st
from dotenv import load_dotenv

from helpers.chunker import chunk_data
from helpers.youtubeloader import load_from_youtube
from helpers.vectorstore import create_vector_store
from helpers.retriever import create_retriever
# Import the new chain function
from helpers.chain import create_rag_chain


# Configure the page title and layout
st.set_page_config(page_title="YouTube Q&A", layout="wide")

# Set the main title of the app
st.title("Ask Questions to any YouTube Video 💬")

# Load environment variables (for Hugging Face API key)
load_dotenv()

# Initialize session state
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# --- Sidebar for inputting the YouTube URL ---
with st.sidebar:
    st.header("Setup")
    youtube_url = st.text_input("Enter YouTube URL:")

    if st.button("Process Video"):
        if youtube_url:
            with st.spinner("Processing video... This may take a few minutes."):
                try:
                    transcript = load_from_youtube(youtube_url)
                    chunks = chunk_data(transcript)
                    vector_store = create_vector_store(chunks)
                    st.session_state.retriever = create_retriever(vector_store)
                    
                    # Create the RAG chain after the retriever is ready
                    st.session_state.rag_chain = create_rag_chain(st.session_state.retriever)
                    
                    st.success("Video processed successfully!")

                except Exception as e:
                    st.error(f"Error occurred: {e}")
        else:
            st.warning("Please enter a YouTube URL.")

# --- Main area for Q&A ---
st.header("Q&A")
# Check if the RAG chain is ready
if st.session_state.rag_chain:
    st.info("Ready to answer questions.")
    question = st.text_input("Ask a question about the video:")
    
    if question:
        with st.spinner("Generating answer..."):
            # The final step: invoke the chain with the user's question
            answer = st.session_state.rag_chain.invoke(question)
            st.write(answer)
else:
    st.info("Please enter a YouTube URL in the sidebar and click 'Process Video'.")