import os
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever


# Persistent directory for Chroma vectorstore
PERSIST_DIR = "./chroma_store"

def store_chunks(chunks):
    """
    Stores the given chunks in ChromaDB with HuggingFace embeddings.
    Returns the vectorstore instance.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # If the vectorstore already exists, append to it
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
        vectorstore.add_documents(chunks)
        vectorstore.persist()
    else:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIR
        )
        vectorstore.persist()

    return vectorstore


def get_bm25_retriever(chunks):
    """
    Initializes and returns a BM25Retriever from text chunks.
    Useful for keyword-based retrieval in hybrid search.
    """

    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 4
    return bm25


def get_vectorstore():
    """
    Loads the existing Chroma vectorstore without reprocessing PDFs.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )