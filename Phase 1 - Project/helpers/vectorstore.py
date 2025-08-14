from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain_core.documents import Document

def create_vector_store(docs: List[Document]):
    """
    Creates a Chroma vector store from a list of documents.

    Args:
        docs: A list of Document objects (chunks).

    Returns:
        A Chroma vector store instance.
    """
    print("Creating vector store... This may take a moment.")
    
    # Initialize the embedding model from Hugging Face
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'} # Use CPU. You can change to 'cuda' if you have a GPU.
    )

    # Create the vector store from the documents and embedding model
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model
    )
    
    print("Vector store created successfully.")
    return vectorstore