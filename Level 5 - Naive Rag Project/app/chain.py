from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableMap, RunnablePassthrough, RunnableLambda
from langchain.schema.runnable import Runnable
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from app.config import GROQ_API_KEY


def retrieve_hybrid_docs(query, vectorstore, top_k=4):
    """
    Performs hybrid document retrieval: semantic (Chroma) + keyword (BM25).
    Deduplicates results based on content.

    Args:
        query (str): User's question
        vectorstore (Chroma): Chroma vectorstore
        bm25_retriever (BM25Retriever): BM25 keyword retriever
        top_k (int): Number of semantic results to fetch

    Returns:
        List[Document]: Combined, deduplicated list of relevant documents
    """
    semantic_docs = vectorstore.similarity_search(query, k=top_k)
    # keyword_docs = bm25_retriever.get_relevant_documents(query)

    # Deduplicate by page_content
    combined = {doc.page_content: doc for doc in semantic_docs}
    return list(combined.values())


def rerank_documents(query, docs, top_k=4):
    """
    Reranks documents using a pretrained CrossEncoder.

    Args:
        query (str): The user question
        docs (List[Document]): The retrieved documents
        top_k (int): Number of top documents to return

    Returns:
        List[Document]: Top reranked documents
    """
    # Load a pretrained CrossEncoder model (you can change to a smaller/faster one if needed)
    model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

    reranker = CrossEncoderReranker(model=model)

    # Apply reranker
    top_docs = reranker.compress_documents(documents=docs, query=query)

    # Return top K documents
    return top_docs[:top_k]


def build_llm_chain():
    """
    Builds a streaming LLM chain using LangChain Runnables with Groq LLaMA model.
    """

    # Prompt template to inject context + user question
    prompt = PromptTemplate.from_template("""
    You are a helpful AI assistant. Answer the question based only on the context below.

    Context:
    {context}

    Question:
    {question}

    Answer:""")

    # Set up Groq's LLaMA 3 model with streaming enabled
    llm = ChatGroq(
        model="llama3-8b-8192",
        api_key=GROQ_API_KEY,
        streaming=True
    )

    # Chain: prepare prompt → pass to LLM → parse output
    chain = (
        RunnableMap({
            "context": lambda x: "\n\n".join([doc.page_content for doc in x["docs"]]),
            "question": lambda x: x["question"]
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain
