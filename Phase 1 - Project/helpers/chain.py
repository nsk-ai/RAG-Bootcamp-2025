# helpers/chain.py

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
# Import ChatGroq instead of HuggingFaceHub
from langchain_groq import ChatGroq
from langchain.retrievers import ContextualCompressionRetriever

def _format_docs(docs: list) -> str:
    """
    Formats the retrieved documents into a single string.
    """
    return "\n\n".join(doc.page_content for doc in docs)

def create_rag_chain(retriever: ContextualCompressionRetriever):
    """
    Creates the full RAG chain for question answering using Groq.

    Args:
        retriever: The retriever instance to fetch relevant documents.

    Returns:
        A runnable RAG chain.
    """
    # 1. Initialize the LLM from Groq
    # We'll use Llama 3, which is very fast on Groq.
    # The API key is automatically read from the GROQ_API_KEY environment variable.
    llm = ChatGroq(
        temperature=0, 
        model_name="llama3-8b-8192"
    )

    # 2. Create the prompt template
    prompt_template = """
    Answer the following question based only on the provided context.
    Your answer should be concise and to the point.
    If the context does not contain the answer, state that the answer is not available in the context.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # 3. Create the RAG chain using LCEL
    rag_chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain