from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.vectorstores import Chroma

def create_retriever(
    vectorstore: Chroma, 
    search_k: int = 10, 
    reranker_top_n: int = 3
):
    """
    Creates a retriever with a reranker to improve search results.

    Args:
        vectorstore: The Chroma vector store instance.
        search_k: The number of documents to retrieve initially.
        reranker_top_n: The number of documents to return after reranking.

    Returns:
        A ContextualCompressionRetriever instance.
    """
    # 1. Initialize the base retriever
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": search_k})
    
    # 2. Initialize the Cross-Encoder model for reranking
    cross_encoder_model = HuggingFaceCrossEncoder(model_name='cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    # 3. Initialize the LangChain reranker
    reranker = CrossEncoderReranker(model=cross_encoder_model, top_n=reranker_top_n)
    
    # 4. Create the compression retriever
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker, 
        base_retriever=base_retriever
    )
    
    return compression_retriever