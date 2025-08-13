from sentence_transformers import CrossEncoder

# Preload this globally for performance
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query, docs, top_k=3):
    """
    Re-rank the list of documents based on semantic relevance to the query.
    Returns top_k reranked documents.
    """
    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)
    
    reranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, score in reranked[:top_k]]
    return top_docs
