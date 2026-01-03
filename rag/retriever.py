from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever

def get_retriever(vectorstore: FAISS, k: int = 3) -> VectorStoreRetriever:
    """
    Returns a retriever object from the vector store.
    
    Args:
        vectorstore (FAISS): The FAISS vector store instance.
        k (int): Number of documents to retrieve.
        
    Returns:
        VectorStoreRetriever: The retriever.
    """
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
