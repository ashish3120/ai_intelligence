import os
import shutil
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Constants
PERSIST_DIR = r"D:\ai_intelligence\personal-kb\vectorstore\faiss_index"
INDEX_NAME = "index"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def get_embeddings():
    """Returns the HuggingFace embeddings model."""
    print(f"Loading embeddings model: {EMBEDDING_MODEL_NAME}...")
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def get_vectorstore():
    """
    Returns the FAISS vector store.
    If it exists on disk, load it. Otherwise, return None (caller must handle creation).
    """
    embeddings = get_embeddings()
    
    if os.path.exists(PERSIST_DIR) and os.path.exists(os.path.join(PERSIST_DIR, f"{INDEX_NAME}.faiss")):
        print(f"Loading existing FAISS index from {PERSIST_DIR}...")
        # allow_dangerous_deserialization is needed if we trust the source (we created it).
        return FAISS.load_local(PERSIST_DIR, embeddings, index_name=INDEX_NAME, allow_dangerous_deserialization=True)
    
    print("No existing FAISS index found.")
    return None

def create_new_store(chunks: List[Document]):
    """Creates a new FAISS index from chunks."""
    embeddings = get_embeddings()
    print(f"Creating new FAISS index with {len(chunks)} chunks...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

def add_documents_to_store(chunks: List[Document]):
    """
    Adds documents to the vector store.
    Since FAISS is in-memory mostly until saved, we usually load, add, save.
    For local RAG, re-building or merging is common. 
    To be safe and simple: We will load existing if available, add, then save.
    """
    vectorstore = get_vectorstore()
    
    if vectorstore is None:
        vectorstore = create_new_store(chunks)
    else:
        print(f"Adding {len(chunks)} new chunks to existing index...")
        vectorstore.add_documents(chunks)
    
    # Persist
    print(f"Saving FAISS index to {PERSIST_DIR}...")
    if not os.path.exists(PERSIST_DIR):
        os.makedirs(PERSIST_DIR)
        
    vectorstore.save_local(PERSIST_DIR, index_name=INDEX_NAME)
    print("Index saved.")
