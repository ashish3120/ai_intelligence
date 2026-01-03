from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def chunk_documents(documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
    """
    Splits documents into smaller chunks.
    
    Args:
        documents (List[Document]): The list of documents to split.
        chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): The overlap between chunks.
        
    Returns:
        List[Document]: A list of chunked documents.
    """
    print(f"Chunking {len(documents)} documents with size={chunk_size}, overlap={chunk_overlap}...")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = splitter.split_documents(documents)
    
    # Enrich metadata if needed (e.g. ensure chunk index is present if we were doing custom indexing)
    # LangChain handles basic metadata copy.
    
    print(f"Created {len(chunks)} chunks.")
    return chunks
