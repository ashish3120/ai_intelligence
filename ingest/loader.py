import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_core.documents import Document

def load_documents(source_dir: str) -> List[Document]:
    """
    Loads PDF, TXT, and MD files from the specified directory.
    
    Args:
        source_dir (str): Path to the directory containing documents.
        
    Returns:
        List[Document]: A list of loaded LangChain documents.
    """
    documents = []
    
    if not os.path.exists(source_dir):
        print(f"Error: Directory {source_dir} not found.")
        return []
        
    for root, _, files in os.walk(source_dir):
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()
            
            try:
                if ext == ".pdf":
                    print(f"Loading PDF: {file}")
                    loader = PyPDFLoader(file_path)
                    documents.extend(loader.load())
                elif ext == ".txt":
                    print(f"Loading Text: {file}")
                    loader = TextLoader(file_path, encoding='utf-8')
                    documents.extend(loader.load())
                elif ext == ".md":
                    print(f"Loading Markdown: {file}")
                    # TextLoader often works better for simple MD than Unstructured if not installed
                    loader = TextLoader(file_path, encoding='utf-8') 
                    documents.extend(loader.load())
                else:
                    # Skip unsupported files
                    pass
            except Exception as e:
                print(f"Failed to load {file}: {e}")
                
    print(f"Total documents loaded: {len(documents)}")
    return documents
