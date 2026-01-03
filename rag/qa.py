import time
from typing import Dict, Any, Generator
from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

from rag.prompt import get_prompt_template
from rag.retriever import get_retriever
from utils.confidence import calculate_confidence

# Model configuration
MODEL_NAME = "llama3.1"

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

class LocalRAGQA:
    def __init__(self, vectorstore, model_name=MODEL_NAME, base_url=None):
        self.vectorstore = vectorstore
        self.retriever = get_retriever(vectorstore)
        self.prompt = get_prompt_template()
        
        # Initialize Ollama LLM
        print(f"Initializing local LLM: {model_name}...")
        if base_url:
            print(f"Using remote Ollama API at: {base_url}")
            self.llm = OllamaLLM(
                model=model_name, 
                base_url=base_url,
                headers={"ngrok-skip-browser-warning": "true"}
            )
        else:
            self.llm = OllamaLLM(model=model_name)
        
        self.chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def stream_answer(self, question: str) -> Generator[str, None, None]:
        """
        Generator that streams the answer chunks.
        """
        # 1. Retrieval
        # FAISS search_with_score returns L2 distance (lower is better) or inner product depending on index.
        # Default HuggingFaceEmbeddings + FAISS usually uses L2 distance (Euclidean).
        docs_with_scores = self.vectorstore.similarity_search_with_score(question, k=3)
        context_text = "\n\n".join([doc.page_content for doc, _ in docs_with_scores])
        
        # 2. Stream Generation
        prompt_val = self.prompt.invoke({"context": context_text, "question": question})
        
        for chunk in self.llm.stream(prompt_val):
            yield chunk
            
        # 3. Yield Metadata
        confidence_label, confidence_score, explanation = calculate_confidence(docs_with_scores)
        
        metadata_str = "\n\n--- METADATA ---\n"
        metadata_str += f"Confidence: {confidence_label} ({confidence_score:.1f}%)\n"
        metadata_str += f"Explanation: {explanation}\n"
        metadata_str += "Sources:\n"
        for i, (doc, score) in enumerate(docs_with_scores):
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            metadata_str += f"{i+1}. {os.path.basename(source)} (Page {page}) - Dist: {score:.4f}\n"
            
        yield metadata_str
