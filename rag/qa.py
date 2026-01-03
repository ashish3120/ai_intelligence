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
    def __init__(self, vectorstore, model_name=MODEL_NAME, base_url=None, mode="standard", verbose=False):
        self.vectorstore = vectorstore
        self.retriever = get_retriever(vectorstore)
        self.mode = mode
        self.verbose = verbose
        # Select prompt based on mode
        self.prompt = get_prompt_template(mode)
        
        # Initialize Ollama LLM
        print(f"Initializing local LLM: {model_name} (Mode: {mode})...")
        if base_url:
            print(f"Using remote Ollama API at: {base_url}")
            self.llm = OllamaLLM(
                model=model_name, 
                base_url=base_url,
                headers={"ngrok-skip-browser-warning": "true"}
            )
        else:
            self.llm = OllamaLLM(model=model_name)
    
    def stream_answer(self, question: str) -> Generator[str, None, None]:
        """
        Generator that streams the answer chunks.
        """
        # 1. Retrieval
        docs_with_scores = self.vectorstore.similarity_search_with_score(question, k=5)
        
        # 2. Confidence Check (Strong No-Answer Detection)
        conf_result = calculate_confidence(docs_with_scores)
        
        if self.verbose:
            print("\n[DIAGNOSTICS]")
            print(f"Top Score: {conf_result['details']['top_1']:.4f}")
            print(f"Avg Score: {conf_result['details']['average']:.4f}")
            print(f"Safe to Answer: {conf_result['safe_to_answer']}")
            print("-" * 20)

        # Skip generation if unsafe
        if not conf_result['safe_to_answer']:
            yield "I cannot find the answer in the provided documents (Confidence too low)."
            # Yield metadata anyway so user sees why
            yield self._format_metadata(conf_result, docs_with_scores)
            return

        # 3. Stream Generation
        context_text = "\n\n".join([doc.page_content for doc, _ in docs_with_scores])
        prompt_val = self.prompt.invoke({"context": context_text, "question": question})
        
        for chunk in self.llm.stream(prompt_val):
            yield chunk
            
        # 4. Yield Metadata
        yield self._format_metadata(conf_result, docs_with_scores)

    def _format_metadata(self, conf_result, docs_with_scores):
        """Formats the metadata footer with source grouping."""
        metadata_str = "\n\n--- METADATA ---\n"
        metadata_str += f"Confidence: {conf_result['label']} ({conf_result['score']:.1f}%)\n"
        metadata_str += f"Explanation: {conf_result['explanation']}\n"
        metadata_str += "Sources:\n"
        
        # Group by Source File
        grouped_sources = {}
        for doc, score in docs_with_scores:
            source = os.path.basename(doc.metadata.get('source', 'Unknown'))
            page = doc.metadata.get('page', 'N/A')
            if source not in grouped_sources:
                grouped_sources[source] = []
            grouped_sources[source].append(f"Page {page} (Dist: {score:.4f})")
            
        # Format Grouped Sources
        for i, (source, details) in enumerate(grouped_sources.items()):
            metadata_str += f"{i+1}. {source}\n"
            for detail in details:
                metadata_str += f"   - {detail}\n"
                
        return metadata_str
