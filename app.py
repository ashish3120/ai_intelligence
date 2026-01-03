import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import sys
import argparse
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ingest.loader import load_documents
from ingest.chunker import chunk_documents
from vectorstore.faiss_store import add_documents_to_store, get_vectorstore
from rag.qa import LocalRAGQA

DOCS_DIR = r"D:\ai_intelligence\personal-kb\data\docs"

def ingest_data():
    """Runs the ingestion pipeline."""
    print(Fore.CYAN + "=== Starting Ingestion Process ===")
    
    docs = load_documents(DOCS_DIR)
    if not docs:
        print(Fore.RED + "No documents found to ingest.")
        return

    chunks = chunk_documents(docs)
    
    # FAISS store handles creation/updating
    add_documents_to_store(chunks)
    print(Fore.GREEN + "=== Ingestion Complete ===")

def query_loop(api_url=None, mode="standard", verbose=False):
    """Starts the interactive query loop."""
    print(Fore.CYAN + "=== Personal Knowledge Base (Local RAG - FAISS) ===")
    print(Fore.YELLOW + f"Loading Vector Database & Model (Mode: {mode})...")
    
    try:
        vectorstore = get_vectorstore()
        if vectorstore is None:
             print(Fore.RED + "Index not found! Please run 'python app.py ingest' first.")
             return
        qa_system = LocalRAGQA(vectorstore, base_url=api_url, mode=mode, verbose=verbose)
    except Exception as e:
        print(Fore.RED + f"Failed to initialize system: {e}")
        return

    print(Fore.GREEN + "System Ready! Type 'exit' to quit.")
    print("-" * 50)

    while True:
        try:
            user_input = input(Fore.BLUE + "\nQuery: " + Style.RESET_ALL).strip()
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("Goodbye!")
                break
            if not user_input:
                continue
                
            print(Fore.CYAN + "Thinking...", end= " ", flush=True)
            print()
            
            for chunk in qa_system.stream_answer(user_input):
                print(chunk, end="", flush=True)
            print("\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(Fore.RED + f"\nError during query: {e}")

def main():
    parser = argparse.ArgumentParser(description="Local Personal Knowledge Base RAG")
    parser.add_argument('mode', nargs='?', choices=['ingest', 'chat'], default='chat', help="Mode: 'ingest' or 'chat'")
    parser.add_argument('--query', '-q', type=str, help="Run a single query and exit")
    parser.add_argument('--api-url', type=str, help="URL for remote Ollama server (e.g. ngrok URL)")
    parser.add_argument('--query-mode', type=str, default="standard", choices=["standard", "summarize", "explain_simple", "exam"], help="Query interaction mode")
    parser.add_argument('--verbose', '-v', action='store_true', help="Enable verbose diagnostics")
    
    args = parser.parse_args()
    
    if args.mode == 'ingest':
        ingest_data()
    else:
        if args.query:
            # Single shot
            print(Fore.CYAN + "=== Personal Knowledge Base (One-shot) ===")
            vectorstore = get_vectorstore()
            if vectorstore is None:
                print(Fore.RED + "Index not found!")
                return
            qa_system = LocalRAGQA(vectorstore, base_url=args.api_url, mode=args.query_mode, verbose=args.verbose)
            print(Fore.CYAN + f"Query: {args.query}")
            print(Fore.CYAN + "Thinking...", end= " ", flush=True)
            print()
            for chunk in qa_system.stream_answer(args.query):
                print(chunk, end="", flush=True)
            print("\n")
        else:
            query_loop(api_url=args.api_url, mode=args.query_mode, verbose=args.verbose)

if __name__ == "__main__":
    main()
