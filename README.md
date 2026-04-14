# Personal Knowledge Base (RAG) - Groq Edition

**A lightning-fast Retrieval-Augmented Generation (RAG) system running on Groq.**  
Privately index your documents locally, and use Groq's high-speed inference endpoints to query them interactively. 

---

## 🚀 Features
- **Privacy First (Local Storage)**: Your documents (`.pdf`, `.txt`, `.md`) and their FAISS vector representations are processed and stored locally on your machine.
- **Lightning Fast Inference**: By leveraging the Groq API (`llama-3.1-8b-instant`), the LLM inference is near-instantaneous.
- **Smart Retrieval**: Uses FAISS vector search with `all-MiniLM-L6-v2` embeddings for fast, accurate context retrieval.
- **Source Citations & Confidence**: Every answer includes a confidence score, thresholds, and exact source citations so you know exactly where the information came from.

## 🛠️ Prerequisites
- **Python 3.10+**
- **Groq API Key**: Get one for free from the [Groq Console](https://console.groq.com/keys).

## 📦 Installation

1.  **Clone or Navigate to the Project**:
    ```bash
    cd D:\ai_intelligence
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Setup your API Key**:
    - Create a new file in the root directory named `.env`.
    - Add your Groq API key inside:
      ```env
      GROQ_API_KEY=gsk_YOUR_GROQ_API_KEY_HERE
      ```

---

## 🏃 Usage

### 1. Add Your Data
Place your documents inside the `data/docs` folder:
- Path: `D:\ai_intelligence\data\docs\`
- Supported formats: **PDF, TXT, Markdown**.

### 2. Ingest Data (Build/Update Knowledge Base)
Run this command whenever you add or modify files. It creates and updates the vector indices locally based on your documents.
```bash
python app.py ingest
```

### 3. Run the Chat Interface
Once your data is ingested, start the interactive chat terminal to query your data seamlessly:
```bash
python app.py
```

---

## 🔍 Examples

**Single Terminal Query**:
```bash
python app.py -q "What is the secret code for this project?"
```

**Query Interaction Output Example**:
```text
Query: What is the secret code for this project?
Thinking... 
Based on the provided context, the secret code for this project is ALPHA-TANGO-77.

--- METADATA ---
Confidence: High (98.2%)
Explanation: Top match distance: 0.8521 (Threshold: 1.6). Avg distance: 0.8521 across 1 chunks.
Sources:
1. test_doc.txt
   - Page N/A (Dist: 0.8521)
```

## 📂 Project Structure
- `app.py`: Main entry point (CLI & Chat loop).
- `rag/qa.py`: Core RAG logic tied to LangChain and Groq.
- `rag/retriever.py` & `rag/prompt.py`: Defines prompting strategies and FAISS retrieval.
- `ingest/`: Document loading and text-splitting (chunking) logic.
- `vectorstore/faiss_store.py`: Local FAISS database generation and management.
- `data/docs/`: **PUT YOUR FILES HERE TO BE SEARCHED**.

## ⚠️ Troubleshooting
- **No Index Found**: You must run `python app.py ingest` at least once before chatting. If you didn't ingest anything, the search algorithms will crash.
- **Protobuf Error**: If you see protocol buffer errors, the app handles this mostly via environment variables, but ensure packages like `google-api-core` are up to date if you modify it to use Google integration.
- **Model Decommissioned Error**: Groq frequently updates their models. If you encounter an error saying a model is decommissioned, head into `rag/qa.py` and change the `ChatGroq(model_name="...")` to the latest model name from the Groq docs.
