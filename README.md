# Personal Knowledge Base (RAG) - Hybrid & Local

**A robust, privacy-focused RAG system designed for Windows.**  
Run fully local (16GB RAM) or partially on the cloud (Google Colab GPU) for faster inference, while keeping your documents private on your machine.

---

## 🚀 Features
- **Privacy First**: Your documents (`.pdf`, `.txt`, `.md`) are processed and stored locally on your `D:` drive. They never leave your computer.
- **Dual Mode**:
    - **Local Mode**: Uses your PC's 16GB RAM to run `llama3.1` (8B) or similar models via Ollama.
    - **Hybrid Mode**: Offload heavy LLM processing to Google Colab (Tesla T4 GPU) via free Ngrok tunnel, while keeping embeddings and vectors local.
- **Smart Retrieval**: Uses FAISS vector search with `all-MiniLM-L6-v2` embeddings for fast, accurate context.
- **Confidence Scoring**: Every answer includes a confidence score and source citations so you know where the info came from.

## 🛠️ Prerequisites
- **Python 3.10+**
- **Ollama** (for Local Mode)
- **Google Account** (for Hybrid Mode - Colab)
- **Ngrok Account** (Free tier is fine - for Hybrid Mode)

## 📦 Installation

1.  **Clone/Navigate to Project**:
    ```bash
    cd D:\ai_intelligence\personal-kb
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Setup Local Ollama (Optional - for Local Mode)**:
    - Install Ollama from [ollama.com](https://ollama.com).
    - Pull the default model:
      ```bash
      ollama pull llama3.1
      ```

---

## 🏃 Usage

### 1. Add Your Data
Place your documents documents in the `data/docs` folder:
- `D:\ai_intelligence\personal-kb\data\docs\`
- Supported formats: **PDF, TXT, Markdown**.

### 2. Ingest Data (Build/Update Knowledge Base)
Run this command whenever you add new files. It creates the vector index locally.
```bash
python app.py ingest
```

### 3. Run the Chat Interface

#### **Option A: Fully Local Mode (Requires 16GB RAM)**
Ensure Ollama is running (`ollama serve`), then:
```bash
python app.py
```

#### **Option B: Hybrid Mode (Google Colab - Recommended for Speed)**
1.  **Prepare Server**:
    - Upload `colab_ollama_server_v2.ipynb` to [Google Colab](https://colab.research.google.com).
    - Change Runtime type to **T4 GPU**.
    - Add your **Ngrok Authtoken** in the code cell.
    - Run all cells. Copy the output URL (e.g., `https://xyz.ngrok-free.app`).

2.  **Connect Locally**:
    ```bash
    python app.py --api-url "https://your-ngrok-url.ngrok-free.app"
    ```

---

## 🔍 Examples

**Hybrid Run Command**:
```bash
python app.py --api-url "https://odontophorous-renetta-unpoisoned.ngrok-free.dev"
```

**Query Interaction**:
```text
Query: what is the project name?

Thinking...
Project Alpha.

--- METADATA ---
Confidence: High (98.2%)
Explanation: Based on 1 sources with very low distance score.
Sources:
1. data1.txt (Page N/A) - Dist: 0.1234
```

## 📂 Project Structure
- `app.py`: Main entry point (CLI & Chat loop).
- `rag/`: Core RAG logic (Retrieval, Prompting, QA Chain).
- `ingest/`: Document loading and chunking logic.
- `vectorstore/`: FAISS database management.
- `data/docs/`: **PUT YOUR FILES HERE**.
- `colab_ollama_server_v2.ipynb`: Notebook to run the LLM server on Colab.

## ⚠️ Troubleshooting
- **Protobuf Error**: If you see protocol buffer errors, the app automatically handles this, but ensure `google-api-core` is up to date if issues persist.
- **No Index Found**: You must run `python app.py ingest` at least once before chatting.
- **Ngrok Errors**: Check if the Colab notebook is still running. Free tunnels expire after some time locally or if the Colab tab is closed.
