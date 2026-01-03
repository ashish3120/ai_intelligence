# Implementation Plan - Advanced RAG Features

## 1. Enhanced Confidence & No-Answer Detection
- **File**: `utils/confidence.py`
- **Goal**: Implement logic to detect when documents are irrelevant (high distance) and skip generation.
- **Logic**:
    - Input: FAISS distances (lower is better).
    - Thresholds: 
        - Distance > 0.8: Unsafe/No Answer.
        - Distance < 0.5: High Confidence.
- **Output**: Return structured confidence stats.

## 2. Query Modes
- **File**: `rag/prompt.py`
- **Goal**: Support multiple interaction styles.
- **Modes**:
    - `standard` (default)
    - `summarize`: "Summarize the key points of..."
    - `explain_simple`: "Explain to a 5 year old..."
    - `exam`: "Generate 3 quiz questions based on..."

## 3. QA Logic Update
- **File**: `rag/qa.py`
- **Goal**: Integrate the above features.
- **Changes**:
    - Check confidence *before* LLM call. If unsafe, yield "I cannot find the answer..." and stop.
    - Group sources by filename and page number.
    - Add `verbose` logging for retrieved chunks (Diagnostics).
    - Allow prompt template switching based on mode.

## 4. CLI Update
- **File**: `app.py`
- **Goal**: Expose new features to user.
- **Changes**:
    - Add `--mode` argument.
    - Add `--verbose` argument.

## 5. Chunking Review
- **File**: `ingest/chunker.py`
- **Goal**: Ensure chunk size is easily configurable (Task 6).
