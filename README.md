# FDE Ultra-Fast Production RAG (FastAPI + Hybrid Search UI)

A production-pattern Retrieval-Augmented Generation (RAG) web application engineered for Forward Deployed Engineers (FDEs).

## Why This Architecture Is Built for Speed & Reliability
1. **Sentence-Aware Overlapping Chunker (`app/rag_engine.py`)**: Extracts text page-by-page from PDFs (`pypdf`), Word docs (`python-docx`), and text files without slicing words or sentences in half, preserving exact page numbers.
2. **Sub-Millisecond Hybrid Search**: L2-normalized vector embeddings stored in contiguous `NumPy` float32 matrices execute exact cosine similarity via matrix dot-product (`np.dot`) in `< 1 ms` for thousands of chunks, boosted by lexical keyword matching so exact policy numbers, product codes, and acronyms never get missed.
3. **Strict Grounding & Inline Citations**: The LLM is instructed to cite `[Source X]` inline for every claim and refuse to guess when facts are absent from retrieved chunks.
4. **3-Pane FDE Observability Console (`app/static/index.html`)**: Upload documents on the left, ask questions in the center, and inspect exact retrieved chunks, similarity scores, and retrieval vs. LLM latency on the right.

---

## Step-by-Step Setup & Run Guide

### 1. Unzip & Enter the Project Folder
```bash
unzip fde_production_rag_app.zip
cd fde-fast-rag
```

### 2. Create a Virtual Environment & Install Dependencies
```bash
python -m venv .venv

# Activate on Windows:
.venv\Scripts\activate

# Activate on macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Add Your API Key (2 Ways)
**Option A — Via `.env` file (Recommended for permanent setup):**
1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` in your code editor and paste your key:
   - For **Google Gemini**: set `AI_PROVIDER=gemini` and `GEMINI_API_KEY=AIzaSy...`
   - For **OpenAI**: set `AI_PROVIDER=openai` and `OPENAI_API_KEY=sk-...`

**Option B — Directly in the Web UI (Zero config):**
- Leave `.env` untouched, start the server, and paste your Gemini or OpenAI API key directly into the **top-right password box** in the browser UI.

### 4. Start the FastAPI Server
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Open the Production UI
Open your browser to:
```
http://localhost:8000
```
