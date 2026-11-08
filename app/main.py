import os
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

from app.rag_engine import HybridFastRAGEngine

load_dotenv()

app = FastAPI(
    title="FDE Ultra-Fast Production RAG v2.0",
    description="Enterprise RAG with RBAC Metadata Filtering, BM25+Vector RRF Hybrid Search, and Stage-2 Reranking",
    version="2.0.0",
)

engine = HybridFastRAGEngine(storage_dir="data")

static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


def resolve_credentials(provider_override: Optional[str], api_key_override: Optional[str]):
    provider = (provider_override or os.getenv("AI_PROVIDER", "gemini")).lower().strip()
    if api_key_override and api_key_override.strip():
        return provider, api_key_override.strip()

    if provider == "gemini":
        key = os.getenv("GEMINI_API_KEY", "").strip()
        if not key or key == "your_gemini_api_key_here":
            raise HTTPException(
                status_code=400,
                detail="Missing Gemini API Key. Add GEMINI_API_KEY to your .env file or paste it in the UI top bar.",
            )
        return provider, key

    if provider == "openai":
        key = os.getenv("OPENAI_API_KEY", "").strip()
        if not key or key == "your_openai_api_key_here":
            raise HTTPException(
                status_code=400,
                detail="Missing OpenAI API Key. Add OPENAI_API_KEY to your .env file or paste it in the UI top bar.",
            )
        return provider, key

    raise HTTPException(status_code=400, detail=f"Invalid provider '{provider}'. Choose 'gemini' or 'openai'.")


class QueryRequest(BaseModel):
    question: str
    top_k: int = 4
    user_department: str = "all"
    user_clearance: str = "restricted"
    source_filter: Optional[str] = "all"
    provider: Optional[str] = None
    api_key: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    index_path = os.path.join(static_dir, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/status")
async def get_status():
    docs = {}
    for c in engine.chunks:
        src = c["source"]
        if src not in docs:
            docs[src] = {
                "name": src,
                "chunks": 0,
                "department": c.get("department", "general"),
                "clearance": c.get("clearance", "public"),
            }
        docs[src]["chunks"] += 1
    return {
        "total_chunks": len(engine.chunks),
        "documents": list(docs.values()),
        "default_provider": os.getenv("AI_PROVIDER", "gemini"),
    }


@app.post("/api/ingest")
async def ingest_file(
    file: UploadFile = File(...),
    provider: Optional[str] = Form(None),
    api_key: Optional[str] = Form(None),
    chunk_size: int = Form(700),
    chunk_overlap: int = Form(120),
    department: str = Form("general"),
    clearance: str = Form("internal"),
):
    prov, key = resolve_credentials(provider, api_key)
    try:
        raw_bytes = await file.read()
        stats = engine.ingest_document(
            filename=file.filename or "uploaded_document.txt",
            file_bytes=raw_bytes,
            provider=prov,
            api_key=key,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            department=department,
            clearance=clearance,
        )
        return {"status": "ok", "stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ask")
async def ask_question(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    prov, key = resolve_credentials(req.provider, req.api_key)
    try:
        result = engine.answer_question(
            query=req.question,
            provider=prov,
            api_key=key,
            top_k=req.top_k,
            user_department=req.user_department,
            user_clearance=req.user_clearance,
            source_filter=req.source_filter,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/clear")
async def clear_knowledge_base():
    engine.clear_index()
    return {"status": "cleared", "total_chunks": 0}
