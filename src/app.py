"""
app.py — FastAPI app with model switching, streaming, and latency headers
Run: uvicorn src.app:app --reload --port 8000
"""

import json
import time
import psutil
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel

OLLAMA_URL = "http://localhost:11434"
MODELS     = ["llama3.2:3b", "gemma2:2b", "qwen2.5:3b"]

app = FastAPI(
    title="LocalSLM",
    description="Offline LLM inference with model switching and latency tracking",
    version="1.0.0",
)

# ── Request/Response models ────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt:      str
    model:       str  = "qwen2.5:3b"
    max_tokens:  int  = 256
    temperature: float = 0.1
    
# ── Helpers ────────────────────────────────────────────────────────

def validate_model(model: str) -> None:
    if model not in MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model}' not available. Choose from: {MODELS}"
        )

# ── Routes ─────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service":  "LocalSLM",
        "models":   MODELS,
        "endpoints": ["/generate", "/generate/stream", "/models", "/health", "/ui"],
    }
    
@app.get("/health")
def health():
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        ollama_ok = response.status_code == 200
    except Exception:
        return {
            "status":"ok" if ollama_ok else "degraded",
            "ollama": ollama_ok,
            "ram_used": round(psutil.virtual_memory().used / (1024 ** 3), 2),
            "ram_total": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        }
        
@app.get("/models")
def list_models():
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        available_models = [m['name'] for m in response.json().get("models",[])]
    except Exception:
        available_models = []
    return {
        "configured" : MODELS,
        "available" : available_models
    }