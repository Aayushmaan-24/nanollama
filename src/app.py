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
    
@app.post("/generate")
def generate(request: GenerateRequest):
    """Non-streaming generation with full latency breakdown in response."""
    validate_model(request.model)
    
    ram_before = psutil.virtual_memory().used / 1024 / 1024
    start = time.perf_counter()
    first_token_time = None
    full_response = []
    token_count = 0
    
    try:
        with requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": request.model,
                "prompt": request.prompt,
                "stream": True,
                "options" : {
                    "temperature" : request.temperature,
                    "num_predict" : request.max_tokens,
                },
            },
            stream=True,
            timeout=120,
        ) as response:
            for line in response.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if first_token_time is None and chunk.get("response"):
                    first_token_time = (time.perf_counter()-start)*1000
                full_response.append(chunk.get("response",""))
                if chunk.get("done"):
                    token_count = chunk.get("eval_count", 0)
                    break
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    total_ms = (time.perf_counter()-start)*1000
    ram_after = psutil.virtual_memory().used / 1024 / 1024
    
    return {
        "model": request.model,
        "response" : "".join(full_response).strip(),
        "latency" : {
            "ttfs_ms":round(first_token_time or 0, 1),
            "total_ms":round(total_ms, 1),
            "tokens_per_sec":round(token_count / (total_ms/1000), 1),
            "token_count":token_count,
            "ram_delta_mb":round(ram_after-ram_before, 1),
        },
    }