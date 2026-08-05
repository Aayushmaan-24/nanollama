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