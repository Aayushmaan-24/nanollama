"""
benchmark.py — Measure tokens/sec, TTFT, RAM per model
Run: python -m src.benchmark
"""

import json
import time
import os
import psutil
from datetime import datetime
from rich.console import Console
from rich.table import Table
from prompts import PROMPTS
import requests

console = Console()

OLLAMA_URL = "http://localhost:11434"
MODELS     = ["llama3.2:3b", "gemma2:2b", "qwen2.5:3b"]
RESULTS_DIR = "results"

# ── 1. Single inference with metrics ──────────────────────────────

def run_inference(model: str, prompt: str) -> dict:
    """
    Run one prompt, capture:
    - TTFT: time to first token (ms)
    - total_time: full response time (ms)
    - tokens_per_sec: generation speed
    - ram_used_mb: RAM delta during inference
    - response: full text
    """
    
    proc = psutil.Process(os.getpid())
    ram_before = psutil.virtual_memory().used / 1024 / 1024  # in MB
    
    payload = {
        "model" : model,
        "prompt" : prompt,
        "stream" : True,
        "options" : {
            "temperature" : 0.1,
            "num_predict" : 256,
        },
    }
    
    first_token_time = None
    full_respone = []
    token_count = 0
    start = time.perf_counter()
    
    try:
        with requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            stream=True,
            timeout=120,
        ) as resp:
            for line in resp.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
            
                if first_token_time is None and chunk.get("response"):
                    first_token_time = (time.perf_counter() - start) * 1000 #ms
                    
                full_respone.append(chunk.get("response", ""))
                
                if chunk.get("done"):
                    token_count = chunk.get("eval_count", 0)
                    break
                
    except Exception as e:
        return {"error":str(e)}
    
    total_time = (time.perf_counter() - start) * 1000 #ms
    ram_after = psutil.virtual_memory().used / 1024 / 1024
    
    return {
        "model" : model,
        "ttft_ms" : first_token_time,
        "total_time_ms" : round(total_time,1),
        "tokens_per_sec" : round(token_count / (total_time / 1000),1) if total_time > 0 else 0,
        "token_count" : token_count,
        "ram_delta_mb" : round(ram_after - ram_before, 1),
        "response" : "".join(full_respone).strip(),
    }
    
# ── 2. Warmup ─────────────────────────────────────────────────────

def warmup(model: str) -> None:
    """Run a throwaway prompt so model is loaded into memory."""
    console.print(f"  [dim]Warmin up {model}... [/dim] ")
    run_inference(model, "Hello, how are you?")
    
