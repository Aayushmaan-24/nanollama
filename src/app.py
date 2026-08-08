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

@app.post("/generate/stream")
def generate_stream(request: GenerateRequest):
    """
    Streaming generation — returns tokens as SSE stream.
    Latency headers: X-TTFT-Ms, X-Model
    """
    
    validate_model(request.model)
    
    def generate_stream():
        start = time.perf_counter()
        first_token_time = None
        
        try:
            with requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model" : request.model,
                    "prompt" : request.prompt,
                    "stream" : True,
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
                    token = chunk.get("response", "")
                    
                    if first_token_time is None and token:
                        first_token_time = (time.perf_counter()-start)*1000
                        yield f"data : {json.dumps({'ttfr_ms': round(first_token_time, 1)})}\n\n"
                    
                    if token:
                        yield f"data : {json.dumps({'token': token})}\n\n"
                        
                    if chunk.get("done"):
                        total_ms = (time.perf_counter()-start)*1000
                        yield f"data : {json.dumps({'done': True, 'total_ms': round(total_ms, 1), 'token_count': chunk.get('eval_count',0)})}\n\n"
                        break
        except Exception as e:
            yield f"data : {json.dumps({'error': str(e)})}\n\n"
            
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "X-Model": request.model,
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
    
# ── Simple browser UI ──────────────────────────────────────────────

@app.get("/ui", response_class=HTMLResponse)
def ui():
    models_js = json.dumps(MODELS)
    return f"""<!DOCTYPE html>
<html>
<head>
  <title>LocalSLM</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: #0a0a0f; color: #e2e8f0; font-family: monospace; padding: 40px; }}
    h1 {{ color: #7c3aed; margin-bottom: 4px; font-size: 24px; }}
    p.sub {{ color: #64748b; font-size: 12px; margin-bottom: 32px; }}
    .row {{ display: flex; gap: 12px; margin-bottom: 16px; align-items: center; }}
    label {{ color: #94a3b8; font-size: 12px; margin-bottom: 6px; display: block; }}
    select, textarea, input {{
      background: #111118; border: 1px solid #1e1e2e; color: #e2e8f0;
      border-radius: 4px; padding: 10px; font-family: monospace; font-size: 13px;
      width: 100%;
    }}
    select {{ width: auto; padding: 10px 16px; }}
    textarea {{ height: 120px; resize: vertical; }}
    button {{
      background: #7c3aed; color: white; border: none; padding: 10px 24px;
      border-radius: 4px; font-family: monospace; font-weight: 700;
      font-size: 13px; cursor: pointer; letter-spacing: 1px;
    }}
    button:hover {{ background: #6d28d9; }}
    #output {{
      background: #111118; border: 1px solid #1e1e2e; border-radius: 4px;
      padding: 20px; min-height: 160px; margin-top: 20px;
      white-space: pre-wrap; line-height: 1.7; font-size: 13px;
    }}
    .metrics {{
      display: flex; gap: 16px; margin-top: 12px; flex-wrap: wrap;
    }}
    .metric {{
      background: #111118; border: 1px solid #1e1e2e; border-radius: 4px;
      padding: 8px 16px; font-size: 11px; color: #94a3b8;
    }}
    .metric span {{ color: #7c3aed; font-weight: 700; font-size: 14px; display: block; }}
  </style>
</head>
<body>
  <h1>LocalSLM ⚡</h1>
  <p class="sub">Offline inference · Model switching · Streaming · Zero API cost</p>

  <div class="row">
    <div>
      <label>Model</label>
      <select id="model">
        {''.join(f'<option value="{m}">{m}</option>' for m in MODELS)}
      </select>
    </div>
    <div style="flex:1">
      <label>Max tokens</label>
      <input type="number" id="max_tokens" value="256" style="width:100px">
    </div>
  </div>

  <div style="margin-bottom:16px">
    <label>Prompt</label>
    <textarea id="prompt" placeholder="Ask anything..."></textarea>
  </div>

  <button onclick="generate()">Generate ▶</button>

  <div id="output">Output will appear here...</div>

  <div class="metrics" id="metrics" style="display:none">
    <div class="metric"><span id="m-ttft">-</span>TTFT (ms)</div>
    <div class="metric"><span id="m-total">-</span>Total (ms)</div>
    <div class="metric"><span id="m-tps">-</span>Tok/s</div>
    <div class="metric"><span id="m-tokens">-</span>Tokens</div>
  </div>

  <script>
    async function generate() {{
      const prompt     = document.getElementById('prompt').value.trim();
      const model      = document.getElementById('model').value;
      const max_tokens = parseInt(document.getElementById('max_tokens').value);
      const output     = document.getElementById('output');
      const metrics    = document.getElementById('metrics');

      if (!prompt) return;

      output.textContent = '';
      metrics.style.display = 'none';
      document.querySelector('button').textContent = 'Generating...';

      let ttft = null, totalMs = null, tokenCount = 0;
      const start = performance.now();

      try {{
        const resp = await fetch('/generate/stream', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{prompt, model, max_tokens}})
        }});

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {{
          const {{done, value}} = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, {{stream: true}});
          const lines = buffer.split('\\n');
          buffer = lines.pop();

          for (const line of lines) {{
            if (!line.startsWith('data: ')) continue;
            const data = JSON.parse(line.slice(6));
            if (data.ttft_ms) ttft = data.ttft_ms;
            if (data.token)   output.textContent += data.token;
            if (data.done) {{
              totalMs    = data.total_ms;
              tokenCount = data.token_count;
            }}
          }}
        }}
      }} catch(e) {{
        output.textContent = 'Error: ' + e.message;
      }}

      document.querySelector('button').textContent = 'Generate ▶';
      document.getElementById('m-ttft').textContent   = ttft || '-';
      document.getElementById('m-total').textContent  = totalMs || Math.round(performance.now() - start);
      document.getElementById('m-tps').textContent    = tokenCount && totalMs ? Math.round(tokenCount / (totalMs/1000)) : '-';
      document.getElementById('m-tokens').textContent = tokenCount || '-';
      metrics.style.display = 'flex';
    }}

    document.getElementById('prompt').addEventListener('keydown', e => {{
      if (e.ctrlKey && e.key === 'Enter') generate();
    }});
  </script>
</body>
</html>"""