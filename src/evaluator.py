"""
evaluator.py — LLM-as-judge quality scoring for local model outputs
Scores each response 0.0-1.0 on: correctness, completeness, format
Uses Ollama itself as the judge (fully offline)
Run: python -m src.evaluator
"""

import json
import os
import requests
from rich.console import Console
from rich.table import Table
from prompts import PROMPTS
import re

console = Console()

OLLAMA_URL  = "http://localhost:11434"
JUDGE_MODEL = "qwen2.5:3b"  # fastest model judges the others
RESULTS_DIR = "results"

# ── 1. Judge a single response ─────────────────────────────────────

def judge_response(prompt: str, response: str, task: str) -> dict:
    """Score a response on correctness, completeness, format (0.0-1.0 each)."""
    judge_prompt = f"""You are an objective evaluator. Score this AI response on three dimensions.
Respond with ONLY a JSON object, nothing else.

TASK TYPE: {task}
QUESTION: {prompt}
RESPONSE: {response}

Score each dimension from 0.0 to 1.0:
- correctness: Is the answer factually correct and accurate?
- completeness: Does it fully address what was asked?
- format: Is it well-structured and appropriate for the task type?

For "structured" tasks, format score should reflect valid JSON output.
For "code" tasks, format score should reflect working, clean code.

Respond with ONLY this JSON, no explanation:
{{"correctness": 0.0, "completeness": 0.0, "format": 0.0}}"""

    try:
        resp = requests.post(
           f"{OLLAMA_URL}/api/generate", 
           json = {
               "model" : JUDGE_MODEL,
               "prompt": judge_prompt,
               "stream" : False,
               "options" : {
                   "temperature" : 0.0,
                   "num_predict" : 256,
               },
               
           },
           timeout=60
        )
        raw = resp.json().get("response","").strip()
        
        # extract JSON from the response
        match = re.search(r'\{[^}]+\}', raw)
        if match:
            scores = json.loads(match.group())
            return {
                "correctness":  round(float(scores.get("correctness", 0)), 2),
                "completeness": round(float(scores.get("completeness", 0)), 2),
                "format":       round(float(scores.get("format", 0)), 2),
                 "overall":      round((
                    float(scores.get("correctness", 0)) +
                    float(scores.get("completeness", 0)) +
                    float(scores.get("format", 0))
                ) / 3, 2),
            }
    except Exception as e:
        console.print(f"[red]Judge error: {e}[/red]")
    
    return {"correctness": 0.0, "completeness": 0.0, "format": 0.0, "overall": 0.0}

# ── 2. Load latest benchmark results ──────────────────────────────

def load_latest_benchmark() -> list[dict]:
    files = sorted([
        f for f in os.listdir(RESULTS_DIR)
        if f.startswith("benchmark_") and f.endswith(".json")
    ])
    if not files:
        raise FileNotFoundError("No benchmark results found. Run `python -m src.benchmark` first.")
    path = os.path.join(RESULTS_DIR, files[-1])
    console.print(f"[cyan]Loading benchmark: {path}[/cyan]")
    with open(path) as f:
        return json.load(f)
    
# ── 3. Full quality evaluation ─────────────────────────────────────

def run_quality_eval(results: list[dict]) -> list[dict]:
    """Score every response in the benchmark results."""
    
    scored = []
    total = len(results)
    
    console.rule("[bold yellow]Quality Evaluation[/bold yellow]")
    console.print(f"Judge model : {JUDGE_MODEL}")
    console.print(f"Responses   : {total}\n")

    for i, result in enumerate(results):
        if "error" in result or not result.get("response"):
            continue
        console.print(
            f"  [{i+1:02d}/{total}] {r['model']:15s} "
            f"[{r['difficulty']}] {r['prompt'][:45]}..."
        )
        
        quality = judge_response(result['prompt'], result['response'], result['task'])
        entry = {**result, "quality": quality}
        scored.append(entry)
        
        console.print(
            f"           correct={quality['correctness']:.2f}  "
            f"complete={quality['completeness']:.2f}  "
            f"format={quality['format']:.2f}  "
            f"overall={quality['overall']:.2f}"
        )
        
        # save scored results
        path = os.path.join(RESULTS_DIR, "quality_scores.json")
        with open(path, "w") as f:
            json.dump(scored, f, indent=2)
        console.print(f"\n[green]✓ Quality scores saved → {path}[/green]")
        return scored