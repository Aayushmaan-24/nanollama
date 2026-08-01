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