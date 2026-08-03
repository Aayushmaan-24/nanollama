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
from collections import defaultdict

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
            f"  [{i+1:02d}/{total}] {result['model']:15s} "
            f"[{result['difficulty']}] {result['prompt'][:45]}..."
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
    
# ── 4. Summary table ───────────────────────────────────────────────

def print_quality_summary(scored: list[dict]) -> None:
    
    stats = defaultdict(lambda: {
        "correctness": [], "completeness": [], "format": [], "overall": []
    })
    
    for result in scored:
        q = result.get("quality", {})
        m = result['model']
        for key in ["correctness", "completeness", "format", "overall"]:
            if key in q:
                stats[m][key].append(q[key])
                
    def avg(lst):
        return round(sum(lst) /  len(lst), 3) if lst else 0
    
    table = Table(title="Quality Scores (LLM-as-Judge)", show_header=True)
    table.add_column("Model",        style="cyan")
    table.add_column("Correctness",  justify="right")
    table.add_column("Completeness", justify="right")
    table.add_column("Format",       justify="right")
    table.add_column("Overall",      justify="right")
    table.add_column("Samples",      justify="right")
    
    from prompts import PROMPTS
    model_order = ["llama3.2:3b", "gemma2:2b", "qwen2.5:3b"]
    for model in model_order:
        s = stats[model]
        if not s['overall']:
            continue
        table.add_row(
            model,
            str(avg(s["correctness"])),
            str(avg(s["completeness"])),
            str(avg(s["format"])),
            str(avg(s["overall"])),
            str(len(s["overall"])),
        )
        
        console.print(table)
        
# ── 5. Tradeoff matrix ─────────────────────────────────────────────

def print_tradeoff_matrix(benchmark: list[dict], scored: list[dict]) -> None:
    """Combine speed + quality into final recommendation matrix."""
    
    speed = defaultdict(list)
    quality = defaultdict(list)
    
    for result in benchmark:
        if "error" not in result:
            speed[result['model']].append(result.get('tokens_per_sec', 0))
            
    for result in scored:
        q = result.get("quality", {})
        if q:
            quality[result['model']].append(q.get('overall', 0))
            
    def avg(lst): 
        return round(sum(lst)/len(lst), 2) if lst else 0
    
    console.rule("[bold yellow]Tradeoff Matrix[/bold yellow]")
    console.print(f"{'Model':20s} {'Tok/s':>8} {'Quality':>9} {'Verdict'}")
    console.print("─" * 60)
    
    rows = []
    for model in ["llama3.2:3b", "gemma2:2b", "qwen2.5:3b"]:
        s = avg(speed[model])
        q = avg(quality[model])
        rows.append((model, s, q))
    
    best_speed   = max(r[1] for r in rows)
    best_quality = max(r[2] for r in rows)

    for model, s, q in rows:
        if s == best_speed:
            verdict = "⚡ Fastest"
        elif q == best_quality:
            verdict = "🎯 Best quality"
        else:
            verdict = "⚖️  Balanced"
        console.print(f"  {model:18s} {s:>8.1f} {q:>9.3f}   {verdict}")

    console.print("\n[dim]Use fastest for latency-critical apps.[/dim]")
    console.print("[dim]Use best quality for accuracy-critical tasks.[/dim]")

# ── 6. Main ───────────────────────────────────────────────────────

if __name__ == "__main__":
    benchmark = load_latest_benchmark()
    scored    = run_quality_eval(benchmark)
    print_quality_summary(scored)
    print_tradeoff_matrix(benchmark, scored)