import os
import sys
import json
import time
import requests
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.metrics import evaluate_response
from timeline.builder import build_timeline
from data.ingest import run_ingestion

API_URL = "http://127.0.0.1:8000"

def baseline_predict(case, tables):
    """
    Non-AI Baseline: simple filtering by subject_id + keyword matching.
    """
    subject_id = case.get("subject_id")
    query = case.get("question", "").lower()
    
    if subject_id is None:
        return {"answer": "I cannot find evidence for that in the structured record.", "citations": [], "abstained": True}
        
    events = build_timeline(tables, subject_id)
    
    # Simple heuristics based on query keywords
    matches = []
    citations = []
    
    for e in events:
        desc = e.description.lower()
        # If it's a diagnosis question, look for diagnoses
        if "diagnos" in query and e.event_type == "Diagnosis":
            if desc in query or any(word in desc for word in query.split() if len(word) > 4):
                matches.append(e.description)
                citations.append({"citation": e.citation_key()})
        # If medication
        elif "prescrib" in query or "medicat" in query and e.event_type == "Medication":
            if desc in query or any(word in desc for word in query.split() if len(word) > 4):
                matches.append(e.description)
                citations.append({"citation": e.citation_key()})
        # If admission
        elif "admiss" in query or "admit" in query and e.event_type == "Admission":
            matches.append(e.description)
            citations.append({"citation": e.citation_key()})
            
    if not matches:
        return {"answer": "I cannot find evidence for that in the structured record.", "citations": [], "abstained": True}
        
    ans_text = "Found: " + ", ".join(matches)
    return {"answer": ans_text, "citations": citations, "abstained": False}


def run_eval():
    print("Loading test split...")
    with open("eval/test_split.json", "r") as f:
        test_cases = json.load(f)
        
    print(f"Loading MIMIC tables for baseline...")
    data_dir = os.getenv("MIMIC_DATA_DIR", "data/mimic-iv-demo")
    tables, _, _ = run_ingestion(data_dir)
    
    chrono_metrics = []
    base_metrics = []
    latencies = []
    
    failed_cases = []
    
    print(f"Running evaluation on {len(test_cases)} cases...")
    for i, case in enumerate(test_cases):
        print(f"  [{i+1}/{len(test_cases)}] Q: {case['question']}")
        
        # 1. Baseline
        base_res = baseline_predict(case, tables)
        b_mets = evaluate_response(case, base_res)
        base_metrics.append(b_mets)
        
        # 2. ChronoMed API
        payload = {"query": case["question"], "subject_id": case.get("subject_id")}
        start_time = time.time()
        try:
            resp = requests.post(f"{API_URL}/ask", json=payload)
            resp.raise_for_status()
            chrono_res = resp.json()
        except Exception as e:
            print(f"API Error: {e}")
            chrono_res = {"answer": "Error", "citations": [], "abstained": False}
        latencies.append(time.time() - start_time)
        
        c_mets = evaluate_response(case, chrono_res)
        chrono_metrics.append(c_mets)
        
        # Log failure if ChronoMed got fact accuracy < 1.0 on a should_abstain=False
        if not case["should_abstain"] and c_mets["structured_fact_accuracy"] < 1.0:
            failed_cases.append({
                "case": case,
                "response": chrono_res,
                "metrics": c_mets
            })
            
    # Aggregate
    def agg(metrics_list):
        agg_res = {}
        keys = metrics_list[0].keys()
        for k in keys:
            vals = [m[k] for m in metrics_list if m[k] is not None]
            agg_res[k] = np.mean(vals) if vals else 0.0
        return agg_res
        
    base_agg = agg(base_metrics)
    chrono_agg = agg(chrono_metrics)
    avg_latency = np.mean(latencies)
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"{'Metric':<30} | {'Baseline':<10} | {'ChronoMed':<10}")
    print("-" * 55)
    for k in base_agg.keys():
        print(f"{k:<30} | {base_agg[k]:.4f}     | {chrono_agg[k]:.4f}")
    print("-" * 55)
    print(f"{'Average Latency (s)':<30} | {'N/A':<10} | {avg_latency:.4f}")
    
    # Save failed cases for review
    with open("eval/failed_cases.json", "w") as f:
        json.dump(failed_cases, f, indent=2)

if __name__ == "__main__":
    run_eval()
