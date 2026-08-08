"""
app/main.py — FastAPI Backend
==============================
Exposes endpoints for patient lists, full timelines, and RAG Q&A.
"""

import os
import sys
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

from data.ingest import run_ingestion
from timeline.builder import build_timeline, get_all_subject_ids
from qa.graph import answer_question

app = FastAPI(title="ChronoMed API - Hackathon Track 1")

# Global state to hold loaded tables
MIMIC_TABLES = {}
SUBJECT_IDS = []

@app.on_event("startup")
async def startup_event():
    """Load tables on startup so timeline generation is fast."""
    global MIMIC_TABLES, SUBJECT_IDS
    data_dir = os.getenv("MIMIC_DATA_DIR", "data/mimic-iv-demo")
    print(f"Starting up: Loading MIMIC tables from {data_dir}...")
    tables, _, _ = run_ingestion(data_dir)
    MIMIC_TABLES = tables
    SUBJECT_IDS = get_all_subject_ids(tables)
    print(f"Startup complete. Loaded {len(SUBJECT_IDS)} patients.")


class AskRequest(BaseModel):
    query: str
    subject_id: Optional[int] = None

class AskResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]
    abstained: bool
    confidence: float


@app.get("/patients", response_model=List[int])
def get_patients():
    """Return a list of all patient subject_ids."""
    return SUBJECT_IDS


@app.get("/patients/{subject_id}/timeline")
def get_patient_timeline(subject_id: int):
    """Return the complete chronologically-ordered timeline for a patient."""
    if subject_id not in SUBJECT_IDS:
        raise HTTPException(status_code=404, detail="Patient not found")
        
    events = build_timeline(MIMIC_TABLES, subject_id=subject_id)
    return {"subject_id": subject_id, "events": [e.to_dict() for e in events]}


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    """Run the LangGraph RAG pipeline to answer a clinical question."""
    if request.subject_id and request.subject_id not in SUBJECT_IDS:
        raise HTTPException(status_code=404, detail="Patient not found")
        
    result = answer_question(request.query, subject_id=request.subject_id)
    return AskResponse(**result)

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
