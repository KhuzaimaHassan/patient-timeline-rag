# SGTDP Hackathon 2026 — Master Plan
## Track 1: Structured Patient Timeline & Evidence Retrieval
### "ChronoMed: Grounded Patient Journey Explorer"

> This is the internal planning document. See README.md for the public-facing project description.

Refer to the full plan in implementation_plan.md (in .gemini artifacts).
This file is kept here for reference during the hackathon build.

## Quick Reference: Open Questions to Decide First
1. LLM: OpenAI GPT-4o-mini OR Ollama llama3.2?
2. MIMIC-IV CSV files: already downloaded? (need PhysioNet account if not)
3. Frontend: Streamlit (recommended) or HTML/JS?
4. Team size: solo or team? (affects CV upload count)
5. Deployment: Streamlit Cloud or demo video only?
6. Eval test set: reference subject_ids in test_set.json?
7. Embedding: MiniLM (fast) or MPNet (better)?

## Build Order (48h)
- T+0→T+8:   Phase 1 — Foundation (setup, data, FAISS, timeline, BM25)
- T+8→T+20:  Phase 2 — Core Pipeline (LangGraph QA, FastAPI, Streamlit)
- T+20→T+30: Phase 3 — Evaluation (test set, 4 required metrics, baseline)
- T+30→T+42: Phase 4 — Docs & Polish (safety statement, README, video)
- T+42→T+48: Phase 5 — Buffer & Submission
