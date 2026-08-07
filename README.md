# ChronoMed: Grounded Patient Journey Explorer

> **⚠️ Research and educational prototype only. Not for clinical use. Do not use for diagnosis, treatment, triage, or emergency decisions.**

A transparent AI prototype for exploring structured hospital data from MIMIC-IV Demo v2.2, submitted to the SGTDP Hackathon 2026 — Track 1: Structured Patient Timeline & Evidence Retrieval.

---

## What We Built
<!-- TODO: Fill in after implementation -->

## The Problem It Addresses
<!-- TODO: Fill in after implementation -->

## How It Works
<!-- TODO: Fill in after implementation -->

## What We'd Improve With More Time
<!-- TODO: Fill in after implementation -->

---

## Setup & Installation

### Prerequisites
- Python 3.11+
- MIMIC-IV Clinical Database Demo v2.2 ([PhysioNet](https://physionet.org/content/mimic-iv-demo/2.2/))
- OpenAI API key (or Ollama for local LLM)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/patient-timeline-rag.git
cd patient-timeline-rag

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your API key

# 5. Place MIMIC-IV Demo CSVs
# Copy hosp/ and icu/ directories to data/mimic-iv-demo/

# 6. Ingest data and build index
python data/ingest.py
python retrieval/build_index.py
```

### Running the Application

```bash
# Start FastAPI backend
uvicorn app.main:app --reload --port 8000

# In a new terminal, start Streamlit frontend
streamlit run app/frontend.py
```

### Running Evaluation

```bash
python eval/run_eval.py
```

---

## Dataset Citation

Johnson AEW, et al. MIMIC-IV Clinical Database Demo (version 2.2). PhysioNet. 2023.
DOI: https://doi.org/10.13026/dp1f-ex47
Licence: PhysioNet Credentialed Health Data Licence 1.5.0
