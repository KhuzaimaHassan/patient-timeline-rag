# ChronoMed: Structured Patient Timeline & Evidence Retrieval

ChronoMed is a grounded RAG QA pipeline built for querying patient records over the MIMIC-IV Demo v2.2 dataset. It synthesizes fragmented clinical data (admissions, diagnoses, labs, and medications) into unified chronological timelines and strictly grounds all LLM answers to verifiable citations, explicitly self-abstaining when unsupported.

## Architecture
1. **Data Ingestion**: Parses raw MIMIC-IV CSVs and standardizes events.
2. **Timeline Builder**: Constructs chronological, patient-specific timelines.
3. **Retrieval**: Uses dense vector embeddings (FAISS) and sparse keyword retrieval (BM25) to find relevant chunks, alongside specialized temporal lookup logic.
4. **LangGraph QA**: Orchestrates retrieval, intent-aware routing, LLM generation (via Groq/Llama-3), and citation formatting.
5. **Backend/Frontend**: FastAPI backend serving a Streamlit frontend for the clinical user interface.

## Setup Instructions

### 1. Prerequisites & Credentialing
Before starting, ensure you have a PhysioNet account and have signed the Data Use Agreement for the **MIMIC-IV Clinical Database Demo**. 

### 2. Clone and Install
```bash
git clone https://github.com/KhuzaimaHassan/patient-timeline-rag.git
cd patient-timeline-rag
pip install -r requirements.txt
```

### 3. Environment Configuration
Copy the `.env.example` file to `.env` and configure your API keys:
```bash
cp .env.example .env
```
Ensure `GROQ_API_KEY` is set to a valid key from your Groq console.

### 4. Download Data
Download the MIMIC-IV demo dataset into the `data/mimic-iv-demo` directory:
```bash
mkdir -p data/mimic-iv-demo
cd data/mimic-iv-demo
wget -r -np -nH --cut-dirs=3 -R "index.html*" https://physionet.org/files/mimic-iv-demo/2.2/
cd ../..
```

### 5. Build the Index
Build the FAISS and BM25 search indices (takes several minutes):
```bash
python retrieval/build_index.py
```

### 6. Run the Application
Start the FastAPI backend server:
```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```
In a new terminal, launch the Streamlit frontend:
```bash
streamlit run app/frontend.py
```

## How to Evaluate
You can automatically evaluate ChronoMed against the non-AI baseline using our evaluation harness.
```bash
python eval/create_test_set.py
python eval/split.py
python eval/run_eval.py
```
This generates aggregate metrics (Fact Accuracy, Temporal Order Accuracy, Source Provenance, Abstention Accuracy, Latency) which are compared directly to a baseline heuristic model. Review `docs/evaluation_report.md` for historical results.
