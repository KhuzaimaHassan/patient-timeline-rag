What I built: ChronoMed is a grounded Retrieval-Augmented Generation (RAG) QA system to interrogate structured Electronic Health Records using the MIMIC-IV Demo v2.2 dataset.

The problem: Clinicians spend excessive time parsing fragmented patient data across admissions, labs, and meds. Finding chronological events (e.g., "first vs latest admission") requires sifting through hundreds of database rows.

How it works: I ingest raw CSV tables into a unified chronological patient timeline. Events are embedded into a FAISS vector index alongside BM25. A LangGraph backend retrieves facts—including intent-aware temporal endpoints—and forces a Groq-powered LLM to answer using only provided context. The system is instructed to append citations and self-abstain when evidence is missing.

What to improve: I would optimize retrieval recall tuning for sparse single-drug-name queries to reduce over-cautious false-negative abstentions.
