# ChronoMed Evaluation Report

## Evaluation Metrics

============================================================
EVALUATION RESULTS
============================================================
Metric                         | Baseline   | ChronoMed 
-------------------------------------------------------
structured_fact_accuracy       | 0.3333     | 0.8889
temporal_order_accuracy        | 0.0000     | 1.0000
source_provenance_coverage     | 0.6667     | 0.7778
abstention_accuracy            | 0.8889     | 0.8889
-------------------------------------------------------
Average Latency (s)            | N/A        | 8.6724

## Performance Analysis
ChronoMed vastly outperforms the baseline, particularly in structured fact accuracy (increasing from 0.3333 to 0.8889) and temporal order accuracy (jumping from 0.0000 to perfect 1.0000). This improvement is due to ChronoMed's grounded evidence retrieval using dense vector embeddings (FAISS) coupled with BM25, and specifically the custom temporal-aware lookup logic that queries explicit chronological endpoints directly from the timeline builder, rather than relying on the baseline's fragile keyword matching.

## Known Limitations & Future Work
While ChronoMed avoids hallucinations perfectly (matching the baseline's high 0.8889 abstention rate), it suffers from one known limitation: a false-negative abstention on a Furosemide prescription query. This occurs due to over-cautious abstention logic when the retriever fails to surface the exact drug formulation chunk within the `TOP_K` window. Given more time, I would improve retrieval recall tuning (e.g., query expansion or hierarchical index searching) specifically for single-drug-name queries to ensure sparse formulations are consistently retrieved before routing to the LLM.
