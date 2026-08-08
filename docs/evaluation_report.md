# ChronoMed Phase 3 Evaluation Report

## Overview
This report presents the evaluation results of the ChronoMed LangGraph RAG pipeline compared against a non-AI keyword search baseline. The evaluation was run on a generated test set of 9 patient-specific questions (after splitting to avoid data leakage), covering factual retrieval, temporal ordering, and out-of-scope (abstention) cases.

## Metrics Table

| Metric                         | Baseline   | ChronoMed  |
|--------------------------------|------------|------------|
| structured_fact_accuracy       | 0.5556     | 0.3333     |
| temporal_order_accuracy        | 0.0000     | 0.0000     |
| source_provenance_coverage     | 0.3333     | 0.3333     |
| abstention_accuracy            | 0.8889     | 0.1111     |
| **Average Latency (s)**        | N/A        | 0.1102     |

## Analysis
ChronoMed significantly underperformed the simple keyword baseline in this evaluation fold. The root cause is entirely due to the **overly strict abstention logic** introduced to fix the "brain MRI" false positive. 

By changing the router to require `confidence >= 0.3 AND max_bm25 >= 5.0`, the system became highly conservative. For almost every standard question (e.g., "Was patient X prescribed medication Y?"), the BM25 score fell short of 5.0 because the queries were short. As a result, LangGraph short-circuited directly to the `abstain` node instead of calling the LLM. 

This is why ChronoMed achieved a near 0% `abstention_accuracy` for answerable questions (it abstained when it shouldn't have) and scored poorly on factual accuracy. The baseline, doing a simple substring match, naturally found the facts and scored higher.

### Recommendation for Phase 4
We must recalibrate `decide_abstain`. Instead of requiring both metrics to clear high absolute thresholds, we should rely more heavily on the LLM's own self-abstention capabilities (which we proved works perfectly in Phase 2) or use a normalized BM25 score that doesn't penalize short queries.
