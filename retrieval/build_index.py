"""
retrieval/build_index.py — Build FAISS + BM25 Indices
======================================================
Orchestrates the full pipeline:
  1. Ingest all MIMIC-IV Demo tables
  2. Build timelines for all patients
  3. Convert events to chunks
  4. Build and save FAISS + BM25 indices

Usage:
    python retrieval/build_index.py [--data-dir data/mimic-iv-demo] [--index-dir data/faiss_index]
"""

import sys
import os
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(data_dir: str, index_dir: str):
    t0 = time.time()

    print("=" * 65)
    print("ChronoMed — Phase 1 Index Builder")
    print("=" * 65)

    # Step 1: Ingest
    print("\n[Step 1/4] Loading MIMIC-IV Demo tables ...")
    from data.ingest import run_ingestion
    tables, missing, schema_issues = run_ingestion(data_dir)
    if missing:
        print(f"⚠️  {len(missing)} tables missing — these event types will be skipped")

    # Step 2: Build chunks for all patients
    print("\n[Step 2/4] Building patient timelines and chunks ...")
    from retrieval.chunker import build_all_chunks
    chunks = build_all_chunks(tables)

    if not chunks:
        print("ERROR: No chunks generated. Check data directory.")
        sys.exit(1)

    # Step 3: Build FAISS index
    print(f"\n[Step 3/4] Building FAISS index ({len(chunks):,} chunks) ...")
    from retrieval.faiss_retriever import FAISSRetriever
    faiss_r = FAISSRetriever(index_dir=index_dir)
    faiss_r.build(chunks)

    # Step 4: Build BM25 index
    print(f"\n[Step 4/4] Building BM25 index ...")
    from retrieval.bm25_retriever import BM25Retriever
    bm25_r = BM25Retriever(index_dir=index_dir)
    bm25_r.build(chunks)

    elapsed = time.time() - t0
    print(f"\n{'='*65}")
    print(f"✅ Index build complete in {elapsed:.1f}s")
    print(f"   Chunks indexed: {len(chunks):,}")
    print(f"   Index saved to: {index_dir}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build ChronoMed FAISS + BM25 indices")
    parser.add_argument("--data-dir",  default="data/mimic-iv-demo")
    parser.add_argument("--index-dir", default="data/faiss_index")
    args = parser.parse_args()
    main(args.data_dir, args.index_dir)
