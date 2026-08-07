"""
retrieval/bm25_retriever.py — BM25 Keyword Retriever
=====================================================
Builds a BM25 index over the same RetrievalChunks as the FAISS retriever.
BM25 is essential for exact-value queries like "creatinine 1.2" or "metoprolol"
where dense retrieval can miss exact string matches.

Complements FAISS — use both and merge results for best coverage.
"""

import os
import sys
import pickle
import math
from typing import Optional
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.chunker import RetrievalChunk

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError("Missing dependency: rank_bm25. Run: pip install rank-bm25")


INDEX_DIR = os.getenv("FAISS_INDEX_DIR", "data/faiss_index")


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    return text.lower().split()


class BM25Retriever:
    """
    BM25 keyword retriever over RetrievalChunks.

    Usage:
        retriever = BM25Retriever()
        retriever.build(chunks)           # or retriever.load()
        results = retriever.search("metoprolol dose", top_k=5)
        for chunk, score in results:
            print(score, chunk.metadata["citation"])
    """

    def __init__(self, index_dir: str = INDEX_DIR):
        self.index_dir = Path(index_dir)
        self._bm25: Optional[BM25Okapi] = None
        self._chunks: list[RetrievalChunk] = []
        self._tokenized_corpus: list[list[str]] = []

    def build(self, chunks: list[RetrievalChunk]) -> None:
        """Build BM25 index from chunks. Saves to disk."""
        if not chunks:
            raise ValueError("No chunks to index.")

        print(f"\nBuilding BM25 index over {len(chunks):,} chunks ...")
        self._chunks = chunks
        self._tokenized_corpus = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        print(f"  BM25 index built: {len(self._chunks):,} documents")
        self._save()

    def _save(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        with open(self.index_dir / "bm25.pkl", "wb") as f:
            pickle.dump({
                "bm25": self._bm25,
                "chunks": self._chunks,
                "tokenized_corpus": self._tokenized_corpus,
            }, f)
        print(f"  Saved BM25 index to: {self.index_dir}")

    def load(self) -> None:
        """Load previously built BM25 index from disk."""
        bm25_path = self.index_dir / "bm25.pkl"
        if not bm25_path.exists():
            raise FileNotFoundError(
                f"No BM25 index found at {self.index_dir}. Run build_index.py first."
            )
        with open(bm25_path, "rb") as f:
            data = pickle.load(f)
        self._bm25 = data["bm25"]
        self._chunks = data["chunks"]
        self._tokenized_corpus = data["tokenized_corpus"]
        print(f"Loaded BM25 index: {len(self._chunks):,} documents")

    def search(
        self,
        query: str,
        top_k: int = 5,
        subject_id: Optional[int] = None,
        event_type: Optional[str] = None,
    ) -> list[tuple[RetrievalChunk, float]]:
        """
        BM25 keyword search.

        Args:
            query      : search string (tokenized internally)
            top_k      : number of results to return
            subject_id : optional patient filter
            event_type : optional event type filter

        Returns:
            List of (RetrievalChunk, score) sorted by BM25 score descending.
            Scores are raw BM25 values (higher = more relevant).
        """
        if self._bm25 is None:
            raise RuntimeError("BM25 index not built or loaded.")

        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)

        # Get sorted indices
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in ranked:
            if score <= 0.0:
                break  # BM25 = 0 means no token overlap — skip
            chunk = self._chunks[idx]
            if subject_id is not None and chunk.metadata.get("subject_id") != subject_id:
                continue
            if event_type is not None and chunk.metadata.get("event_type") != event_type:
                continue
            results.append((chunk, float(score)))
            if len(results) >= top_k:
                break

        return results

    def is_loaded(self) -> bool:
        return self._bm25 is not None
