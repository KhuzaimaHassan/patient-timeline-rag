"""
retrieval/faiss_retriever.py — FAISS Dense Vector Retriever
============================================================
Embeds all RetrievalChunks using sentence-transformers/all-MiniLM-L6-v2,
builds a FAISS flat inner-product index, and exposes search().

The retriever returns chunks with their FULL metadata intact — nothing is
stripped or summarized. The QA layer uses metadata for exact citations.
"""

import os
import sys
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.chunker import RetrievalChunk

try:
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError(
        f"Missing dependency: {e}. Run: pip install faiss-cpu sentence-transformers"
    ) from e


# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
INDEX_DIR  = os.getenv("FAISS_INDEX_DIR", "data/faiss_index")
TOP_K_DEFAULT = int(os.getenv("TOP_K_RETRIEVAL", "5"))


class FAISSRetriever:
    """
    Dense vector retriever backed by FAISS.

    Usage:
        retriever = FAISSRetriever()
        retriever.build(chunks)           # or retriever.load()
        results = retriever.search("What was patient 10000032's creatinine?", top_k=5)
        for chunk, score in results:
            print(score, chunk.metadata["citation"])
    """

    def __init__(self, model_name: str = MODEL_NAME, index_dir: str = INDEX_DIR):
        self.model_name = model_name
        self.index_dir  = Path(index_dir)
        self._model: Optional[SentenceTransformer] = None
        self._index: Optional[faiss.Index] = None
        self._chunks: list[RetrievalChunk] = []

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            print(f"Loading embedding model: {self.model_name} ...")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _embed(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        model = self._get_model()
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,   # cosine sim via inner product
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def build(self, chunks: list[RetrievalChunk], batch_size: int = 64) -> None:
        """Embed all chunks and build the FAISS index. Saves to disk."""
        if not chunks:
            raise ValueError("No chunks to index.")

        print(f"\nBuilding FAISS index over {len(chunks):,} chunks ...")
        self._chunks = chunks
        texts = [c.text for c in chunks]

        embeddings = self._embed(texts, batch_size)
        dim = embeddings.shape[1]
        print(f"  Embedding dim: {dim}, total vectors: {len(embeddings):,}")

        # Use IndexFlatIP (inner product on normalized vectors = cosine sim)
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)
        print(f"  FAISS index built: {self._index.ntotal:,} vectors")

        self._save()

    def _save(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_dir / "faiss.index"))
        with open(self.index_dir / "chunks.pkl", "wb") as f:
            pickle.dump(self._chunks, f)
        with open(self.index_dir / "model_name.txt", "w") as f:
            f.write(self.model_name)
        print(f"  Saved index + chunks to: {self.index_dir}")

    def load(self) -> None:
        """Load a previously built index from disk."""
        index_path  = self.index_dir / "faiss.index"
        chunks_path = self.index_dir / "chunks.pkl"
        if not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                f"No FAISS index found at {self.index_dir}. Run build_index.py first."
            )
        self._index = faiss.read_index(str(index_path))
        with open(chunks_path, "rb") as f:
            self._chunks = pickle.load(f)
        # Read the model name used during build
        model_file = self.index_dir / "model_name.txt"
        if model_file.exists():
            self.model_name = model_file.read_text().strip()
        print(f"Loaded FAISS index: {self._index.ntotal:,} vectors | "
              f"model: {self.model_name} | chunks: {len(self._chunks):,}")

    def search(
        self,
        query: str,
        top_k: int = TOP_K_DEFAULT,
        subject_id: Optional[int] = None,
        event_type: Optional[str] = None,
    ) -> list[tuple[RetrievalChunk, float]]:
        """
        Search the FAISS index.

        Args:
            query      : natural language query string
            top_k      : number of results to return
            subject_id : optional filter — only return chunks for this patient
            event_type : optional filter — only return chunks of this event type

        Returns:
            List of (RetrievalChunk, score) sorted by score descending.
            score is cosine similarity (0.0 – 1.0).
        """
        if self._index is None:
            raise RuntimeError("Index not built or loaded. Call build() or load() first.")

        query_emb = self._embed([query])  # shape (1, dim)

        # Retrieve more if filtering — we'll post-filter
        fetch_k = top_k * 10 if (subject_id or event_type) else top_k * 2
        fetch_k = min(fetch_k, self._index.ntotal)

        scores, indices = self._index.search(query_emb, fetch_k)
        scores = scores[0].tolist()   # float32 → python float
        indices = indices[0].tolist()

        results = []
        for idx, score in zip(indices, scores):
            if idx < 0 or idx >= len(self._chunks):
                continue
            chunk = self._chunks[idx]
            # Post-filter
            if subject_id is not None and chunk.metadata.get("subject_id") != subject_id:
                continue
            if event_type is not None and chunk.metadata.get("event_type") != event_type:
                continue
            results.append((chunk, float(score)))
            if len(results) >= top_k:
                break

        return results

    def is_loaded(self) -> bool:
        return self._index is not None
