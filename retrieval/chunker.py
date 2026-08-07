"""
retrieval/chunker.py — Timeline Event → Text Chunk Converter
=============================================================
Converts each TimelineEvent into a RetrievalChunk with:
  - A natural-language text string for embedding/BM25
  - Structured metadata fields (NOT just embedded in text) for citation

The text is human-readable and designed for dense embedding similarity,
while the metadata dict enables exact-citation lookups by the QA layer.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass, field, asdict
from typing import Optional
from timeline.models import TimelineEvent


@dataclass
class RetrievalChunk:
    """
    A text chunk ready for embedding and retrieval.

    text     : the natural-language string to embed / BM25-index
    metadata : structured citation fields — ALWAYS present, NEVER lost in text
    chunk_id : unique identifier for this chunk (used to look up full metadata)
    """
    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def event_to_chunk(event: TimelineEvent) -> RetrievalChunk:
    """
    Convert a TimelineEvent into a RetrievalChunk.

    Text format is: "[EventType] [Description] | Patient [subject_id] | [timestamp]"
    This makes the text useful for both semantic (dense) and keyword (BM25) retrieval.
    """
    # Build the text
    parts = [
        f"[{event.event_type.upper().replace('_', ' ')}]",
        event.description,
    ]
    if event.value and event.unit:
        # Include explicit value+unit so BM25 can find exact values like "1.2 mg/dL"
        parts.append(f"Value: {event.value} {event.unit}")
    elif event.value:
        parts.append(f"Value: {event.value}")

    if event.code and event.code_system:
        parts.append(f"Code: {event.code} ({event.code_system})")

    parts.append(f"Patient {event.subject_id}")
    parts.append(f"Time: {event.timestamp}")

    if event.flag:
        parts.append(f"Flag: {event.flag}")

    text = " | ".join(parts)

    # Build structured metadata (citation fields) — separate from text
    metadata = {
        # Citation-critical fields
        "source_table": event.source_table,
        "source_field": event.source_field,
        "source_row_id": str(event.source_row_id) if event.source_row_id is not None else None,
        "subject_id": event.subject_id,
        "timestamp": event.timestamp,
        # Clinical fields for filtering
        "event_type": event.event_type,
        "hadm_id": event.hadm_id,
        "stay_id": event.stay_id,
        "value": event.value,
        "unit": event.unit,
        "code": event.code,
        "code_system": event.code_system,
        "flag": event.flag,
        "end_timestamp": event.end_timestamp,
        "description": event.description,
        # Full citation string
        "citation": event.citation_key(),
    }
    # Add extra fields flat into metadata
    if event.extra:
        for k, v in event.extra.items():
            metadata[f"extra_{k}"] = v

    # Deterministic chunk_id from source identity
    chunk_id = (
        f"{event.source_table}|{event.source_row_id}|"
        f"{event.subject_id}|{event.timestamp}|{event.event_type}"
    )

    return RetrievalChunk(
        chunk_id=chunk_id,
        text=text,
        metadata=metadata,
    )


def events_to_chunks(events: list[TimelineEvent]) -> list[RetrievalChunk]:
    """Convert a list of TimelineEvents to RetrievalChunks."""
    chunks = []
    seen_ids = set()
    for event in events:
        chunk = event_to_chunk(event)
        # Deduplicate by chunk_id (can happen with overlapping table joins)
        if chunk.chunk_id not in seen_ids:
            chunks.append(chunk)
            seen_ids.add(chunk.chunk_id)
    return chunks


def build_all_chunks(tables: dict) -> list[RetrievalChunk]:
    """
    Build RetrievalChunks for ALL patients in the dataset.
    Used by build_index.py to create the FAISS and BM25 indices.
    """
    from timeline.builder import build_timeline, get_all_subject_ids

    subject_ids = get_all_subject_ids(tables)
    print(f"Building chunks for {len(subject_ids)} patients...")

    all_chunks = []
    for sid in subject_ids:
        events = build_timeline(tables, subject_id=sid)
        chunks = events_to_chunks(events)
        all_chunks.extend(chunks)
        print(f"  subject_id={sid}: {len(events)} events → {len(chunks)} chunks")

    print(f"\nTotal chunks: {len(all_chunks):,}")
    return all_chunks
