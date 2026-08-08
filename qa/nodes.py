"""
qa/nodes.py — LangGraph Nodes for RAG Pipeline
===============================================
Implements the core logic nodes for the QA graph:
retrieval, confidence scoring, LLM generation, and citation formatting.
"""

import os
import sys
from typing import TypedDict, List, Dict, Any, Optional
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.faiss_retriever import FAISSRetriever
from retrieval.bm25_retriever import BM25Retriever
from qa.prompts import SYSTEM_PROMPT, SAFETY_NOTICE

# ── State Definition ────────────────────────────────────────────────────────
class GraphState(TypedDict):
    query: str
    subject_id: Optional[int]
    retrieved_chunks: List[Any]  # list of RetrievalChunk objects
    confidence: float
    abstained: bool
    answer: str
    citations: List[Dict[str, Any]]

# ── Global instances (loaded lazily) ─────────────────────────────────────────
faiss_r = None
bm25_r = None
llm = None

def get_faiss():
    global faiss_r
    if faiss_r is None:
        faiss_r = FAISSRetriever()
        faiss_r.load()
    return faiss_r

def get_bm25():
    global bm25_r
    if bm25_r is None:
        bm25_r = BM25Retriever()
        bm25_r.load()
    return bm25_r

def get_llm():
    global llm
    if llm is None:
        model_name = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        llm = ChatGroq(temperature=0, model=model_name, groq_api_key=api_key)
    return llm


# ── Nodes ────────────────────────────────────────────────────────────────────

def retrieve(state: GraphState) -> Dict:
    """Retrieve chunks using both FAISS and BM25, then merge/dedupe."""
    query = state["query"]
    subject_id = state.get("subject_id")
    top_k = int(os.getenv("TOP_K_RETRIEVAL", "5"))

    f_retriever = get_faiss()
    b_retriever = get_bm25()

    faiss_results = f_retriever.search(query, top_k=top_k, subject_id=subject_id)
    bm25_results = b_retriever.search(query, top_k=top_k, subject_id=subject_id)

    # Merge and dedupe by chunk_id, preserving max score info for confidence
    merged_chunks = {}
    
    for chunk, score in faiss_results:
        merged_chunks[chunk.chunk_id] = {"chunk": chunk, "faiss_score": score, "bm25_score": 0.0}
        
    for chunk, score in bm25_results:
        if chunk.chunk_id in merged_chunks:
            merged_chunks[chunk.chunk_id]["bm25_score"] = score
        else:
            merged_chunks[chunk.chunk_id] = {"chunk": chunk, "faiss_score": 0.0, "bm25_score": score}

    # Sort by a combined heuristic (FAISS > 0.5 or BM25 > 5.0 are strong hits)
    def sort_key(item):
        return item["faiss_score"] + (item["bm25_score"] / 10.0)
    
    sorted_items = sorted(merged_chunks.values(), key=sort_key, reverse=True)
    
    # Attach the scores to the chunks for the next node
    final_chunks = []
    for item in sorted_items[:top_k * 2]:  # Keep union top K
        chunk = item["chunk"]
        chunk.metadata["_faiss_score"] = item["faiss_score"]
        chunk.metadata["_bm25_score"] = item["bm25_score"]
        final_chunks.append(chunk)

    return {"retrieved_chunks": final_chunks}


def score_confidence(state: GraphState) -> Dict:
    """Compute confidence from FAISS scores and BM25 presence."""
    chunks = state["retrieved_chunks"]
    
    if not chunks:
        return {"confidence": 0.0}

    max_faiss = max([c.metadata.get("_faiss_score", 0.0) for c in chunks], default=0.0)
    max_bm25 = max([c.metadata.get("_bm25_score", 0.0) for c in chunks], default=0.0)
    
    # Confidence is primarily the FAISS similarity score
    confidence = max_faiss
    
    # Boost confidence slightly if there's a strong exact keyword match
    if max_bm25 > 5.0 and confidence < 0.8:
        confidence += 0.1
        
    return {"confidence": confidence}


def generate_answer(state: GraphState) -> Dict:
    """Call LLM with chunks as context to generate grounded answer."""
    chunks = state["retrieved_chunks"]
    query = state["query"]
    
    # Format context
    context_parts = []
    for c in chunks:
        cite_str = c.metadata.get('citation', c.chunk_id)
        context_parts.append(f"Content: {c.text}\nCitation: {cite_str}")
    
    context_str = "\n\n".join(context_parts)
    
    system_msg = SYSTEM_PROMPT.format(context=context_str)
    
    chat = get_llm()
    messages = [
        SystemMessage(content=system_msg),
        HumanMessage(content=query)
    ]
    
    response = chat.invoke(messages)
    
    return {"answer": response.content, "abstained": False}


def abstain(state: GraphState) -> Dict:
    """Fixed abstain response when confidence is too low."""
    ans = f"I cannot find evidence for that in the structured record.\n\n{SAFETY_NOTICE}"
    return {"answer": ans, "abstained": True, "citations": []}


def format_citations(state: GraphState) -> Dict:
    """Attach structured citation objects from chunks."""
    if state.get("abstained", False):
        return {"citations": []}
        
    chunks = state.get("retrieved_chunks", [])
    citations = []
    for c in chunks:
        citations.append({
            "source_table": c.metadata.get("source_table"),
            "source_row_id": c.metadata.get("source_row_id"),
            "subject_id": c.metadata.get("subject_id"),
            "timestamp": c.metadata.get("timestamp"),
            "citation": c.metadata.get("citation")
        })
        
    return {"citations": citations}
