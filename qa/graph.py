"""
qa/graph.py — LangGraph StateGraph Wiring
==========================================
Wires the QA nodes into an executable directed graph.
Exposes the main answer_question() entrypoint.
"""

import os
import sys
from typing import Dict, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, END
from qa.nodes import (
    GraphState,
    retrieve,
    score_confidence,
    generate_answer,
    abstain,
    format_citations
)

def decide_abstain(state: GraphState) -> str:
    """
    Conditional routing function.
    Hard-abstain ONLY if retrieved_chunks is empty.
    Otherwise, trust the LLM's grounding prompt to abstain if needed.
    """
    chunks = state.get("retrieved_chunks", [])
    
    if len(chunks) == 0:
        return "abstain"
    return "generate_answer"


def build_graph():
    """Build and compile the LangGraph workflow."""
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("score_confidence", score_confidence)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("abstain", abstain)
    workflow.add_node("format_citations", format_citations)
    
    # Define edges
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "score_confidence")
    
    workflow.add_conditional_edges(
        "score_confidence",
        decide_abstain,
        {
            "abstain": "abstain",
            "generate_answer": "generate_answer"
        }
    )
    
    workflow.add_edge("generate_answer", "format_citations")
    workflow.add_edge("abstain", "format_citations")
    workflow.add_edge("format_citations", END)
    
    return workflow.compile()


# Compile graph once
app_graph = build_graph()


def answer_question(query: str, subject_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Main entrypoint for the RAG pipeline.
    
    Args:
        query: User question
        subject_id: Optional patient filter
        
    Returns:
        dict containing answer, citations, abstained flag, and confidence.
    """
    initial_state = {
        "query": query,
        "subject_id": subject_id,
        "retrieved_chunks": [],
        "confidence": 0.0,
        "abstained": False,
        "answer": "",
        "citations": []
    }
    
    result = app_graph.invoke(initial_state)
    
    return {
        "answer": result["answer"],
        "citations": result["citations"],
        "abstained": result["abstained"],
        "confidence": result["confidence"]
    }
