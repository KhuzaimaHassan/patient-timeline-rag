def calculate_structured_fact_accuracy(expected_facts, answer_text):
    if not expected_facts:
        return 1.0  # If no facts expected, it's correct (e.g. abstention cases)
    
    hits = 0
    answer_lower = answer_text.lower()
    for fact in expected_facts:
        if fact.lower() in answer_lower:
            hits += 1
            
    return hits / len(expected_facts)


def calculate_temporal_order_accuracy(expected_facts, answer_text):
    if len(expected_facts) < 2:
        return None  # Not a temporal ordering question
        
    # Check if fact 1 appears before fact 2 in the answer text
    idx1 = answer_text.lower().find(expected_facts[0].lower())
    idx2 = answer_text.lower().find(expected_facts[1].lower())
    
    if idx1 == -1 or idx2 == -1:
        return 0.0
        
    if idx1 < idx2:
        return 1.0
    return 0.0


def calculate_source_provenance_coverage(expected_citations, returned_citations):
    if not expected_citations:
        return 1.0
        
    returned_cite_keys = [c.get("citation", "") for c in returned_citations]
    
    hits = 0
    for ec in expected_citations:
        if ec in returned_cite_keys:
            hits += 1
            
    return hits / len(expected_citations)


def calculate_abstention_accuracy(should_abstain, actual_abstained):
    return 1.0 if should_abstain == actual_abstained else 0.0


def evaluate_response(case, response_dict):
    metrics = {}
    
    ans_text = response_dict.get("answer", "")
    citations = response_dict.get("citations", [])
    abstained = response_dict.get("abstained", False)
    
    exp_facts = case.get("expected_answer_facts", [])
    exp_cites = case.get("expected_citations", [])
    should_abstain = case.get("should_abstain", False)
    
    metrics["structured_fact_accuracy"] = calculate_structured_fact_accuracy(exp_facts, ans_text)
    
    # Only calculate temporal accuracy for questions that imply ordering (i.e., we put 2 dates in exp_facts)
    if "first" in case["question"].lower() and "recent" in case["question"].lower():
        metrics["temporal_order_accuracy"] = calculate_temporal_order_accuracy(exp_facts, ans_text)
    else:
        metrics["temporal_order_accuracy"] = None
        
    metrics["source_provenance_coverage"] = calculate_source_provenance_coverage(exp_cites, citations)
    metrics["abstention_accuracy"] = calculate_abstention_accuracy(should_abstain, abstained)
    
    return metrics
