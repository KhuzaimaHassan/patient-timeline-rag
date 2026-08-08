import os
import json
import random
from typing import List, Tuple

def create_patient_split(test_cases: List[dict], test_ratio: float = 0.5, seed: int = 42) -> Tuple[List[dict], List[dict]]:
    """
    Groups test cases by subject_id and performs a patient-level train/test split.
    This guarantees no data leakage (no subject_id appears in both folds).
    """
    # Group cases by patient
    patient_cases = {}
    for case in test_cases:
        sid = case["subject_id"]
        if sid not in patient_cases:
            patient_cases[sid] = []
        patient_cases[sid].append(case)
        
    patients = list(patient_cases.keys())
    random.seed(seed)
    random.shuffle(patients)
    
    num_test = int(len(patients) * test_ratio)
    test_patients = set(patients[:num_test])
    
    train_fold = []
    test_fold = []
    
    for sid, cases in patient_cases.items():
        if sid in test_patients:
            test_fold.extend(cases)
        else:
            train_fold.extend(cases)
            
    return train_fold, test_fold

if __name__ == "__main__":
    if not os.path.exists("eval/test_set.json"):
        print("eval/test_set.json not found. Run create_test_set.py first.")
        exit(1)
        
    with open("eval/test_set.json", "r") as f:
        cases = json.load(f)
        
    train, test = create_patient_split(cases)
    
    with open("eval/train_split.json", "w") as f:
        json.dump(train, f, indent=2)
        
    with open("eval/test_split.json", "w") as f:
        json.dump(test, f, indent=2)
        
    print(f"Created patient-grouped splits: {len(train)} train cases, {len(test)} test cases.")
