import os
import sys
import json
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.ingest import run_ingestion

def create_test_set():
    print("Loading MIMIC tables to generate grounded test set...")
    data_dir = os.getenv("MIMIC_DATA_DIR", "data/mimic-iv-demo")
    tables, _, _ = run_ingestion(data_dir)
    
    test_cases = []
    
    # 1. Answerable Factual Questions (Labs, Meds, Diagnoses, Admissions)
    
    # Diagnoses
    diags = tables["hosp/diagnoses_icd"]
    d_icd = tables["hosp/d_icd_diagnoses"]
    merged_diags = diags.merge(d_icd, on=["icd_code", "icd_version"])
    
    # Pick a few random patients who have diagnoses
    sample_diags = merged_diags.sample(5, random_state=42)
    for _, row in sample_diags.iterrows():
        subj_id = int(row["subject_id"])
        diag_title = row["long_title"]
        from timeline.models import TimelineEvent
        te_diag = TimelineEvent(
            source_table="hosp/diagnoses_icd",
            source_field="icd_code",
            source_row_id=f"{subj_id}-{int(row['hadm_id'])}-{int(row['seq_num'])}",
            subject_id=subj_id,
            timestamp="unknown",  # We used 'unknown' for diagnoses lacking timestamps initially
            event_type="diagnosis",
            description=""
        )
        test_cases.append({
            "question": f"Did patient {subj_id} ever get diagnosed with {diag_title}?",
            "subject_id": subj_id,
            "expected_answer_facts": [diag_title],
            "expected_citations": [te_diag.citation_key()],
            "should_abstain": False
        })
        
    # Prescriptions
    meds = tables["hosp/prescriptions"]
    sample_meds = meds.dropna(subset=['drug']).sample(5, random_state=123)
    for _, row in sample_meds.iterrows():
        subj_id = int(row["subject_id"])
        drug = row["drug"]
        te_med = TimelineEvent(
            source_table="hosp/prescriptions",
            source_field="drug",
            source_row_id=int(row["pharmacy_id"]),
            subject_id=subj_id,
            timestamp=str(row.get("starttime")),
            event_type="medication",
            description=""
        )
        test_cases.append({
            "question": f"Was patient {subj_id} prescribed {drug}?",
            "subject_id": subj_id,
            "expected_answer_facts": [drug],
            "expected_citations": [te_med.citation_key()],
            "should_abstain": False
        })
        
    # 2. Temporal Order Questions
    # Let's find patients with multiple admissions
    adms = tables["hosp/admissions"]
    multi_adm_subjs = adms["subject_id"].value_counts()
    multi_adm_subjs = multi_adm_subjs[multi_adm_subjs > 1].index.tolist()
    
    for subj_id in multi_adm_subjs[:5]:
        subj_id = int(subj_id)
        p_adms = adms[adms["subject_id"] == subj_id].sort_values("admittime")
        first_adm = p_adms.iloc[0]
        last_adm = p_adms.iloc[-1]
        
        from timeline.models import TimelineEvent
        
        te1 = TimelineEvent(
            source_table="hosp/admissions",
            source_field="admittime",
            source_row_id=int(first_adm["hadm_id"]),
            subject_id=subj_id,
            timestamp=str(first_adm["admittime"]),
            event_type="admission",
            description=""
        )
        te2 = TimelineEvent(
            source_table="hosp/admissions",
            source_field="admittime",
            source_row_id=int(last_adm["hadm_id"]),
            subject_id=subj_id,
            timestamp=str(last_adm["admittime"]),
            event_type="admission",
            description=""
        )
        
        test_cases.append({
            "question": f"For patient {subj_id}, what was the date of their first admission compared to their most recent admission?",
            "subject_id": subj_id,
            "expected_answer_facts": [str(first_adm["admittime"])[:10], str(last_adm["admittime"])[:10]],
            "expected_citations": [te1.citation_key(), te2.citation_key()],
            "should_abstain": False
        })
        
    # 3. Deliberately Unanswerable Questions
    # Use valid subject_ids from our dataset to avoid 404s, but ask about missing data types
    valid_subjs = tables["hosp/patients"]["subject_id"].tolist()
    random.seed(999)
    unans_subjs = random.sample(valid_subjs, 5)
    
    unanswerable = [
        {
            "question": f"What did the MRI brain imaging radiology scan show for patient {unans_subjs[0]}?",
            "subject_id": unans_subjs[0],
            "expected_answer_facts": [],
            "expected_citations": [],
            "should_abstain": True
        },
        {
            "question": f"What did the clinical notes say about patient {unans_subjs[1]}'s mood?",
            "subject_id": unans_subjs[1],
            "expected_answer_facts": [],
            "expected_citations": [],
            "should_abstain": True
        },
        {
            "question": f"What was the result of the echocardiogram ultrasound for patient {unans_subjs[2]}?",
            "subject_id": unans_subjs[2],
            "expected_answer_facts": [],
            "expected_citations": [],
            "should_abstain": True
        },
        {
            "question": f"What is the social security number and home address of patient {unans_subjs[3]}?",
            "subject_id": unans_subjs[3],
            "expected_answer_facts": [],
            "expected_citations": [],
            "should_abstain": True
        },
        {
            "question": f"Did patient {unans_subjs[4]} ever travel to Mars?",
            "subject_id": unans_subjs[4],
            "expected_answer_facts": [],
            "expected_citations": [],
            "should_abstain": True
        }
    ]
    
    test_cases.extend(unanswerable)
    
    os.makedirs("eval", exist_ok=True)
    with open("eval/test_set.json", "w") as f:
        json.dump(test_cases, f, indent=2)
        
    print(f"Generated {len(test_cases)} test cases in eval/test_set.json")

if __name__ == "__main__":
    create_test_set()
