import json, requests
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval.metrics import evaluate_response

with open('eval/test_split.json') as f:
    cases = json.load(f)

temporal_cases = [c for c in cases if 'first' in c['question'].lower() and 'recent' in c['question'].lower()]

outputs = []
for c in temporal_cases:
    resp = requests.post('http://127.0.0.1:8000/ask', json={'query': c['question'], 'subject_id': c['subject_id']}).json()
    mets = evaluate_response(c, resp)
    outputs.append({'case': c, 'response': resp, 'metrics': mets})

print(json.dumps(outputs, indent=2))
