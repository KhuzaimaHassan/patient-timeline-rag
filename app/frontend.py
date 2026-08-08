"""
app/frontend.py — Streamlit Frontend
=====================================
A functional MVP frontend for the Hackathon Track 1 submission.
Displays the patient timeline and a Q&A interface grounded in RAG.
"""

import os
import requests
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_URL = f"http://{os.getenv('API_HOST', '127.0.0.1')}:{os.getenv('API_PORT', '8000')}"

st.set_page_config(page_title="ChronoMed MVP", layout="wide")

st.title("ChronoMed: Patient Timeline & Evidence Retrieval")
st.markdown("**Track 1: AI for Smarter Patient Care (Sofstica Hackathon)**")

# --- Layout: Sidebar for patient selection, Main for timeline, Right column for QA ---
st.sidebar.header("Patient Selection")

# 1. Fetch Patients
@st.cache_data(ttl=300)
def fetch_patients():
    try:
        resp = requests.get(f"{API_URL}/patients")
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        st.sidebar.error(f"Failed to connect to backend: {e}")
    return []

patients = fetch_patients()
if not patients:
    st.warning("No patients loaded or backend is unreachable.")
    st.stop()

selected_patient = st.sidebar.selectbox("Select Patient (subject_id)", patients)

# 2. Fetch Timeline
@st.cache_data(ttl=60)
def fetch_timeline(subject_id):
    resp = requests.get(f"{API_URL}/patients/{subject_id}/timeline")
    if resp.status_code == 200:
        return resp.json()["events"]
    return []

events = fetch_timeline(selected_patient)

# Layout columns
col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader(f"Timeline for Patient {selected_patient}")
    if events:
        # Simple tabular representation for the timeline
        df = pd.DataFrame(events)
        # Select key columns for display
        display_cols = ["timestamp", "event_type", "description"]
        if "value" in df.columns:
            display_cols.append("value")
        if "unit" in df.columns:
            display_cols.append("unit")
            
        st.dataframe(df[display_cols], use_container_width=True, height=600)
    else:
        st.info("No events found for this patient.")

with col2:
    st.subheader("Grounded Q&A")
    st.write(f"Ask questions about Patient {selected_patient}'s record.")
    
    query = st.text_input("Enter your clinical question:")
    
    if st.button("Ask"):
        if not query:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching and analyzing timeline..."):
                try:
                    payload = {"query": query, "subject_id": selected_patient}
                    resp = requests.post(f"{API_URL}/ask", json=payload)
                    
                    if resp.status_code == 200:
                        data = resp.json()
                        
                        # Visible abstention state
                        if data.get("abstained"):
                            st.error("⚠️ NO ANSWER FOUND")
                            st.write(data["answer"])
                        else:
                            st.success("Answer Generated")
                            st.write(data["answer"])
                            
                        # Confidence indicator
                        conf = data.get("confidence", 0.0)
                        st.progress(min(conf, 1.0), text=f"Retrieval Confidence: {conf:.2f}")
                        
                        # Citations
                        citations = data.get("citations", [])
                        if citations:
                            with st.expander(f"View Citations ({len(citations)})"):
                                for i, cite in enumerate(citations):
                                    st.markdown(f"**[{i+1}] {cite.get('source_table')}**")
                                    st.markdown(f"- **Time:** {cite.get('timestamp')}")
                                    st.markdown(f"- **Citation Key:** `{cite.get('citation')}`")
                                    st.divider()
                    else:
                        st.error(f"Error: {resp.text}")
                except Exception as e:
                    st.error(f"Failed to query backend: {e}")
