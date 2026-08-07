"""
timeline/builder.py — Patient Timeline Builder
===============================================
Reconstructs a chronologically-ordered list of TimelineEvent objects
for a given subject_id (and optionally hadm_id) by joining across
all MIMIC-IV Demo v2.2 tables.

Every event preserves:
  - source_table  : exact CSV table name
  - source_field  : the field holding the primary value
  - source_row_id : row-level identifier for citation
  - timestamp     : the event time (or best available time)

Usage:
    from data.ingest import run_ingestion
    from timeline.builder import build_timeline

    tables, _, _ = run_ingestion("data/mimic-iv-demo")
    events = build_timeline(tables, subject_id=10000032)
    for e in events:
        print(e.timestamp, e.event_type, e.description)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from typing import Optional
from timeline.models import TimelineEvent


def _safe_str(val) -> Optional[str]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return str(val)


def _safe_int(val) -> Optional[int]:
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        return int(val)
    except (ValueError, TypeError):
        return None


def _ts(val) -> Optional[str]:
    """Normalize a timestamp to ISO 8601 string. Return None if missing."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip()
    return s if s and s.lower() not in ("nat", "none", "nan", "") else None


# ── Individual table → events converters ────────────────────────────────────

def _admissions_events(tables: dict, subject_id: int, hadm_id: Optional[int]) -> list[TimelineEvent]:
    events = []
    df = tables.get("hosp/admissions")
    if df is None:
        return events
    rows = df[df["subject_id"] == subject_id]
    if hadm_id is not None:
        rows = rows[rows["hadm_id"] == hadm_id]
    for _, row in rows.iterrows():
        ts = _ts(row.get("admittime"))
        if not ts:
            continue
        events.append(TimelineEvent(
            source_table="hosp/admissions",
            source_field="admittime",
            source_row_id=_safe_int(row.get("hadm_id")),
            subject_id=subject_id,
            hadm_id=_safe_int(row.get("hadm_id")),
            timestamp=ts,
            end_timestamp=_ts(row.get("dischtime")),
            event_type="admission",
            description=(
                f"Hospital admission — type: {_safe_str(row.get('admission_type'))}, "
                f"location: {_safe_str(row.get('admission_location'))}, "
                f"insurance: {_safe_str(row.get('insurance'))}, "
                f"race: {_safe_str(row.get('race'))}"
            ),
            extra={
                "discharge_location": _safe_str(row.get("discharge_location")),
                "hospital_expire_flag": _safe_int(row.get("hospital_expire_flag")),
                "marital_status": _safe_str(row.get("marital_status")),
                "language": _safe_str(row.get("language")),
            },
        ))
        disch_ts = _ts(row.get("dischtime"))
        if disch_ts:
            events.append(TimelineEvent(
                source_table="hosp/admissions",
                source_field="dischtime",
                source_row_id=_safe_int(row.get("hadm_id")),
                subject_id=subject_id,
                hadm_id=_safe_int(row.get("hadm_id")),
                timestamp=disch_ts,
                event_type="discharge",
                description=(
                    f"Hospital discharge — to: {_safe_str(row.get('discharge_location'))}, "
                    f"expired: {_safe_int(row.get('hospital_expire_flag'))}"
                ),
            ))
    return events


def _transfers_events(tables: dict, subject_id: int, hadm_id: Optional[int]) -> list[TimelineEvent]:
    events = []
    df = tables.get("hosp/transfers")
    if df is None:
        return events
    rows = df[df["subject_id"] == subject_id]
    if hadm_id is not None:
        rows = rows[rows["hadm_id"] == hadm_id]
    for _, row in rows.iterrows():
        ts = _ts(row.get("intime"))
        if not ts:
            continue
        etype = _safe_str(row.get("eventtype")) or "transfer"
        careunit = _safe_str(row.get("careunit")) or "unknown unit"
        events.append(TimelineEvent(
            source_table="hosp/transfers",
            source_field="intime",
            source_row_id=_safe_int(row.get("transfer_id")),
            subject_id=subject_id,
            hadm_id=_safe_int(row.get("hadm_id")),
            timestamp=ts,
            end_timestamp=_ts(row.get("outtime")),
            event_type="transfer",
            description=f"Transfer — event: {etype}, careunit: {careunit}",
            code=etype,
            code_system="eventtype",
        ))
    return events


def _diagnoses_events(tables: dict, subject_id: int, hadm_id: Optional[int]) -> list[TimelineEvent]:
    events = []
    df = tables.get("hosp/diagnoses_icd")
    desc_df = tables.get("hosp/d_icd_diagnoses")
    if df is None:
        return events
    rows = df[df["subject_id"] == subject_id]
    if hadm_id is not None:
        rows = rows[rows["hadm_id"] == hadm_id]
    # Get discharge time from admissions for diagnosis timestamp
    adm_df = tables.get("hosp/admissions")
    adm_times = {}
    if adm_df is not None:
        pat_adm = adm_df[adm_df["subject_id"] == subject_id][["hadm_id", "dischtime"]]
        adm_times = dict(zip(pat_adm["hadm_id"], pat_adm["dischtime"]))
    # Build ICD description lookup
    icd_desc = {}
    if desc_df is not None:
        for _, r in desc_df.iterrows():
            key = (str(r["icd_code"]).strip(), int(r["icd_version"]))
            icd_desc[key] = str(r["long_title"])
    for _, row in rows.iterrows():
        hid = _safe_int(row.get("hadm_id"))
        ts = _ts(adm_times.get(hid)) if hid else None
        if not ts:
            ts = "unknown"
        icd_code = _safe_str(row.get("icd_code"))
        icd_ver = _safe_int(row.get("icd_version")) or 10
        title = icd_desc.get((str(icd_code), icd_ver), icd_code)
        events.append(TimelineEvent(
            source_table="hosp/diagnoses_icd",
            source_field="icd_code",
            source_row_id=f"{subject_id}-{hid}-{row.get('seq_num')}",
            subject_id=subject_id,
            hadm_id=hid,
            timestamp=ts,
            event_type="diagnosis",
            description=f"Diagnosis ICD-{icd_ver}: {icd_code} — {title} (seq {row.get('seq_num')})",
            code=icd_code,
            code_system=f"ICD-{icd_ver}",
        ))
    return events


def _procedures_events(tables: dict, subject_id: int, hadm_id: Optional[int]) -> list[TimelineEvent]:
    events = []
    df = tables.get("hosp/procedures_icd")
    desc_df = tables.get("hosp/d_icd_procedures")
    if df is None:
        return events
    rows = df[df["subject_id"] == subject_id]
    if hadm_id is not None:
        rows = rows[rows["hadm_id"] == hadm_id]
    icd_desc = {}
    if desc_df is not None:
        for _, r in desc_df.iterrows():
            key = (str(r["icd_code"]).strip(), int(r["icd_version"]))
            icd_desc[key] = str(r["long_title"])
    for _, row in rows.iterrows():
        ts = _ts(row.get("chartdate")) or "unknown"
        icd_code = _safe_str(row.get("icd_code"))
        icd_ver = _safe_int(row.get("icd_version")) or 10
        title = icd_desc.get((str(icd_code), icd_ver), icd_code)
        events.append(TimelineEvent(
            source_table="hosp/procedures_icd",
            source_field="icd_code",
            source_row_id=f"{subject_id}-{row.get('hadm_id')}-{row.get('seq_num')}",
            subject_id=subject_id,
            hadm_id=_safe_int(row.get("hadm_id")),
            timestamp=ts,
            event_type="procedure",
            description=f"Procedure ICD-{icd_ver}: {icd_code} — {title} (seq {row.get('seq_num')})",
            code=icd_code,
            code_system=f"ICD-{icd_ver}",
        ))
    return events


def _lab_events(tables: dict, subject_id: int, hadm_id: Optional[int],
                max_labs: int = 500) -> list[TimelineEvent]:
    events = []
    df = tables.get("hosp/labevents")
    items_df = tables.get("hosp/d_labitems")
    if df is None:
        return events
    rows = df[df["subject_id"] == subject_id]
    if hadm_id is not None:
        rows = rows[rows["hadm_id"] == hadm_id]
    rows = rows.sort_values("charttime").head(max_labs)
    # Build item lookup
    item_labels = {}
    if items_df is not None:
        item_labels = dict(zip(items_df["itemid"], items_df["label"]))
    for _, row in rows.iterrows():
        ts = _ts(row.get("charttime"))
        if not ts:
            continue
        itemid = _safe_int(row.get("itemid"))
        label = item_labels.get(itemid, f"itemid={itemid}")
        value = _safe_str(row.get("value"))
        valuenum = _safe_str(row.get("valuenum"))
        unit = _safe_str(row.get("valueuom"))
        flag = _safe_str(row.get("flag"))
        display_val = valuenum or value or "no value"
        events.append(TimelineEvent(
            source_table="hosp/labevents",
            source_field="valuenum",
            source_row_id=_safe_int(row.get("labevent_id")),
            subject_id=subject_id,
            hadm_id=_safe_int(row.get("hadm_id")),
            timestamp=ts,
            event_type="lab",
            description=(
                f"Lab result: {label} = {display_val} {unit or ''}"
                + (f" [{flag}]" if flag else "")
            ),
            value=display_val,
            unit=unit,
            code=str(itemid) if itemid else None,
            code_system="MIMIC_itemid",
            flag=flag,
            extra={
                "ref_range_lower": _safe_str(row.get("ref_range_lower")),
                "ref_range_upper": _safe_str(row.get("ref_range_upper")),
                "specimen_id": _safe_int(row.get("specimen_id")),
            },
        ))
    return events


def _prescription_events(tables: dict, subject_id: int, hadm_id: Optional[int]) -> list[TimelineEvent]:
    events = []
    df = tables.get("hosp/prescriptions")
    if df is None:
        return events
    rows = df[df["subject_id"] == subject_id]
    if hadm_id is not None:
        rows = rows[rows["hadm_id"] == hadm_id]
    for _, row in rows.iterrows():
        ts = _ts(row.get("starttime"))
        if not ts:
            continue
        drug = _safe_str(row.get("drug")) or "unknown drug"
        dose = _safe_str(row.get("dose_val_rx"))
        unit = _safe_str(row.get("dose_unit_rx"))
        route = _safe_str(row.get("route"))
        dose_str = f"{dose} {unit}" if dose else "dose unknown"
        events.append(TimelineEvent(
            source_table="hosp/prescriptions",
            source_field="drug",
            source_row_id=_safe_int(row.get("pharmacy_id")),
            subject_id=subject_id,
            hadm_id=_safe_int(row.get("hadm_id")),
            timestamp=ts,
            end_timestamp=_ts(row.get("stoptime")),
            event_type="medication",
            description=f"Prescription: {drug} — {dose_str} via {route or 'unknown route'}",
            code=drug,
            code_system="drug_name",
            value=dose,
            unit=unit,
        ))
    return events


def _icustays_events(tables: dict, subject_id: int, hadm_id: Optional[int]) -> list[TimelineEvent]:
    events = []
    df = tables.get("icu/icustays")
    if df is None:
        return events
    rows = df[df["subject_id"] == subject_id]
    if hadm_id is not None:
        rows = rows[rows["hadm_id"] == hadm_id]
    for _, row in rows.iterrows():
        ts = _ts(row.get("intime"))
        if not ts:
            continue
        careunit = _safe_str(row.get("first_careunit")) or "ICU"
        los = _safe_str(row.get("los"))
        events.append(TimelineEvent(
            source_table="icu/icustays",
            source_field="intime",
            source_row_id=_safe_int(row.get("stay_id")),
            subject_id=subject_id,
            hadm_id=_safe_int(row.get("hadm_id")),
            stay_id=_safe_int(row.get("stay_id")),
            timestamp=ts,
            end_timestamp=_ts(row.get("outtime")),
            event_type="icu_stay",
            description=f"ICU stay — unit: {careunit}, LOS: {los} days",
            code=careunit,
            code_system="careunit",
            value=los,
            unit="days",
        ))
    return events


def _chartevents_events(tables: dict, subject_id: int, hadm_id: Optional[int],
                        max_chart: int = 200) -> list[TimelineEvent]:
    events = []
    df = tables.get("icu/chartevents")
    items_df = tables.get("icu/d_items")
    if df is None:
        return events
    rows = df[df["subject_id"] == subject_id]
    if hadm_id is not None:
        rows = rows[rows["hadm_id"] == hadm_id]
    rows = rows.sort_values("charttime").head(max_chart)
    item_labels = {}
    if items_df is not None:
        item_labels = dict(zip(items_df["itemid"], items_df["label"]))
    for _, row in rows.iterrows():
        ts = _ts(row.get("charttime"))
        if not ts:
            continue
        itemid = _safe_int(row.get("itemid"))
        label = item_labels.get(itemid, f"itemid={itemid}")
        value = _safe_str(row.get("value"))
        valuenum = _safe_str(row.get("valuenum"))
        unit = _safe_str(row.get("valueuom"))
        display_val = valuenum or value or "no value"
        events.append(TimelineEvent(
            source_table="icu/chartevents",
            source_field="valuenum",
            source_row_id=f"{row.get('stay_id')}-{itemid}-{ts}",
            subject_id=subject_id,
            hadm_id=_safe_int(row.get("hadm_id")),
            stay_id=_safe_int(row.get("stay_id")),
            timestamp=ts,
            event_type="icu_obs",
            description=f"ICU observation: {label} = {display_val} {unit or ''}",
            value=display_val,
            unit=unit,
            code=str(itemid) if itemid else None,
            code_system="MIMIC_itemid",
            flag="warning" if row.get("warning") == 1 else None,
        ))
    return events


def _inputevents_events(tables: dict, subject_id: int, hadm_id: Optional[int]) -> list[TimelineEvent]:
    events = []
    df = tables.get("icu/inputevents")
    items_df = tables.get("icu/d_items")
    if df is None:
        return events
    rows = df[df["subject_id"] == subject_id]
    if hadm_id is not None:
        rows = rows[rows["hadm_id"] == hadm_id]
    item_labels = {}
    if items_df is not None:
        item_labels = dict(zip(items_df["itemid"], items_df["label"]))
    for _, row in rows.iterrows():
        ts = _ts(row.get("starttime"))
        if not ts:
            continue
        itemid = _safe_int(row.get("itemid"))
        label = item_labels.get(itemid, f"itemid={itemid}")
        amount = _safe_str(row.get("amount"))
        unit = _safe_str(row.get("amountuom"))
        events.append(TimelineEvent(
            source_table="icu/inputevents",
            source_field="amount",
            source_row_id=_safe_int(row.get("orderid")),
            subject_id=subject_id,
            hadm_id=_safe_int(row.get("hadm_id")),
            stay_id=_safe_int(row.get("stay_id")),
            timestamp=ts,
            end_timestamp=_ts(row.get("endtime")),
            event_type="icu_input",
            description=f"ICU input: {label} — {amount or '?'} {unit or ''}",
            value=amount,
            unit=unit,
            code=str(itemid) if itemid else None,
            code_system="MIMIC_itemid",
        ))
    return events


# ── Main builder function ────────────────────────────────────────────────────

def build_timeline(
    tables: dict,
    subject_id: int,
    hadm_id: Optional[int] = None,
    max_labs: int = 500,
    max_chart: int = 200,
) -> list[TimelineEvent]:
    """
    Build a time-ordered list of TimelineEvent objects for a patient.

    Args:
        tables      : dict of table_key → DataFrame (from data/ingest.py)
        subject_id  : patient subject_id
        hadm_id     : optional filter to one hospital admission
        max_labs    : cap on lab events per patient (can be very large)
        max_chart   : cap on ICU chart events per patient (often huge)

    Returns:
        List of TimelineEvent sorted by timestamp ascending.
        Events with timestamp="unknown" are appended last.
    """
    all_events: list[TimelineEvent] = []

    collectors = [
        _admissions_events,
        _transfers_events,
        _diagnoses_events,
        _procedures_events,
        lambda t, s, h: _lab_events(t, s, h, max_labs),
        _prescription_events,
        _icustays_events,
        lambda t, s, h: _chartevents_events(t, s, h, max_chart),
        _inputevents_events,
    ]

    for collector in collectors:
        try:
            events = collector(tables, subject_id, hadm_id)
            all_events.extend(events)
        except Exception as e:
            print(f"  Warning: collector {collector.__name__ if hasattr(collector, '__name__') else '?'} "
                  f"failed for subject {subject_id}: {e}", file=sys.stderr)

    # Sort: known timestamps first, "unknown" last
    def sort_key(e: TimelineEvent):
        if e.timestamp == "unknown" or e.timestamp is None:
            return "9999"
        return e.timestamp

    all_events.sort(key=sort_key)
    return all_events


def get_all_subject_ids(tables: dict) -> list[int]:
    """Return sorted list of all subject_ids in the dataset."""
    df = tables.get("hosp/patients")
    if df is not None:
        return sorted(df["subject_id"].unique().tolist())
    df = tables.get("hosp/admissions")
    if df is not None:
        return sorted(df["subject_id"].unique().tolist())
    return []


def print_timeline(events: list[TimelineEvent], subject_id: int, max_show: int = 50):
    """Pretty-print a timeline for debugging."""
    print(f"\n{'='*70}")
    print(f"PATIENT TIMELINE — subject_id={subject_id}  ({len(events)} events)")
    print(f"{'='*70}")
    for i, e in enumerate(events[:max_show]):
        print(f"[{i+1:3d}] {e.timestamp[:19] if len(e.timestamp) >= 10 else e.timestamp:<20} "
              f"{e.event_type:<14} {e.description[:65]}")
        print(f"       ↳ CITE: {e.citation_key()}")
    if len(events) > max_show:
        print(f"  ... {len(events) - max_show} more events not shown")
    print(f"{'='*70}\n")
