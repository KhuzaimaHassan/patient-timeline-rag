"""
timeline/models.py — TimelineEvent dataclass
=============================================
Every event in a patient's timeline is represented as a TimelineEvent.
The source_* fields provide the full citation trail back to the raw CSV.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Any
import json


@dataclass
class TimelineEvent:
    """
    A single timestamped event in a patient's clinical record.

    source_table  : the CSV table name (e.g. "hosp/labevents")
    source_field  : primary value field (e.g. "valuenum")
    source_row_id : row identifier within that table (e.g. labevent_id)
    subject_id    : patient identifier
    hadm_id       : hospital admission id (None for ICU-only events)
    stay_id       : ICU stay id (None for hosp-only events)
    timestamp     : primary event time (ISO 8601 string, already shifted/deidentified)
    event_type    : high-level category (admission, transfer, diagnosis, procedure,
                    lab, medication, icu_obs, icu_input, icu_output)
    description   : human-readable summary (NOT clinical interpretation)
    value         : primary measurement value (string to handle mixed types)
    unit          : unit of measurement
    code          : structured code (ICD, LOINC item_id, drug name, etc.)
    code_system   : which coding system (ICD9/ICD10/itemid/drug)
    extra         : any additional metadata not captured above
    """
    # --- Required fields ---
    source_table: str
    source_field: str
    source_row_id: Any
    subject_id: int
    timestamp: str
    event_type: str
    description: str

    # --- Optional clinical fields ---
    hadm_id: Optional[int] = None
    stay_id: Optional[int] = None
    value: Optional[str] = None
    unit: Optional[str] = None
    code: Optional[str] = None
    code_system: Optional[str] = None
    flag: Optional[str] = None          # e.g. "abnormal" from labevents
    end_timestamp: Optional[str] = None # for ranged events (ICU stay, med infusion)

    # --- Flexible overflow ---
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    def citation_key(self) -> str:
        """Returns a short citation string suitable for display in the UI."""
        return (
            f"{self.source_table} | "
            f"row={self.source_row_id} | "
            f"subject_id={self.subject_id} | "
            f"t={self.timestamp}"
        )
