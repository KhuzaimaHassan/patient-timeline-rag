"""
data/ingest.py — MIMIC-IV Demo v2.2 Data Ingestion
====================================================
Loads every CSV.GZ table, validates columns against the official schema,
reports row counts and null statistics. Does NOT silently drop or alter
any source values — flags issues for downstream awareness.

Usage:
    python data/ingest.py [--data-dir data/mimic-iv-demo]
"""

import os
import sys
import gzip
import shutil
import argparse
import pandas as pd
from datetime import datetime
from pathlib import Path

# ── Official MIMIC-IV Demo v2.2 expected schema (table → required columns) ──
EXPECTED_SCHEMA = {
    # HOSP module
    "hosp/admissions": [
        "subject_id", "hadm_id", "admittime", "dischtime", "deathtime",
        "admission_type", "admit_provider_id", "admission_location",
        "discharge_location", "insurance", "language", "marital_status",
        "race", "edregtime", "edouttime", "hospital_expire_flag",
    ],
    "hosp/d_hcpcs": ["code", "category", "long_description", "short_description"],
    "hosp/d_icd_diagnoses": ["icd_code", "icd_version", "long_title"],
    "hosp/d_icd_procedures": ["icd_code", "icd_version", "long_title"],
    "hosp/d_labitems": ["itemid", "label", "fluid", "category"],
    "hosp/diagnoses_icd": ["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"],
    "hosp/drgcodes": ["subject_id", "hadm_id", "drg_type", "drg_code", "description", "drg_severity", "drg_mortality"],
    "hosp/emar": ["subject_id", "hadm_id", "emar_id", "emar_seq", "poe_id", "pharmacy_id",
                  "enter_provider_id", "charttime", "medication", "event_txt",
                  "scheduletime", "storetime"],
    "hosp/emar_detail": ["subject_id", "emar_id", "emar_seq", "parent_field_ordinal",
                         "administration_type", "pharmacy_id", "barcode_type",
                         "reason_for_action", "route", "site", "side", "frequency",
                         "frequency_int", "dose_due", "dose_due_unit", "dose_given",
                         "dose_given_unit", "will_remainder_of_dose_be_given",
                         "product_amount_given", "product_unit", "product_code",
                         "product_description", "product_description_other",
                         "prior_infusion_rate", "infusion_rate", "infusion_rate_adjustment",
                         "infusion_rate_adjustment_amount", "infusion_rate_unit",
                         "new_iv_bag_hung", "continued_infusion_in_other_location",
                         "restart_reason", "side_effect", "warning", "reason_for_no_barcode",
                         "override_reason", "override_provider_id", "administered_dose",
                         "administered_dose_unit", "actual_dose_due_date"],
    "hosp/hcpcsevents": ["subject_id", "hadm_id", "chartdate", "hcpcs_cd", "seq_num", "short_description"],
    "hosp/labevents": ["labevent_id", "subject_id", "hadm_id", "specimen_id", "itemid",
                       "order_provider_id", "charttime", "storetime", "value", "valuenum",
                       "valueuom", "ref_range_lower", "ref_range_upper", "flag",
                       "priority", "comments"],
    "hosp/microbiologyevents": ["microevent_id", "subject_id", "hadm_id", "micro_specimen_id",
                                "order_provider_id", "chartdate", "charttime", "spec_itemid",
                                "spec_type_desc", "test_seq", "storedate", "storetime",
                                "test_itemid", "test_name", "org_itemid", "org_name",
                                "isolate_num", "quantity", "ab_itemid", "ab_name",
                                "dilution_text", "dilution_comparison", "dilution_value",
                                "interpretation", "comments"],
    "hosp/omr": ["subject_id", "chartdate", "seq_num", "result_name", "result_value"],
    "hosp/patients": ["subject_id", "gender", "anchor_age", "anchor_year",
                      "anchor_year_group", "dod"],
    "hosp/pharmacy": ["subject_id", "hadm_id", "pharmacy_id", "poe_id", "starttime",
                      "stoptime", "medication", "proc_type", "status", "entertime",
                      "verifiedtime", "route", "frequency", "disp_sched",
                      "infusion_type", "sliding_scale", "lockout_interval",
                      "basal_rate", "one_hr_max", "doses_per_24_hrs", "duration",
                      "duration_interval", "expiration_value", "expiration_unit",
                      "expirationdate", "dispensation", "fill_quantity"],
    "hosp/poe": ["poe_id", "poe_seq", "subject_id", "hadm_id", "ordertime",
                 "order_type", "order_subtype", "transaction_type", "discontinue_of_poe_id",
                 "discontinued_by_poe_id", "enter_provider_id", "order_status"],
    "hosp/poe_detail": ["poe_id", "poe_seq", "subject_id", "field_name", "field_value"],
    "hosp/prescriptions": ["subject_id", "hadm_id", "pharmacy_id", "poe_id", "poe_seq",
                           "order_provider_id", "starttime", "stoptime", "drug_type",
                           "drug", "formulary_drug_cd", "gsn", "ndc", "prod_strength",
                           "form_rx", "dose_val_rx", "dose_unit_rx", "form_val_disp",
                           "form_unit_disp", "doses_per_24_hrs", "route"],
    "hosp/procedures_icd": ["subject_id", "hadm_id", "seq_num", "chartdate", "icd_code", "icd_version"],
    "hosp/provider": ["provider_id"],
    "hosp/services": ["subject_id", "hadm_id", "transfertime", "prev_service", "curr_service"],
    "hosp/transfers": ["subject_id", "hadm_id", "transfer_id", "eventtype",
                       "careunit", "intime", "outtime"],
    # ICU module
    "icu/caregiver": ["caregiver_id"],
    "icu/chartevents": ["subject_id", "hadm_id", "stay_id", "caregiver_id",
                        "charttime", "storetime", "itemid", "value", "valuenum",
                        "valueuom", "warning"],
    "icu/d_items": ["itemid", "label", "abbreviation", "linksto",
                    "category", "unitname", "param_type", "lownormalvalue", "highnormalvalue"],
    "icu/datetimeevents": ["subject_id", "hadm_id", "stay_id", "caregiver_id",
                           "charttime", "storetime", "itemid", "value", "valueuom", "warning"],
    "icu/icustays": ["subject_id", "hadm_id", "stay_id", "first_careunit",
                     "last_careunit", "intime", "outtime", "los"],
    "icu/ingredientevents": ["subject_id", "hadm_id", "stay_id", "caregiver_id",
                             "starttime", "endtime", "storetime", "itemid",
                             "amount", "amountuom", "rate", "rateuom",
                             "orderid", "linkorderid", "statusdescription",
                             "originalamount", "originalrate"],
    "icu/inputevents": ["subject_id", "hadm_id", "stay_id", "caregiver_id",
                        "starttime", "endtime", "storetime", "itemid",
                        "amount", "amountuom", "rate", "rateuom",
                        "orderid", "linkorderid", "ordercategoryname",
                        "secondaryordercategoryname", "ordercomponenttypedescription",
                        "ordercategorydescription", "patientweight",
                        "totalamount", "totalamountuom", "isopenbag",
                        "continueinnextdept", "statusdescription",
                        "originalamount", "originalrate"],
    "icu/outputevents": ["subject_id", "hadm_id", "stay_id", "caregiver_id",
                         "charttime", "storetime", "itemid", "value", "valueuom"],
    "icu/procedureevents": ["subject_id", "hadm_id", "stay_id", "caregiver_id",
                            "starttime", "endtime", "storetime", "itemid",
                            "value", "valueuom", "location", "locationcategory",
                            "orderid", "linkorderid", "ordercategoryname",
                            "secondaryordercategoryname", "ordercategorydescription",
                            "patientweight", "isopenbag", "continueinnextdept",
                            "statusdescription", "originalamount", "originalrate"],
}


def find_data_dir(base_dir: str) -> Path:
    """Find the MIMIC-IV Demo directory, handling physionet.org/files/... structure."""
    base = Path(base_dir)
    # Direct path
    if (base / "hosp").exists():
        return base
    # wget -r style: physionet.org/files/mimic-iv-demo/2.2/
    for pattern in [
        base / "physionet.org" / "files" / "mimic-iv-demo" / "2.2",
        base.parent / "physionet.org" / "files" / "mimic-iv-demo" / "2.2",
    ]:
        if pattern.exists():
            return pattern
    return base


def decompress_gz_if_needed(gz_path: Path) -> Path:
    """Decompress .csv.gz → .csv next to the original. Returns the csv path."""
    csv_path = gz_path.with_suffix("")  # strip .gz
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return csv_path
    print(f"    Decompressing {gz_path.name} ...", end="", flush=True)
    with gzip.open(gz_path, "rb") as f_in, open(csv_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    print(f" → {csv_path.name} ({csv_path.stat().st_size:,} bytes)")
    return csv_path


def load_table(data_dir: Path, table_key: str) -> pd.DataFrame | None:
    """Load a table (auto-handles .csv and .csv.gz). Returns None if file missing."""
    module, tname = table_key.split("/")
    for suffix in [".csv", ".csv.gz"]:
        fpath = data_dir / module / f"{tname}{suffix}"
        if fpath.exists():
            if suffix == ".csv.gz":
                fpath = decompress_gz_if_needed(fpath)
            return pd.read_csv(fpath, low_memory=False)
    return None


def validate_columns(df: pd.DataFrame, table_key: str, expected_cols: list[str]) -> list[str]:
    """Return list of expected columns missing from df."""
    actual = set(df.columns.str.lower())
    expected = set(c.lower() for c in expected_cols)
    return sorted(expected - actual)


def null_report(df: pd.DataFrame, top_n: int = 5) -> dict:
    """Return {col: null_count} for top_n columns with most nulls."""
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0].sort_values(ascending=False).head(top_n)
    return nulls.to_dict()


def run_ingestion(data_dir_arg: str) -> dict:
    """Main ingestion routine. Returns dict of table_key → DataFrame."""
    data_dir = find_data_dir(data_dir_arg)
    print(f"\n{'='*65}")
    print(f"MIMIC-IV Demo v2.2 — Ingestion Report")
    print(f"Data directory: {data_dir}")
    print(f"Timestamp: {datetime.utcnow().isoformat()}Z")
    print(f"{'='*65}\n")

    results = {}
    missing_tables = []
    schema_issues = {}

    for table_key, expected_cols in sorted(EXPECTED_SCHEMA.items()):
        module = table_key.split("/")[0].upper()
        tname = table_key.split("/")[1]
        print(f"[{module}] {tname}")

        df = load_table(data_dir, table_key)
        if df is None:
            print(f"  ⚠️  FILE NOT FOUND — skipping")
            missing_tables.append(table_key)
            continue

        # Column validation
        missing_cols = validate_columns(df, table_key, expected_cols)
        extra_cols = sorted(set(df.columns) - set(expected_cols))

        print(f"  Rows: {len(df):>10,}  |  Cols: {len(df.columns)}")
        if missing_cols:
            print(f"  ❌ Missing cols: {missing_cols}")
            schema_issues[table_key] = missing_cols
        if extra_cols:
            print(f"  ℹ️  Extra cols (not in schema): {extra_cols}")

        # Null report
        nulls = null_report(df)
        if nulls:
            null_str = ", ".join(f"{c}:{n}" for c, n in nulls.items())
            print(f"  Nulls (top 5): {null_str}")

        results[table_key] = df
        print()

    # Summary
    print(f"\n{'='*65}")
    print(f"SUMMARY")
    print(f"  Tables loaded:  {len(results)}/{len(EXPECTED_SCHEMA)}")
    print(f"  Missing tables: {len(missing_tables)}")
    if missing_tables:
        for t in missing_tables:
            print(f"    - {t}")
    print(f"  Schema issues:  {len(schema_issues)}")
    if schema_issues:
        for t, cols in schema_issues.items():
            print(f"    - {t}: missing {cols}")
    total_rows = sum(len(df) for df in results.values())
    print(f"  Total rows across all tables: {total_rows:,}")
    print(f"{'='*65}\n")

    return results, missing_tables, schema_issues


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest MIMIC-IV Demo v2.2")
    parser.add_argument(
        "--data-dir",
        default=os.path.join(os.path.dirname(__file__), "mimic-iv-demo"),
        help="Path to the mimic-iv-demo directory",
    )
    args = parser.parse_args()
    tables, missing, issues = run_ingestion(args.data_dir)
    print(f"Ingestion complete. {len(tables)} tables loaded.")
