# Data Manifest — MIMIC-IV Demo v2.2

> **Research and educational prototype only. Not for clinical use.**

## Dataset

| Field | Value |
|---|---|
| Dataset | MIMIC-IV Clinical Database Demo |
| Version | 2.2 |
| Source | https://physionet.org/content/mimic-iv-demo/2.2/ |
| DOI | https://doi.org/10.13026/dp1f-ex47 |
| Licence | PhysioNet Credentialed Health Data Licence 1.5.0 |
| Patients | 100 (deidentified, date-shifted) |
| Institution | Beth Israel Deaconess Medical Center, Boston, USA |

## Tables Used

### Hospital Module (`hosp/`)

| Table | Key Fields Used | Purpose in App |
|---|---|---|
| `patients.csv` | `subject_id`, `gender`, `anchor_age`, `dod` | Patient demographics |
| `admissions.csv` | `subject_id`, `hadm_id`, `admittime`, `dischtime`, `admission_type`, `insurance`, `race` | Hospital admission events |
| `transfers.csv` | `subject_id`, `hadm_id`, `transfer_id`, `eventtype`, `careunit`, `intime`, `outtime` | Ward/ICU movement timeline |
| `diagnoses_icd.csv` | `subject_id`, `hadm_id`, `seq_num`, `icd_code`, `icd_version` | Diagnosis codes |
| `d_icd_diagnoses.csv` | `icd_code`, `icd_version`, `long_title` | Diagnosis descriptions |
| `procedures_icd.csv` | `subject_id`, `hadm_id`, `seq_num`, `icd_code`, `icd_version`, `chartdate` | Procedure codes |
| `d_icd_procedures.csv` | `icd_code`, `icd_version`, `long_title` | Procedure descriptions |
| `labevents.csv` | `subject_id`, `hadm_id`, `itemid`, `charttime`, `value`, `valuenum`, `valueuom`, `flag` | Lab results |
| `d_labitems.csv` | `itemid`, `label`, `fluid`, `category` | Lab item descriptions |
| `prescriptions.csv` | `subject_id`, `hadm_id`, `starttime`, `stoptime`, `drug`, `dose_val_rx`, `dose_unit_rx`, `route` | Medication orders |
| `pharmacy.csv` | `subject_id`, `hadm_id`, `poe_id`, `starttime`, `drug`, `status` | Pharmacy events |

### ICU Module (`icu/`)

| Table | Key Fields Used | Purpose in App |
|---|---|---|
| `icustays.csv` | `subject_id`, `hadm_id`, `stay_id`, `first_careunit`, `last_careunit`, `intime`, `outtime`, `los` | ICU stay metadata |
| `chartevents.csv` | `subject_id`, `hadm_id`, `stay_id`, `itemid`, `charttime`, `value`, `valuenum`, `valueuom` | ICU vitals/observations (sampled) |
| `inputevents.csv` | `subject_id`, `hadm_id`, `stay_id`, `itemid`, `starttime`, `endtime`, `amount`, `amountuom`, `rate` | IV fluids, vasopressors |
| `d_items.csv` | `itemid`, `label`, `category`, `unitname` | ICU item descriptions |

## Tables NOT Used

| Table | Reason |
|---|---|
| MIMIC-IV-Note (noteevents, etc.) | Not part of MIMIC-IV Demo v2.2; free-text notes explicitly excluded per challenge rules |
| `microbiologyevents.csv` | Out of scope for initial prototype |
| `procedureevents.csv` | Overlaps with procedures_icd; not included in v1 |
| `datetimeevents.csv` | Supplementary; not in initial scope |

## Privacy Handling

- All data is deidentified and date-shifted per PhysioNet protocol
- No patient-level rows are committed to the repository
- FAISS index is gitignored (regenerated locally)
- No patient data is sent to external APIs (local FAISS + careful LLM prompt design)
- `test_set.json` contains annotated QA pairs with minimal patient identifiers (subject_id only, no row dumps)

## Transformations Applied

| Step | Description | Reversible |
|---|---|---|
| Timestamp normalization | Convert all datetime fields to UTC ISO 8601 | Yes (original CSVs preserved) |
| ICD code join | Join diagnoses/procedures with description tables | Yes |
| Fact serialization | Convert each row to natural-language text chunk for embedding | Yes (code in `data/ingest.py`) |
| Sampling | chartevents sampled to max 1000 events per patient (configurable) | Yes (`CHARTEVENT_SAMPLE_SIZE` in `.env`) |

## Citation

```
Johnson, A. E. W., Bulgarelli, L., Shen, L., Gayles, A., Shammout, A., Horng, S., 
Pollard, T. J., Hao, S., Moody, B., Gow, B., Lehman, L. H., Celi, L. A., & Mark, R. G. (2023).
MIMIC-IV, a freely accessible electronic health record dataset.
Scientific Data, 10(1). https://doi.org/10.1038/s41597-022-01899-x

PhysioNet dataset (v2.2): https://doi.org/10.13026/dp1f-ex47
```
