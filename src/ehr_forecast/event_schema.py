"""Event columns and training-membership helpers."""

from __future__ import annotations

EVENT_COLUMNS = (
    "patient_id",
    "split",
    "source_table",
    "source_file",
    "event_date",
    "event_time_raw",
    "event_ts",
    "code",
    "encounter_id",
    "value_raw",
    "units",
    "obs_type",
    "modality_code",
    "bodysite_code",
    "sop_code",
    "reason_code",
)


def set_fit_ids(con, ids: list[str]):
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("nonempty unique fitting patients required")
    con.execute("CREATE OR REPLACE TEMP TABLE fit_ids(patient_id VARCHAR PRIMARY KEY)")
    con.executemany("INSERT INTO fit_ids VALUES (?)", [(p,) for p in ids])
    n = con.execute("SELECT count(*) FROM fit_ids JOIN people USING(patient_id)").fetchone()[0]
    if n != len(ids):
        raise ValueError("fit IDs outside prepared official train patients")
