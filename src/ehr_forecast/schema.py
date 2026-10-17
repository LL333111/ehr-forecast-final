"""Raw CSV schemas and retrospective anchor construction."""

from __future__ import annotations

from pathlib import Path

import duckdb

PATIENTS_HEADER = [
    "Id",
    "BIRTHDATE",
    "DEATHDATE",
    "SSN",
    "DRIVERS",
    "PASSPORT",
    "PREFIX",
    "FIRST",
    "LAST",
    "SUFFIX",
    "MAIDEN",
    "MARITAL",
    "RACE",
    "ETHNICITY",
    "GENDER",
    "BIRTHPLACE",
    "ADDRESS",
    "CITY",
    "STATE",
    "COUNTY",
    "ZIP",
    "LAT",
    "LON",
    "HEALTHCARE_EXPENSES",
    "HEALTHCARE_COVERAGE",
]
EVENT_SPECS: dict[str, dict[str, str | None]] = {
    "allergies": {"patient_key": "PATIENT", "time_col": "START", "end_col": "STOP"},
    "careplans": {"patient_key": "PATIENT", "time_col": "START", "end_col": "STOP"},
    "conditions": {"patient_key": "PATIENT", "time_col": "START", "end_col": "STOP"},
    "devices": {"patient_key": "PATIENT", "time_col": "START", "end_col": "STOP"},
    "encounters": {"patient_key": "PATIENT", "time_col": "START", "end_col": "STOP"},
    "imaging_studies": {"patient_key": "PATIENT", "time_col": "DATE", "end_col": None},
    "immunizations": {"patient_key": "PATIENT", "time_col": "DATE", "end_col": None},
    "medications": {"patient_key": "PATIENT", "time_col": "START", "end_col": "STOP"},
    "observations": {"patient_key": "PATIENT", "time_col": "DATE", "end_col": None},
    "procedures": {"patient_key": "PATIENT", "time_col": "DATE", "end_col": None},
}
EVENT_HEADERS: dict[str, list[str]] = {
    "allergies": ["START", "STOP", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION"],
    "careplans": [
        "Id",
        "START",
        "STOP",
        "PATIENT",
        "ENCOUNTER",
        "CODE",
        "DESCRIPTION",
        "REASONCODE",
        "REASONDESCRIPTION",
    ],
    "conditions": ["START", "STOP", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION"],
    "devices": ["START", "STOP", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION", "UDI"],
    "encounters": [
        "Id",
        "START",
        "STOP",
        "PATIENT",
        "ORGANIZATION",
        "PROVIDER",
        "PAYER",
        "ENCOUNTERCLASS",
        "CODE",
        "DESCRIPTION",
        "BASE_ENCOUNTER_COST",
        "TOTAL_CLAIM_COST",
        "PAYER_COVERAGE",
        "REASONCODE",
        "REASONDESCRIPTION",
    ],
    "imaging_studies": [
        "Id",
        "DATE",
        "PATIENT",
        "ENCOUNTER",
        "BODYSITE_CODE",
        "BODYSITE_DESCRIPTION",
        "MODALITY_CODE",
        "MODALITY_DESCRIPTION",
        "SOP_CODE",
        "SOP_DESCRIPTION",
    ],
    "immunizations": ["DATE", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION", "BASE_COST"],
    "medications": [
        "START",
        "STOP",
        "PATIENT",
        "PAYER",
        "ENCOUNTER",
        "CODE",
        "DESCRIPTION",
        "BASE_COST",
        "PAYER_COVERAGE",
        "DISPENSES",
        "TOTALCOST",
        "REASONCODE",
        "REASONDESCRIPTION",
    ],
    "observations": [
        "DATE",
        "PATIENT",
        "ENCOUNTER",
        "CODE",
        "DESCRIPTION",
        "VALUE",
        "UNITS",
        "TYPE",
    ],
    "procedures": [
        "DATE",
        "PATIENT",
        "ENCOUNTER",
        "CODE",
        "DESCRIPTION",
        "BASE_COST",
        "REASONCODE",
        "REASONDESCRIPTION",
    ],
}
FULL_HISTORY_ENCOUNTERS = ("train_val", "encounters.csv")


class AuditError(Exception):
    """Raised when the audit cannot account for every input row."""


def sql_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def date_expr(alias: str, column: str) -> str:
    """Calendar date from a Synthea date or UTC timestamp string."""
    ident = sql_ident(column)
    return f"try_cast(left(nullif(trim({alias}.{ident}), ''), 10) AS DATE)"


def _register_csv(con: duckdb.DuckDBPyConnection, view: str, path: Path) -> None:
    con.execute(
        f"\n        CREATE OR REPLACE VIEW {sql_ident(view)} AS\n        SELECT * FROM read_csv('{sql_path(path)}', header=true, all_varchar=true)\n        "
    )


def _load_references(con: duckdb.DuckDBPyConnection, raw: Path) -> None:
    _register_csv(con, "splits_raw", raw / "patient_splits.csv")
    con.execute(
        f"\n        CREATE TEMP TABLE splits AS\n        SELECT trim({sql_ident('Id')}) AS patient_id, trim({sql_ident('split')}) AS split\n        FROM splits_raw\n        "
    )
    duplicate_splits = con.execute("""
        SELECT patient_id, count(*) AS n
        FROM splits
        GROUP BY 1
        HAVING count(*) > 1
        LIMIT 5
        """).fetchall()
    if duplicate_splits:
        raise AuditError(f"duplicate patient ids in patient_splits: {duplicate_splits}")
    bad_splits = con.execute("""
        SELECT DISTINCT split FROM splits
        WHERE split NOT IN ('train', 'val', 'test') OR split IS NULL
        """).fetchall()
    if bad_splits:
        raise AuditError(f"unexpected split labels: {bad_splits}")
    _register_csv(con, "anchors_raw", raw / "test_anchors.csv")
    con.execute(
        f"\n        CREATE TEMP TABLE test_anchor_dates AS\n        SELECT\n            trim({sql_ident('Id')}) AS patient_id,\n            {date_expr('anchors_raw', 'anchor_date')} AS anchor_date\n        FROM anchors_raw\n        "
    )
    duplicate_anchors = con.execute("""
        SELECT patient_id, count(*) AS n
        FROM test_anchor_dates
        GROUP BY 1
        HAVING count(*) > 1
        LIMIT 5
        """).fetchall()
    if duplicate_anchors:
        raise AuditError(f"duplicate test anchor ids: {duplicate_anchors}")
    for source in ("train_val", "test"):
        _register_csv(con, f"patients_{source}", raw / source / "patients.csv")
        con.execute(
            f"\n            CREATE TEMP TABLE patient_ids_{source} AS\n            SELECT trim({sql_ident('Id')}) AS patient_id\n            FROM {sql_ident(f'patients_{source}')}\n            "
        )
        duplicates = con.execute(
            f"\n            SELECT patient_id, count(*) AS n\n            FROM patient_ids_{source}\n            GROUP BY 1\n            HAVING count(*) > 1\n            LIMIT 5\n            "
        ).fetchall()
        if duplicates:
            raise AuditError(f"duplicate patient ids in {source}/patients.csv: {duplicates}")


def _build_anchors(con: duckdb.DuckDBPyConnection, raw: Path) -> None:
    """Anchor train/val from full encounters. Never read test encounters here."""
    encounters = raw.joinpath(*FULL_HISTORY_ENCOUNTERS)
    if encounters.parent.name != "train_val":
        raise AuditError("full-history encounters must come from train_val")
    _register_csv(con, "full_history_encounters", encounters)
    recorded = "coalesce(" + date_expr("e", "STOP") + ", " + date_expr("e", "START") + ")"
    con.execute(
        f"\n        CREATE TEMP TABLE last_encounters AS\n        SELECT trim(e.{sql_ident('PATIENT')}) AS patient_id,\n               max({recorded}) AS last_encounter_date\n        FROM full_history_encounters e\n        WHERE nullif(trim(e.{sql_ident('PATIENT')}), '') IS NOT NULL\n        GROUP BY 1\n        "
    )
    con.execute("""
        CREATE TEMP TABLE anchors AS
        SELECT
            s.patient_id,
            s.split,
            CAST(
                CASE
                    WHEN s.split = 'test' THEN ta.anchor_date
                    WHEN s.split IN ('train', 'val')
                        THEN l.last_encounter_date - INTERVAL 5 YEAR
                    ELSE NULL
                END
            AS DATE) AS anchor_date,
            CAST(
                CASE
                    WHEN s.split IN ('train', 'val') THEN l.last_encounter_date
                    ELSE NULL
                END
            AS DATE) AS last_encounter_date,
            CAST(
                CASE
                    WHEN s.split = 'test' AND ta.anchor_date IS NOT NULL
                        THEN ta.anchor_date + INTERVAL 5 YEAR
                    WHEN s.split IN ('train', 'val') AND l.last_encounter_date IS NOT NULL
                        THEN (l.last_encounter_date - INTERVAL 5 YEAR) + INTERVAL 5 YEAR
                    ELSE NULL
                END
            AS DATE) AS label_end_date
        FROM splits s
        LEFT JOIN last_encounters l
            ON s.patient_id = l.patient_id
           AND s.split IN ('train', 'val')
        LEFT JOIN test_anchor_dates ta
            ON s.patient_id = ta.patient_id
           AND s.split = 'test'
        """)
