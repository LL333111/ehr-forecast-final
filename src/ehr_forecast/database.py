"""Read-only source ingestion into a strictly past clinical database."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import duckdb

from .io import sha, write
from .schema import EVENT_HEADERS, PATIENTS_HEADER
from .supplemental import COST_FIELDS, coverage_ledger

TABLES = tuple(EVENT_HEADERS)


def scan(con, path, expected, view):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        header = next(csv.reader(f))
    if header != list(expected) or len(header) != len(set(header)):
        raise ValueError(f"Unreviewed schema: {Path(path).name}")
    con.read_csv(str(path), header=True, all_varchar=True).create_view(view, replace=True)
    return header


def safe_text(expression):
    return f"CASE WHEN length({expression})<=512 AND NOT regexp_matches(coalesce({expression},''), '^(iVBOR|/9j/|data:|\\{{|\\[|<)') AND NOT regexp_matches(coalesce({expression},''), '^[-+0-9.eE ]{{80,}}$') THEN {expression} ELSE NULL END"


def file_inventory(data_root):
    rows = []
    summary = defaultdict(lambda: {"files": 0, "bytes": 0})
    for p in sorted(Path(data_root).rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(data_root).as_posix()
        key = "/".join(p.relative_to(data_root).parts[:-1]) or "."
        summary[key]["files"] += 1
        summary[key]["bytes"] += p.stat().st_size
        rows.append((rel, p.stat().st_size, p.suffix.lower()))
    return (rows, dict(summary))


def build(raw_root, anchors_path, root, *, fit_ids, dev_ids=()):
    """Build local storage; callers must explicitly supply training membership."""
    raw_root, anchors_path, root = map(Path, (raw_root, anchors_path, root))
    fit_ids, dev_ids = (list(fit_ids), list(dev_ids))
    if not fit_ids or len(set(fit_ids + dev_ids)) != len(fit_ids) + len(dev_ids):
        raise ValueError("Invalid research membership")
    root.mkdir(parents=True, exist_ok=True)
    final = root / "ehr.duckdb"
    pending = root / "ehr.partial.duckdb"
    if final.exists() or pending.exists():
        raise FileExistsError("Immutable database exists; choose a fresh output directory")
    files, inventory = file_inventory(raw_root)
    sources = [raw_root / "patient_splits.csv", anchors_path]
    sources += [
        raw_root / d / (t + ".csv") for d in ["train_val", "test"] for t in ["patients", *TABLES]
    ]
    source_hashes = {p.as_posix(): sha(p) for p in sources}
    con = duckdb.connect(str(pending), config={"threads": 2, "memory_limit": "3GB"})
    try:
        con.execute("SET TimeZone='UTC'")
        for schema in ["clinical", "research", "audit"]:
            con.execute(f"CREATE SCHEMA {schema}")
        con.execute("CREATE TABLE audit.files(path VARCHAR, bytes BIGINT, extension VARCHAR)")
        con.executemany("INSERT INTO audit.files VALUES (?,?,?)", files)
        ledger = coverage_ledger({**EVENT_HEADERS, "patients": PATIENTS_HEADER})
        con.execute(
            "CREATE TABLE audit.field_policy(source_table VARCHAR, source_column VARCHAR, disposition VARCHAR, reason VARCHAR, feature_group VARCHAR)"
        )
        con.executemany(
            "INSERT INTO audit.field_policy VALUES (?,?,?,?,?)",
            [(r["table"], r["column"], r["disposition"], r["reason"], r["group"]) for r in ledger],
        )
        con.execute(
            "INSERT INTO audit.field_policy VALUES ('observations','VALUE:29303009','quarantined_payload','ECG generation lineage unverified; examination fact retained',NULL), ('genomics','*','quarantined_modality','DNA and presence/proxy features excluded',NULL), ('imaging','*','quarantined_modality','DICOM content/date provenance unverified; past study fact retained',NULL)"
        )
        scan(con, raw_root / "patient_splits.csv", ["Id", "split"], "splits")
        con.read_csv(str(anchors_path), all_varchar=True).create_view("anchor_source")
        required = {"patient_id", "split", "anchor_date", "birth_date", "gender"}
        if not required <= {r[0] for r in con.execute("DESCRIBE anchor_source").fetchall()}:
            raise ValueError("Anchor schema mismatch")
        con.execute(
            "CREATE TABLE research.membership(patient_id VARCHAR PRIMARY KEY, role VARCHAR CHECK(role IN ('fit','dev')))"
        )
        con.executemany(
            "INSERT INTO research.membership VALUES (?,?)",
            [(p, "fit") for p in fit_ids] + [(p, "dev") for p in dev_ids],
        )
        for d in ["train_val", "test"]:
            scan(con, raw_root / d / "patients.csv", PATIENTS_HEADER, d + "_people")
        con.execute("""CREATE TABLE clinical.patients AS
          SELECT s.Id patient_id,s.split,try_cast(a.anchor_date AS DATE) anchor_date,
            try_cast(p.BIRTHDATE AS DATE) birth_date,p.GENDER gender,p.RACE race,
            p.ETHNICITY ethnicity,p.BIRTHPLACE birthplace,coalesce(m.role,'sealed') research_role
          FROM splits s JOIN anchor_source a ON s.Id=a.patient_id AND s.split=a.split
          JOIN (SELECT * FROM train_val_people UNION ALL SELECT * FROM test_people) p ON p.Id=s.Id
          LEFT JOIN research.membership m ON m.patient_id=s.Id""")
        con.execute("ALTER TABLE clinical.patients ADD PRIMARY KEY(patient_id)")
        if (
            con.execute("SELECT count(*) FROM clinical.patients").fetchone()[0]
            != con.execute("SELECT count(*) FROM splits").fetchone()[0]
        ):
            raise ValueError("Patient/anchor join not one-to-one")
        if con.execute(
            "SELECT count(*) FROM clinical.patients WHERE anchor_date IS NULL OR birth_date>anchor_date OR (research_role<>'sealed' AND split<>'train')"
        ).fetchone()[0]:
            raise ValueError("Invalid anchor or nontrain research patient")
        if con.execute(
            "SELECT count(*) FROM clinical.patients WHERE research_role<>'sealed'"
        ).fetchone()[0] != len(fit_ids) + len(dev_ids):
            raise ValueError("Missing research patient")
        con.execute(
            "CREATE TABLE audit.row_counts(source_table VARCHAR, raw_rows BIGINT, unknown_patient BIGINT, invalid_date BIGINT, prebirth BIGINT, time_mismatch BIGINT, at_or_after_anchor BIGINT, eligible BIGINT, stored BIGINT, duplicate_excess BIGINT)"
        )
        counts = {}
        for table in TABLES:
            header = EVENT_HEADERS[table]
            for d in ["train_val", "test"]:
                scan(con, raw_root / d / (table + ".csv"), header, d + "_source")
            start = "START" if "START" in header else "DATE"
            con.execute(
                f"CREATE OR REPLACE TEMP TABLE candidates AS\n              SELECT r.*,p.patient_id,p.anchor_date,p.birth_date,p.split,\n                try_cast(substr(r.{start},1,10) AS DATE) event_date,\n                CASE WHEN length(r.{start})>=19 THEN try_cast(r.{start} AS TIMESTAMPTZ)::TIMESTAMP ELSE NULL END event_ts\n              FROM (SELECT *,row_number() OVER() source_row,'train_val' source_partition FROM train_val_source\n                UNION ALL SELECT *,row_number() OVER(),'test' FROM test_source) r\n              LEFT JOIN clinical.patients p ON r.PATIENT=p.patient_id"
            )
            bad_clock = (
                f"(length({start})>=19 AND (event_ts IS NULL OR event_ts::DATE<>event_date))"
            )
            good = f"patient_id IS NOT NULL AND event_date IS NOT NULL AND event_date>=birth_date AND event_date<anchor_date AND NOT {bad_clock}"
            n = con.execute(
                f"SELECT count(*),count(*) FILTER(WHERE patient_id IS NULL),\n              count(*) FILTER(WHERE event_date IS NULL),count(*) FILTER(WHERE event_date<birth_date),\n              count(*) FILTER(WHERE {bad_clock}),count(*) FILTER(WHERE event_date>=anchor_date),\n              count(*) FILTER(WHERE {good}) FROM candidates"
            ).fetchone()
            if con.execute(
                "SELECT count(*) FROM candidates WHERE (source_partition='test')<>(split='test')"
            ).fetchone()[0]:
                raise ValueError("Source partition contradicts official split")

            def col(name):
                return '"' + name + '"' if name in header else "NULL::VARCHAR"

            def textcol(name):
                return safe_text(col(name))

            if "STOP" in header:
                con.execute(
                    f"CREATE OR REPLACE TEMP VIEW dated AS SELECT *,\n                   try_cast(substr(STOP,1,10) AS DATE) end_day,\n                   CASE WHEN length(STOP)>=19 THEN try_cast(STOP AS TIMESTAMPTZ)::TIMESTAMP END end_ts\n                   FROM candidates WHERE {good}"
                )
                end_good = "end_day>=event_date AND end_day<anchor_date AND (length(STOP)<19 OR end_ts IS NOT NULL) AND (end_ts IS NULL OR (end_ts::DATE=end_day AND end_ts>=coalesce(event_ts,event_date::TIMESTAMP)))"
                observed_end = f"CASE WHEN {end_good} THEN end_day END"
                duration = f"CASE WHEN {end_good} THEN CASE WHEN event_ts IS NOT NULL AND end_ts IS NOT NULL THEN (epoch(end_ts)-epoch(event_ts))/86400. ELSE date_diff('day',event_date,end_day) END END"
            else:
                con.execute(
                    f"CREATE OR REPLACE TEMP VIEW dated AS SELECT * FROM candidates WHERE {good}"
                )
                observed_end, duration = ("NULL::DATE", "NULL::DOUBLE")
            value = (
                f"CASE WHEN CODE='29303009' THEN NULL ELSE {textcol('VALUE')} END"
                if table == "observations"
                else "NULL::VARCHAR"
            )
            fields = [
                "patient_id",
                "split",
                "source_partition",
                "source_row",
                "event_date",
                "event_ts",
                f"{col('Id')} source_id",
                f"{(col('Id') if table == 'encounters' else col('ENCOUNTER'))} encounter_id",
                f"{col('CODE')} code",
                f"{textcol('DESCRIPTION')} description",
                f"{col('REASONCODE')} reason_code",
                f"{textcol('REASONDESCRIPTION')} reason_description",
                f"{observed_end} observed_end",
                f"{duration} observed_duration_days",
                f"{value} value_raw",
                f"{col('UNITS')} units",
                f"{col('TYPE')} obs_type",
                f"{col('ENCOUNTERCLASS')} encounter_class",
                f"try_cast({col(COST_FIELDS.get(table, ''))} AS DOUBLE) base_cost",
                f"{col('ORGANIZATION')} organization_id",
                f"{col('PROVIDER')} provider_id",
                f"{col('PAYER')} payer_id",
                f"{col('UDI')} device_link_id",
            ]
            for name in ["BODYSITE", "MODALITY", "SOP"]:
                fields += [
                    f"{col(name + '_CODE')} {name.lower()}_code",
                    f"{textcol(name + '_DESCRIPTION')} {name.lower()}_description",
                ]
            con.execute(
                "CREATE OR REPLACE TEMP TABLE projection AS SELECT "
                + ",".join(fields)
                + " FROM dated"
            )
            con.execute(
                f"CREATE TABLE clinical.{table} AS SELECT row_number() OVER() event_id,\n                * EXCLUDE(source_row),min(source_row) first_source_row,count(*)::BIGINT multiplicity\n                FROM projection GROUP BY ALL"
            )
            if table == "observations":
                con.execute("""CREATE TEMP TABLE conflicts AS SELECT patient_id,code,coalesce(units,'') units,event_ts
                  FROM clinical.observations WHERE lower(obs_type)='numeric' AND event_ts IS NOT NULL
                    AND isfinite(try_cast(value_raw AS DOUBLE)) GROUP BY 1,2,3,4
                    HAVING count(DISTINCT try_cast(value_raw AS DOUBLE))>1""")
                con.execute(
                    "ALTER TABLE clinical.observations ADD COLUMN numeric_conflict BOOLEAN DEFAULT false"
                )
                con.execute("""UPDATE clinical.observations o SET numeric_conflict=true FROM conflicts c
                  WHERE o.patient_id=c.patient_id AND o.code=c.code AND coalesce(o.units,'')=c.units AND o.event_ts=c.event_ts""")
                con.execute(
                    "UPDATE clinical.observations SET value_raw=NULL WHERE numeric_conflict OR (lower(obs_type)='numeric' AND NOT coalesce(isfinite(try_cast(value_raw AS DOUBLE)),false))"
                )
            else:
                con.execute(
                    f"ALTER TABLE clinical.{table} ADD COLUMN numeric_conflict BOOLEAN DEFAULT false"
                )
            con.execute(f"ALTER TABLE clinical.{table} ADD COLUMN numeric_value DOUBLE")
            if table == "observations":
                con.execute(
                    "UPDATE clinical.observations SET numeric_value=try_cast(value_raw AS DOUBLE) WHERE lower(obs_type)='numeric'"
                )
            con.execute(
                f"UPDATE clinical.{table} SET base_cost=NULL WHERE NOT coalesce(isfinite(base_cost),false)"
            )
            con.execute(f"ALTER TABLE clinical.{table} ADD PRIMARY KEY(event_id)")
            stored = con.execute(f"SELECT count(*) FROM clinical.{table}").fetchone()[0]
            con.execute(
                "INSERT INTO audit.row_counts VALUES (?,?,?,?,?,?,?,?,?,?)",
                [table, *n, stored, n[-1] - stored],
            )
            counts[table] = dict(
                zip(
                    [
                        "raw",
                        "unknown_patient",
                        "invalid_date",
                        "prebirth",
                        "time_mismatch",
                        "at_or_after_anchor",
                        "eligible",
                        "stored",
                        "duplicate_excess",
                    ],
                    [*n, stored, n[-1] - stored],
                )
            )
            print(json.dumps({"database_table": table, **counts[table]}), flush=True)
        unions = " UNION ALL ".join(
            (f"SELECT '{t}' source_table,* FROM clinical.{t}" for t in TABLES)
        )
        con.execute("CREATE VIEW clinical.events AS " + unions)
        con.execute(
            "CREATE VIEW research.patients AS SELECT p.* FROM clinical.patients p JOIN research.membership USING(patient_id)"
        )
        con.execute(
            "CREATE VIEW research.events AS SELECT e.*,m.role FROM clinical.events e JOIN research.membership m USING(patient_id)"
        )
        con.execute("""CREATE TABLE clinical.event_visit_links AS
          SELECT e.source_table,e.event_id,e.patient_id,e.encounter_id,
            count(v.event_id) matched_past_encounters FROM clinical.events e
          LEFT JOIN clinical.encounters v ON e.patient_id=v.patient_id AND e.encounter_id=v.encounter_id
          GROUP BY 1,2,3,4""")
        con.execute("CREATE TABLE audit.source_hashes(path VARCHAR, sha256 VARCHAR)")
        con.executemany("INSERT INTO audit.source_hashes VALUES (?,?)", list(source_hashes.items()))
        failures = con.execute("""SELECT count(*) FROM clinical.events e JOIN clinical.patients p USING(patient_id)
          WHERE e.event_date>=p.anchor_date OR e.event_date<p.birth_date OR e.event_date IS NULL
            OR e.observed_end>=p.anchor_date OR e.observed_duration_days<0""").fetchone()[0]
        if failures:
            raise ValueError("Temporal gate failed")
        if con.execute(
            "SELECT count(*) FROM clinical.observations WHERE code='29303009' AND value_raw IS NOT NULL"
        ).fetchone()[0]:
            raise ValueError("ECG payload survived")
        report = dict(
            complete=True,
            local_only=True,
            inventory=inventory,
            tables=counts,
            patients=dict(
                con.execute(
                    "SELECT split,count(*) FROM clinical.patients GROUP BY split"
                ).fetchall()
            ),
            research=dict(
                con.execute(
                    "SELECT role,count(*) FROM research.membership GROUP BY role"
                ).fetchall()
            ),
            research_events=con.execute("SELECT sum(multiplicity) FROM research.events").fetchone()[
                0
            ],
            numeric_conflict_rows=con.execute(
                "SELECT sum(multiplicity) FROM clinical.observations WHERE numeric_conflict"
            ).fetchone()[0],
            unmatched_visit_events=con.execute(
                "SELECT count(*) FROM clinical.event_visit_links WHERE matched_past_encounters=0 AND encounter_id IS NOT NULL"
            ).fetchone()[0],
            temporal_violations=failures,
            source_hashes=source_hashes,
            labels_read=False,
            official_outcomes_read=False,
            preprocessing_fitted=False,
            limitations=[
                "Synthetic data; clinical plausibility is not certified",
                "Source timestamps establish an event date, not an independently verified recording-availability timestamp",
                "DNA/ECG/DICOM contents remain quarantined; no file-presence features",
                "Official held-out patients have local past-only storage and never fit feature statistics",
                "Exact projected duplicates compacted with multiplicity; not assumed erroneous",
                "No unit conversion without a verified equivalence; unavailable numeric values remain null",
            ],
        )
        con.execute("CHECKPOINT")
    finally:
        con.close()
    if any((sha(p) != digest for p, digest in source_hashes.items())):
        raise ValueError("Source changed during database build")
    pending.replace(final)
    report["database_sha256"] = sha(final)
    write(root / "audit.json", report)
    write(root / "field_policy.json", ledger)
    return report
