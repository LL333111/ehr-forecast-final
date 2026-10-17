"""Raw CSV preparation and frozen feature transforms."""

from __future__ import annotations

import csv
import hashlib
import json
from contextlib import contextmanager
from datetime import date
from pathlib import Path

import duckdb
import numpy as np

from . import baseline as base
from . import database as database
from . import engineering as engineering
from . import events as event_data
from . import schema as audit
from . import supplemental as supplemental
from . import temporal as trajectories
from .interactions import bind_clinical, fit_extra, prepare_extra, transform_extra
from .io import read, sha, write
from .labels import labels as make_labels
from .labels import shift_years
from .payloads import sanitize_events
from .policy import annual_row, cost_free_columns, quarantine_reason_fields


def id_hash(ids):
    return hashlib.sha256(json.dumps(sorted(ids), separators=(",", ":")).encode()).hexdigest()


def references(raw):
    raw = Path(raw)
    with (raw / "patient_splits.csv").open(encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames != ["Id", "split"]:
            raise ValueError("Unreviewed split schema")
        rows = list(r)
    splits = {r["Id"]: r["split"] for r in rows}
    if (
        len(splits) != len(rows)
        or not splits
        or any((not p or s not in ["train", "val", "test"] for p, s in splits.items()))
    ):
        raise ValueError("Invalid official split membership")
    with (raw / "target_conditions.csv").open(encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames != ["CODE", "DESCRIPTION"]:
            raise ValueError("Unreviewed target schema")
        targets = list(r)
    codes = [r["CODE"] for r in targets]
    if len(set(codes)) != len(codes) or any((not c for c in codes)):
        raise ValueError("Invalid target order")
    roles = {s: sorted((p for p, v in splits.items() if v == s)) for s in ["train", "val", "test"]}
    return (splits, roles, targets)


def assert_train_ids(ids, splits, require_all=False):
    if not ids or len(ids) != len(set(ids)) or any((splits.get(p) != "train" for p in ids)):
        raise ValueError("Fitting is restricted to unique official train patients")
    if require_all and set(ids) != {p for p, s in splits.items() if s == "train"}:
        raise ValueError("Final fitting must use the complete official train split")


def raw_hashes(raw):
    raw = Path(raw)
    paths = [
        raw / n
        for n in ["patient_splits.csv", "target_conditions.csv", "test_anchors.csv", "ReadMe.md"]
    ]
    paths += [
        raw / d / (t + ".csv")
        for d in ["train_val", "test"]
        for t in ["patients", *database.TABLES]
    ]
    return {p.resolve().as_posix(): sha(p) for p in paths}


def strict_database(path):
    """Apply the frozen availability policy to this new database."""
    with duckdb.connect(str(path)) as c:
        predicate = "upper(trim(coalesce(code,''))) IN ('QALY','DALY','QOLS') OR upper(trim(coalesce(description,''))) IN ('QALY','DALY','QOLS')"
        removed = c.execute(
            f"SELECT sum(multiplicity) FROM clinical.observations WHERE {predicate}"
        ).fetchone()[0]
        c.execute(
            f"DELETE FROM clinical.event_visit_links WHERE source_table='observations' AND event_id IN (SELECT event_id FROM clinical.observations WHERE {predicate})"
        )
        c.execute(f"DELETE FROM clinical.observations WHERE {predicate}")
        for t in database.TABLES:
            c.execute(
                f"UPDATE clinical.{t} SET base_cost=NULL,reason_code=NULL,reason_description=NULL"
            )
        c.execute(
            "UPDATE audit.field_policy SET disposition='excluded_strict',reason='Frozen availability policy',feature_group=NULL WHERE source_column IN ('BASE_COST','BASE_ENCOUNTER_COST','REASONCODE','REASONDESCRIPTION')"
        )
        for schema, name in c.execute(
            "SELECT schema_name,view_name FROM duckdb_views() WHERE NOT internal AND NOT temporary AND schema_name='main'"
        ).fetchall():
            if not name.replace("_", "").isalnum():
                raise ValueError("Unreviewed loader view")
            c.execute(f'DROP VIEW main."{name}"')
        if c.execute(
            "SELECT count(*) FROM duckdb_views() WHERE NOT internal AND regexp_matches(lower(sql),'read_csv|read_parquet|csv_scan|parquet_scan')"
        ).fetchone()[0]:
            raise ValueError("External loader survived database finalization")
        n = c.execute("SELECT count(*),sum(multiplicity) FROM clinical.events").fetchone()
        c.execute("CHECKPOINT")
    return dict(
        annual_source_rows_removed=int(removed or 0),
        stored_events=n[0],
        qualified_source_events=n[1],
    )


def build_raw_cache(raw, cache):
    raw, cache = map(Path, (raw, cache))
    splits, roles, targets = references(raw)
    before = raw_hashes(raw)
    if (cache / "complete.json").exists():
        report = read(cache / "complete.json")
        if report["raw_hashes"] != before or any(
            (sha(cache / p) != h for p, h in report["files"].items())
        ):
            raise ValueError("Raw inputs or cached history changed")
        return report
    if cache.exists() and any(cache.iterdir()):
        raise FileExistsError("Use a fresh cache directory")
    cache.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(config={"threads": 2, "memory_limit": "3GB"}) as c:
        c.execute("SET TimeZone='UTC'")
        audit._load_references(c, raw)
        audit._build_anchors(c, raw)
        cohort = base._cohort_tables(c, raw, cache)
        if cohort["patients"] != roles:
            raise ValueError("Anchor patients differ from official membership")
        reconciliation = base._collect_events(c, raw)
        c.execute("""CREATE TEMP TABLE strict_safe AS SELECT * REPLACE(NULL::VARCHAR AS reason_code,
            CASE WHEN source_table='observations' AND code='29303009' THEN NULL ELSE value_raw END AS value_raw)
            FROM safe_events WHERE NOT(source_table='observations' AND upper(trim(coalesce(code,''))) IN ('QALY','DALY','QOLS'))""")
        path = str((cache / "safe_events.parquet").resolve()).replace("'", "''")
        c.execute(f"COPY strict_safe TO '{path}' (FORMAT PARQUET)")
    database.build(
        raw, cache / "anchors.csv", cache / "database", fit_ids=roles["train"], dev_ids=[]
    )
    policy = strict_database(cache / "database" / "ehr.duckdb")
    if before != raw_hashes(raw):
        raise ValueError("Raw inputs changed during construction")
    result = dict(
        complete=True,
        raw_hashes=before,
        official_counts={s: len(v) for s, v in roles.items()},
        target_codes=[t["CODE"] for t in targets],
        labels_opened=False,
        preprocessing_fitted=False,
        strict_policy=policy,
        reconciliation=reconciliation,
        files={
            p: sha(cache / p) for p in ["anchors.csv", "safe_events.parquet", "database/ehr.duckdb"]
        },
    )
    write(cache / "complete.json", result)
    return result


def source_stream(raw, table, allowed):
    expected = (
        supplemental.PATIENTS_HEADER if table == "patients" else supplemental.EVENT_HEADERS[table]
    )
    for partition in ["train_val", "test"]:
        with (Path(raw) / partition / (table + ".csv")).open(encoding="utf-8-sig", newline="") as f:
            r = csv.DictReader(f)
            if r.fieldnames != list(expected):
                raise ValueError("Unreviewed raw table schema")
            for row in r:
                if row["Id" if table == "patients" else "PATIENT"] not in allowed or annual_row(
                    table, row
                ):
                    continue
                yield quarantine_reason_fields(row)


def baseline_views(con, ids, dates, splits):
    con.execute(
        "CREATE OR REPLACE TEMP TABLE requested(patient_id VARCHAR PRIMARY KEY,anchor_date DATE,split VARCHAR)"
    )
    con.executemany(
        "INSERT INTO requested VALUES (?,?,?)", [(p, dates[p][0], splits[p]) for p in ids]
    )
    con.execute("CREATE OR REPLACE TEMP VIEW anchors AS SELECT * FROM requested")
    con.execute(
        "CREATE OR REPLACE TEMP VIEW safe_events AS SELECT e.* FROM base_events e JOIN requested r USING(patient_id) WHERE e.event_date<r.anchor_date AND e.split=r.split"
    )
    con.execute(
        "CREATE OR REPLACE TEMP VIEW cohort AS SELECT p.* FROM base_people p JOIN requested USING(patient_id)"
    )
    con.execute("""CREATE OR REPLACE TEMP VIEW code_mentions AS
        SELECT patient_id,split,event_date,encounter_id,source_table AS family,code FROM safe_events WHERE code IS NOT NULL
        UNION ALL SELECT patient_id,split,event_date,encounter_id,source_table||'_reason',reason_code FROM safe_events WHERE reason_code IS NOT NULL
        UNION ALL SELECT patient_id,split,event_date,encounter_id,'imaging_modality',modality_code FROM safe_events WHERE modality_code IS NOT NULL
        UNION ALL SELECT patient_id,split,event_date,encounter_id,'imaging_bodysite',bodysite_code FROM safe_events WHERE bodysite_code IS NOT NULL
        UNION ALL SELECT patient_id,split,event_date,encounter_id,'imaging_sop',sop_code FROM safe_events WHERE sop_code IS NOT NULL""")


def baseline_transform(con, ids, dates, demo, splits, schema):
    roles = {splits[p] for p in ids}
    if len(roles) != 1:
        raise ValueError("Transform one official split at a time")
    role = next(iter(roles))
    baseline_views(con, ids, dates, splits)
    x = base._blank_matrix(len(ids), schema)
    index = {p: i for i, p in enumerate(ids)}
    base._fill_demographics(x, schema, ids, demo, dates)
    base._fill_history(con, x, schema, index, role)
    base._fill_codes(con, x, schema, index, dates, role)
    base._fill_labs(con, x, schema, index, dates, role)
    base._fill_text(con, x, schema, index, role)
    return x


@contextmanager
def feature_context(raw, cache, ids=None):
    raw, cache = map(Path, (raw, cache))
    splits, roles, targets = references(raw)
    ids = list(ids) if ids is not None else sum(roles.values(), [])
    if not ids or len(ids) != len(set(ids)) or (not set(ids) <= set(splits)):
        raise ValueError("Invalid transform membership")
    with duckdb.connect(str(cache / "database" / "ehr.duckdb"), read_only=True) as db:
        anchors = dict(
            db.execute("SELECT patient_id,anchor_date FROM clinical.patients").fetchall()
        )
    additions = supplemental.extract_from_rows(
        {t: source_stream(raw, t, set(ids)) for t in supplemental.TABLES},
        source_stream(raw, "patients", set(ids)),
        anchors,
        ids,
        text_hash_dim=1024,
    )
    print(json.dumps(dict(stage="supplemental", patients=len(ids))), flush=True)
    with (
        duckdb.connect(config={"threads": 2, "memory_limit": "3GB"}) as rich,
        duckdb.connect(config={"threads": 2, "memory_limit": "3GB"}) as c,
    ):
        bind_clinical(rich, ids, cache / "database" / "ehr.duckdb")
        rich.execute(
            "UPDATE research.membership SET role=(SELECT 'transform_'||split FROM research.patients p WHERE p.patient_id=research.membership.patient_id)"
        )
        generated, generated_schema = engineering.engineer(rich, ids)
        prepare_extra(rich)
        c.execute("SET TimeZone='UTC'")
        c.read_parquet(str(cache / "safe_events.parquet")).create_view("source_events")
        c.read_csv(str(cache / "anchors.csv"), all_varchar=True).create_view("a")
        sanitize_events(c)
        event_report = event_data.prepare(c, ids)
        trajectories.prepare_tabular(c)
        c.execute(
            "CREATE TEMP TABLE base_people AS SELECT patient_id,official_split split,anchor_date,birth_date,gender FROM people"
        )
        c.execute("""CREATE TEMP VIEW base_events AS SELECT e.* REPLACE(
          CASE WHEN e.source_table='observations' AND lower(coalesce(e.obs_type,''))='numeric'
            AND (f.patient_id IS NOT NULL OR NOT coalesce(isfinite(try_cast(e.value_raw AS DOUBLE)),false))
            THEN NULL ELSE e.value_raw END AS value_raw)
          FROM e LEFT JOIN conflicts f ON e.patient_id=f.patient_id AND e.code=f.code
            AND coalesce(e.units,'')=f.units AND e.event_ts=f.event_ts""")
        rows = c.execute(
            "SELECT patient_id,anchor_date,birth_date,gender FROM base_people"
        ).fetchall()
        dates = {p: (a, shift_years(a, 5)) for p, a, _, _ in rows}
        demo = {p: (b, g) for p, _, b, g in rows}
        yield dict(
            c=c,
            rich=rich,
            additions=additions,
            splits=splits,
            roles=roles,
            dates=dates,
            demo=demo,
            generated=generated,
            generated_schema=generated_schema,
            positions={p: i for i, p in enumerate(ids)},
            event_report=event_report,
            targets=targets,
        )


def fit_representation(ctx, ids, require_all=True):
    assert_train_ids(ids, ctx["splits"], require_all)
    baseline_views(ctx["c"], ids, ctx["dates"], ctx["splits"])
    schema = base._feature_schema(ctx["c"])
    temporal = trajectories.fit(ctx["c"], ids, min_support=1)
    supp = supplemental.fit_protocol(ctx["additions"], ids)
    extra = fit_extra(ctx["rich"], ids, min_support=20)
    catalog = [dict(r, group="baseline_" + r["family"]) for r in schema]
    catalog += [
        dict(r, group=json.loads(r["name"])[0], family=json.loads(r["name"])[0])
        for r in temporal["columns"]
    ]
    catalog += supp["schema"] + ctx["generated_schema"]
    keep = cost_free_columns(catalog)
    catalog = [catalog[i] for i in keep]
    return dict(
        baseline_schema=schema,
        temporal=temporal,
        supplemental=supp,
        extra=extra,
        keep=keep.tolist(),
        catalog=catalog + extra,
        base_width=len(catalog),
        fit_count=len(ids),
        fit_hash=id_hash(ids),
        policy="v18 strict definitions, refitted on official train only",
    )


def transform(ctx, protocol, ids):
    if not ids or len(set(ids)) != len(ids) or (not set(ids) <= set(ctx["positions"])):
        raise ValueError("Invalid transform IDs")
    parts = [
        baseline_transform(
            ctx["c"], ids, ctx["dates"], ctx["demo"], ctx["splits"], protocol["baseline_schema"]
        ),
        trajectories.transform_tabular(ctx["c"], protocol["temporal"], ids),
        supplemental.transform(ctx["additions"], ids, protocol["supplemental"]),
        ctx["generated"][[ctx["positions"][p] for p in ids]],
    ]
    keep = np.asarray(protocol["keep"], dtype=np.int64)
    x = np.concatenate(
        [
            np.concatenate(parts, axis=1)[:, keep],
            transform_extra(ctx["rich"], protocol["extra"], ids),
        ],
        axis=1,
    )
    if np.isinf(x).any():
        raise ValueError("Infinite model input")
    return x.astype(np.float32, copy=False)


def priors(cache, ids, codes):
    result = np.zeros((len(ids), len(codes)), np.float32)
    rows = {p: i for i, p in enumerate(ids)}
    cols = {c: j for j, c in enumerate(codes)}
    with duckdb.connect(str(Path(cache) / "database" / "ehr.duckdb"), read_only=True) as c:
        for pid, code in c.execute(
            "SELECT DISTINCT patient_id,code FROM clinical.conditions"
        ).fetchall():
            if pid in rows and code in cols:
                result[rows[pid], cols[code]] = 1
    return result


def train_labels(raw, cache, ids, codes):
    splits, _, _ = references(raw)
    assert_train_ids(ids, splits, True)
    allowed = set(ids)
    first = {}
    with (Path(raw) / "train_val" / "conditions.csv").open(encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames != list(audit.EVENT_HEADERS["conditions"]):
            raise ValueError("Unreviewed label schema")
        for row in r:
            if row["PATIENT"] not in allowed or row["CODE"] not in codes:
                continue
            d = date.fromisoformat(row["START"][:10])
            key = (row["PATIENT"], row["CODE"])
            if key not in first or d < first[key]:
                first[key] = d
    with duckdb.connect(str(Path(cache) / "database" / "ehr.duckdb"), read_only=True) as c:
        dates = {
            p: (a, shift_years(a, 5))
            for p, a in c.execute(
                "SELECT patient_id,anchor_date FROM clinical.patients WHERE split='train'"
            ).fetchall()
        }
    y, prior = make_labels(first, ids, codes, dates)
    if not np.array_equal(prior, priors(cache, ids, codes)):
        raise ValueError("Train prior mismatch")
    return (y, prior)
