import csv
from datetime import date, datetime

import duckdb

from ehr_forecast.event_schema import EVENT_COLUMNS


def row(
    patient="p1",
    source="observations",
    code="lab",
    value="10",
    unit="mg",
    when="2020-01-01 12:00:00",
    kind="numeric",
    **kw,
):
    stamp = datetime.fromisoformat(when) if when else None
    r = dict(
        patient_id=patient,
        split="train",
        source_table=source,
        source_file="train_val/" + source + ".csv",
        event_date=stamp.date() if stamp else date(2020, 1, 1),
        event_time_raw=when,
        event_ts=stamp,
        code=code,
        encounter_id="enc",
        value_raw=value,
        units=unit,
        obs_type=kind,
        modality_code=None,
        bodysite_code=None,
        sop_code=None,
        reason_code=None,
    )
    r.update(kw)
    return r


def connection(rows, people=None):
    c = duckdb.connect(config={"threads": 2, "memory_limit": "512MB"})
    types = {"event_date": "DATE", "event_ts": "TIMESTAMP"}
    c.execute(
        "CREATE TABLE e("
        + ",".join((f"{k} {types.get(k, 'VARCHAR')}" for k in EVENT_COLUMNS))
        + ")"
    )
    if rows:
        c.executemany(
            "INSERT INTO e VALUES (" + ",".join(("?" for _ in EVENT_COLUMNS)) + ")",
            [[r[k] for k in EVENT_COLUMNS] for r in rows],
        )
    c.execute(
        "CREATE TABLE a(patient_id VARCHAR,split VARCHAR,anchor_date DATE,birth_date DATE,gender VARCHAR)"
    )
    c.executemany(
        "INSERT INTO a VALUES (?,?,?,?,?)",
        people
        or [
            ("p1", "train", date(2021, 1, 1), date(1980, 1, 1), "F"),
            ("p2", "train", date(2021, 1, 1), None, "M"),
            ("v", "val", date(2021, 1, 1), date(1990, 1, 1), "F"),
        ],
    )
    return c


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
