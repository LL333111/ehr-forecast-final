"""Clinical history features and training-only quality filtering."""

from __future__ import annotations

import hashlib

import numpy as np

from .database import TABLES

WINDOWS = [("d30", 30), ("d90", 90), ("y1", 365), ("y5", 1826), ("full", None)]


def engineer(con, ids):
    """No outcome argument, fitted vocabulary, population quantile, or scaling."""
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Unique patient order required")
    membership = dict(con.execute("SELECT patient_id,role FROM research.membership").fetchall())
    if not set(ids) <= set(membership):
        raise ValueError("Patient is outside the requested extraction context")
    con.execute("CREATE OR REPLACE TEMP TABLE requested(patient_id VARCHAR PRIMARY KEY)")
    con.executemany("INSERT INTO requested VALUES (?)", [(p,) for p in ids])
    con.execute("""CREATE OR REPLACE TEMP VIEW e AS SELECT e.*,p.anchor_date,
      date_diff('day',e.event_date,p.anchor_date) days_before
      FROM research.events e JOIN research.patients p USING(patient_id) JOIN requested USING(patient_id)""")
    if con.execute("SELECT count(*) FROM e WHERE days_before<=0").fetchone()[0]:
        raise ValueError("Future data entered feature store")
    index = {p: i for i, p in enumerate(ids)}
    vectors = []
    catalog = []

    def add(name, values, group, default="nan", source="multiple"):
        arr = np.full(len(ids), 0.0 if default == "zero" else np.nan, dtype=np.float32)
        for p, v in values:
            if v is not None:
                arr[index[p]] = v
        if np.isinf(arr).any():
            raise ValueError("Infinite engineered feature")
        vectors.append(arr)
        catalog.append(
            dict(
                name="v15|" + name,
                group=group,
                family="v15_" + group,
                source=source,
                default=default,
            )
        )

    table_data = {}
    for table in TABLES:
        for label, days in WINDOWS:
            cutoff = "" if days is None else f" AND days_before<={days}"
            rows = con.execute(
                f"SELECT patient_id,sum(multiplicity) n,count(DISTINCT event_date) active_days,\n              count(DISTINCT code) codes,count(DISTINCT encounter_id) visits,\n              avg(base_cost) mean_base_cost,\n              sum(multiplicity) FILTER(WHERE observed_end IS NOT NULL)*1.0/sum(multiplicity) observed_end_fraction\n              FROM e WHERE source_table=? {cutoff} GROUP BY patient_id",
                [table],
            ).fetchall()
            for j, stat in enumerate(
                [
                    "events",
                    "active_days",
                    "distinct_codes",
                    "visits",
                    "mean_base_cost",
                    "observed_end_fraction",
                ],
                1,
            ):
                add(
                    f"{table}|{label}|{stat}",
                    [(r[0], r[j]) for r in rows],
                    "recent_utilization",
                    "zero" if j <= 4 else "nan",
                    table,
                )
            table_data[table, label] = {r[0]: r[1] for r in rows}
        for halflife in [90, 365, 1826]:
            rows = con.execute(
                """SELECT patient_id,sum(multiplicity*exp(-ln(2)*days_before/?))
              FROM e WHERE source_table=? GROUP BY patient_id""",
                [halflife, table],
            ).fetchall()
            add(f"{table}|decay_half_life_{halflife}", rows, "time_decay", "zero", table)
        rows = con.execute(
            """WITH dates AS (SELECT DISTINCT patient_id,event_date,days_before FROM e WHERE source_table=?),
          gaps AS (SELECT *,date_diff('day',lag(event_date) OVER(PARTITION BY patient_id ORDER BY event_date),event_date) gap FROM dates)
          SELECT patient_id,min(days_before),max(days_before),avg(gap),stddev_pop(gap),
            median(gap),max(gap),count(*) FROM gaps GROUP BY patient_id""",
            [table],
        ).fetchall()
        for j, stat in enumerate(
            [
                "recency",
                "history_days",
                "gap_mean",
                "gap_std",
                "gap_median",
                "gap_max",
                "active_days",
            ],
            1,
        ):
            add(
                f"{table}|rhythm|{stat}", [(r[0], r[j]) for r in rows], "event_rhythm", "nan", table
            )
        history = {r[0]: r[2] for r in rows}
        for recent, longer in [("d30", "y1"), ("d90", "y1"), ("y1", "y5")]:
            a = dict(WINDOWS)[recent]
            b = dict(WINDOWS)[longer]
            vals = []
            for p, h in history.items():
                nr = table_data[table, recent].get(p, 0)
                nl = table_data[table, longer].get(p, 0)
                ratio = (nr / max(1, min(h, a)) + 0.001) / (
                    max(0, nl - nr) / max(1, min(h, b) - min(h, a)) + 0.001
                )
                vals.append((p, float(np.log1p(ratio))))
            add(
                f"{table}|rate_ratio|{recent}_vs_previous_{longer}",
                vals,
                "utilization_change",
                "nan",
                table,
            )
        rows = con.execute(
            """WITH c AS (SELECT patient_id,code,sum(multiplicity) n FROM e WHERE source_table=? GROUP BY 1,2),
          p AS (SELECT *,n/sum(n) OVER(PARTITION BY patient_id) fraction FROM c)
          SELECT patient_id,-sum(fraction*ln(fraction)),max(fraction) FROM p GROUP BY patient_id""",
            [table],
        ).fetchall()
        add(f"{table}|code_entropy", [(r[0], r[1]) for r in rows], "composition", "nan", table)
        add(
            f"{table}|largest_code_fraction",
            [(r[0], r[2]) for r in rows],
            "composition",
            "nan",
            table,
        )
    ages = dict(
        con.execute(
            "SELECT patient_id,date_diff('day',birth_date,anchor_date)/365.2425 FROM research.patients"
        ).fetchall()
    )
    for label, _ in WINDOWS:
        totals = {p: sum((table_data[t, label].get(p, 0) for t in TABLES)) for p in ids}
        for table in TABLES:
            vals = table_data[table, label]
            add(
                f"{table}|{label}|share_of_all_records",
                [(p, vals.get(p, 0) / totals[p] if totals[p] else None) for p in ids],
                "composition",
                "nan",
                table,
            )
            add(
                f"{table}|{label}|age_x_log_burden",
                [(p, ages[p] / 100 * np.log1p(vals.get(p, 0))) for p in ids if ages[p] is not None],
                "explicit_interaction",
                "nan",
                table,
            )
        encounters = table_data["encounters", label]
        for table in ["conditions", "medications", "procedures", "observations"]:
            vals = table_data[table, label]
            add(
                f"{table}|{label}|events_per_encounter",
                [(p, vals.get(p, 0) / encounters[p]) for p in ids if encounters.get(p, 0)],
                "explicit_interaction",
                "nan",
                table,
            )
    x = np.stack(vectors, axis=1)
    return (x, catalog)


def fit_quality(x):
    """Remove only training-constant and exactly duplicate projected columns.

    Missing is a distinct state, not zero. Hashes accelerate equality detection;
    an actual value comparison guards the collision case.
    """
    if x.ndim != 2 or not len(x) or np.isinf(x).any():
        raise ValueError("Invalid fit matrix")
    keep = []
    reasons = []
    seen = {}
    for j in range(x.shape[1]):
        values = np.array(x[:, j], dtype=np.float32, copy=True)
        present = np.isfinite(values)
        if not present.any() or (present.all() and np.all(values == values[0])):
            reasons.append(dict(column=j, reason="fit_constant_or_all_missing"))
            continue
        values[~present] = np.nan
        digest = hashlib.sha256(values.tobytes()).hexdigest()
        candidates = seen.get(digest, [])
        duplicate = next(
            (k for k in candidates if np.array_equal(values, x[:, k], equal_nan=True)), None
        )
        if duplicate is not None:
            reasons.append(dict(column=j, reason="fit_exact_duplicate", representative=duplicate))
            continue
        seen.setdefault(digest, []).append(j)
        keep.append(j)
    if not keep:
        raise ValueError("No varying feature")
    return (np.asarray(keep, dtype=np.int64), reasons)
