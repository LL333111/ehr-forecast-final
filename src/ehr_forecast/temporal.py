"""Measurement trajectories and historical event rhythm."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from .event_schema import set_fit_ids

CHANNELS = [
    "value_asinh_robust",
    "value_missing",
    "is_numeric",
    "invalid_numeric",
    "numeric_conflict",
    "scale_unknown",
    "age_centuries",
    "age_missing",
    "log1p_days_before",
    "log1p_delta_days",
    "date_only",
    "log1p_multiplicity",
    "is_anchor",
    "delta_exact",
]


def digest_ids(ids):
    return hashlib.sha256(json.dumps(sorted(ids), separators=(",", ":")).encode()).hexdigest()


def prepare_tabular(con):
    """Create long feature rows; no labels or cohort-fitted statistics are used here."""
    con.execute("""CREATE OR REPLACE TEMP TABLE feature_rows(
        patient_id VARCHAR,feature VARCHAR,value DOUBLE,zero_default BOOLEAN)""")
    con.execute("""INSERT INTO feature_rows
        WITH s AS (
          SELECT c.patient_id,source_table,count(*) AS compact_n,sum(multiplicity) AS raw_n,
            count(DISTINCT event_date) AS active_days,
            count(DISTINCT nullif(encounter_id,'')) AS encounters,
            date_diff('day',min(event_date),max(event_date)) AS span,
            avg(date_only::INT) AS date_only_fraction,
            count(DISTINCT event_date) FILTER(WHERE event_date>=anchor_date-INTERVAL 1 YEAR) AS active_days_y1,
            count(DISTINCT event_date) FILTER(WHERE event_date>=anchor_date-INTERVAL 90 DAY) AS active_days_d90,
            sum(multiplicity-1) AS duplicates,
            sum(numeric_conflict::INT) AS conflicts,
            sum(invalid_numeric::INT) AS invalids
          FROM clean c JOIN people p USING(patient_id) GROUP BY 1,2
        ) SELECT patient_id,CAST(to_json(['quality',source_table,stat]) AS VARCHAR),val,true
          FROM s,LATERAL (VALUES ('active_days',active_days),('distinct_encounters',encounters),
            ('active_days_y1',active_days_y1),('active_days_d90',active_days_d90),
            ('date_only_fraction',date_only_fraction),('duplicate_excess',duplicates),
            ('numeric_conflict_rows',conflicts),('invalid_numeric_rows',invalids),
            ('records_per_observed_year',raw_n*365.2425/(span+1)),
            ('unique_rows_per_active_day',compact_n*1.0/active_days)
          ) AS f(stat,val)""")
    con.execute("""INSERT INTO feature_rows
        WITH s AS (
          SELECT c.patient_id,source_table,code,
            date_diff('day',min(event_date),anchor_date) AS first_days,
            date_diff('day',min(event_date),max(event_date)) AS span_days,
            count(DISTINCT event_date) AS observed_dates
          FROM clean c JOIN people p USING(patient_id)
          WHERE source_table IN ('conditions','medications','careplans') AND code IS NOT NULL
          GROUP BY 1,2,3,anchor_date
        ) SELECT patient_id,CAST(to_json(['record_clock',source_table,code,stat]) AS VARCHAR),val,false
          FROM s,LATERAL (VALUES ('days_since_first',first_days),('observed_span_days',span_days),
            ('observed_dates',observed_dates)) AS f(stat,val)""")
    con.execute("""INSERT INTO feature_rows
        WITH windows AS (
          SELECT m.*,w.win,epoch(event_ts)/86400. AS day,
            epoch(CAST(anchor_date AS TIMESTAMP))/86400. AS anchor_day
          FROM measurements m JOIN people p USING(patient_id),
            LATERAL (VALUES ('full',true),('y1',event_ts>=anchor_date-INTERVAL 1 YEAR),
              ('y5',event_ts>=anchor_date-INTERVAL 5 YEAR)) AS w(win,keep) WHERE keep
        ), ranked AS (
          SELECT *,row_number() OVER(PARTITION BY patient_id,code,units,win ORDER BY event_ts DESC) rn FROM windows
        ), s AS (
          SELECT patient_id,code,units,win,count(*) n,median(value) med,
            quantile_cont(value,.75)-quantile_cont(value,.25) iqr,stddev_pop(value) sd,
            max(day) FILTER(WHERE rn=1)-max(day) FILTER(WHERE rn=2) gap,
            max(value) FILTER(WHERE rn=1)-max(value) FILTER(WHERE rn=2) delta,
            regr_slope(value,day)*365.2425 slope,
            max(anchor_day)-max(day) days_since_last_usable
          FROM ranked GROUP BY 1,2,3,4
        ) SELECT patient_id,CAST(to_json(['numeric_trajectory',code,units,win,stat]) AS VARCHAR),val,false
          FROM s,LATERAL (VALUES ('median',med),('iqr',iqr),('std',sd),
            ('unique_timestamps',n),('last_usable_gap_days',gap),
            ('last_usable_delta_per_year',CASE WHEN gap>0 THEN delta/gap*365.2425 END),
            ('ols_slope_per_year',CASE WHEN n>1 THEN slope END),
            ('days_since_last_usable',days_since_last_usable)) AS f(stat,val)""")
    con.execute("""INSERT INTO feature_rows
        WITH s AS (
          SELECT c.patient_id,code,coalesce(units,'') units,
            bool_or(numeric_conflict) FILTER(WHERE event_ts=last_ts) AS latest_conflict,
            sum(numeric_conflict::INT) AS conflict_rows
          FROM (SELECT *,max(event_ts) OVER(PARTITION BY patient_id,code,coalesce(units,'')) last_ts
                FROM clean WHERE is_numeric) c GROUP BY 1,2,3
        ) SELECT patient_id,CAST(to_json(['numeric_quality',code,units,stat]) AS VARCHAR),val,false
          FROM s,LATERAL (VALUES ('latest_record_conflict',latest_conflict::INT),
            ('conflict_rows',conflict_rows)) AS f(stat,val)""")
    con.execute("""CREATE OR REPLACE TEMP TABLE text_states AS
        SELECT patient_id,code,event_ts,min(value_raw) AS value,count(DISTINCT value_raw) AS variants
        FROM clean WHERE source_table='observations' AND NOT is_numeric
          AND event_ts IS NOT NULL AND value_raw IS NOT NULL AND value_raw<>''
        GROUP BY 1,2,3""")
    con.execute("""INSERT INTO feature_rows
        WITH s AS (
          SELECT c.patient_id,code,value_raw,
            date_diff('day',max(event_date),anchor_date) recency,
            count(DISTINCT event_date) observed_dates
          FROM clean c JOIN people p USING(patient_id)
          WHERE source_table='observations' AND NOT is_numeric AND value_raw IS NOT NULL AND value_raw<>''
          GROUP BY 1,2,3,anchor_date
        ) SELECT patient_id,CAST(to_json(['text_value',code,value_raw,stat]) AS VARCHAR),val,z
          FROM s,LATERAL (VALUES ('days_since_last',recency,false),('observed_dates',observed_dates,true)) f(stat,val,z)""")
    con.execute("""INSERT INTO feature_rows
        WITH ranked AS (
          SELECT *,row_number() OVER(PARTITION BY patient_id,code ORDER BY event_ts DESC) rn,
            lag(value) OVER(PARTITION BY patient_id,code ORDER BY event_ts) prev,
            lag(variants) OVER(PARTITION BY patient_id,code ORDER BY event_ts) prev_variants
          FROM text_states
        ), s AS (
          SELECT patient_id,code,count(*) n,sum((variants>1)::INT) conflicts,
            max((variants>1)::INT) FILTER(WHERE rn=1) latest_conflict,
            count(*) FILTER(WHERE variants=1 AND prev_variants=1 AND value<>prev) changes
          FROM ranked GROUP BY 1,2
        ) SELECT patient_id,CAST(to_json(['text_state',code,stat]) AS VARCHAR),val,false
          FROM s,LATERAL (VALUES ('observed_timestamps',n),('conflicting_timestamps',conflicts),
            ('latest_record_conflict',latest_conflict),('adjacent_unambiguous_changes',changes)) f(stat,val)""")
    con.execute("""INSERT INTO feature_rows
        SELECT patient_id,CAST(to_json(['text_value',code,value,'latest_unambiguous']) AS VARCHAR),1,true
        FROM (SELECT *,row_number() OVER(PARTITION BY patient_id,code ORDER BY event_ts DESC) rn FROM text_states)
        WHERE rn=1 AND variants=1""")
    if con.execute("""SELECT count(*) FROM (SELECT patient_id,feature,count(*) n
        FROM feature_rows GROUP BY 1,2 HAVING n>1)""").fetchone()[0]:
        raise ValueError("duplicate additive feature keys")
    con.execute("UPDATE feature_rows SET value=NULL WHERE NOT isfinite(value)")


def fit(con, train_ids, min_support=20):
    if not isinstance(min_support, int) or min_support < 1:
        raise ValueError("min_support must be a positive integer")
    set_fit_ids(con, train_ids)
    keys = [
        r[0]
        for r in con.execute("""SELECT DISTINCT token_key FROM tokens
        JOIN fit_ids USING(patient_id) ORDER BY 1""").fetchall()
    ]
    families = [
        r[0]
        for r in con.execute("""SELECT DISTINCT family FROM tokens
        JOIN fit_ids USING(patient_id) ORDER BY 1""").fetchall()
    ]
    labs = con.execute("""SELECT CAST(to_json([code,units]) AS VARCHAR),median(value),
        quantile_cont(value,.75)-quantile_cont(value,.25),stddev_pop(value),
        count(*),count(DISTINCT patient_id)
        FROM measurements JOIN fit_ids USING(patient_id) GROUP BY code,units ORDER BY 1""").fetchall()
    scales = {
        k: dict(
            center=float(m),
            scale=float(i if i > 1e-08 else s if s > 1e-08 else 1.0),
            n=int(n),
            patients=int(p),
        )
        for k, m, i, s, n, p in labs
    }
    columns = con.execute(
        """SELECT feature,any_value(zero_default),count(DISTINCT patient_id)
        FROM feature_rows JOIN fit_ids USING(patient_id) WHERE value IS NOT NULL
        GROUP BY feature HAVING count(DISTINCT patient_id)>=? ORDER BY feature""",
        [min_support],
    ).fetchall()
    return dict(
        version=3,
        fit_patient_count=len(train_ids),
        fit_patient_hash=digest_ids(train_ids),
        min_support=min_support,
        channels=CHANNELS,
        types={k: i + 2 for i, k in enumerate(families)},
        codes={k: i + 2 for i, k in enumerate(keys)},
        numeric_scalers=scales,
        columns=[
            dict(name=k, zero_default=bool(z), training_patient_support=int(n))
            for k, z, n in columns
        ],
        scale_policy="per code/unit median/IQR; zero IQR uses population std then 1; asinh; no clipping",
        time_policy="date_only flag; midnight is sorting convention; anchor demographics are metadata",
        value_policy="conflicts masked; unknown numeric scaler gives zero plus scale_unknown; text values kept",
    )


def transform_tabular(con, protocol, ids):
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("nonempty unique transform IDs required")
    columns = protocol["columns"]
    x = np.empty((len(ids), len(columns)), dtype=np.float32)
    for j, col in enumerate(columns):
        x[:, j] = 0.0 if col["zero_default"] else np.nan
    patients = {p: i for i, p in enumerate(ids)}
    lookup = {c["name"]: j for j, c in enumerate(columns)}
    con.execute("CREATE OR REPLACE TEMP TABLE tab_ids(patient_id VARCHAR PRIMARY KEY)")
    con.executemany("INSERT INTO tab_ids VALUES (?)", [(p,) for p in ids])
    if con.execute("SELECT count(*) FROM people JOIN tab_ids USING(patient_id)").fetchone()[
        0
    ] != len(ids):
        raise ValueError("transform IDs outside prepared patients")
    for p, k, v in con.execute("""SELECT patient_id,feature,value FROM feature_rows
        JOIN tab_ids USING(patient_id)""").fetchall():
        j = lookup.get(k)
        if j is not None:
            x[patients[p], j] = np.nan if v is None else v
    if np.isinf(x).any():
        raise ValueError("nonfinite float32 additive feature")
    return x
