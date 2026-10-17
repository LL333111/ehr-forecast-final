"""Measurement context, visit interactions and time-aware features."""

from __future__ import annotations

import json

import numpy as np


def bind_clinical(con, ids, database):
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Unique nonempty patient IDs required")
    con.execute("SET TimeZone='UTC'")
    con.execute("ATTACH '" + str(database).replace("'", "''") + "' AS db (READ_ONLY)")
    for table, needed in [
        ("patients", {"patient_id", "anchor_date", "birth_date"}),
        (
            "events",
            {
                "patient_id",
                "event_date",
                "source_table",
                "code",
                "multiplicity",
                "reason_code",
                "reason_description",
                "base_cost",
                "observed_end",
            },
        ),
    ]:
        columns = {r[0] for r in con.execute("DESCRIBE db.clinical." + table).fetchall()}
        if not needed <= columns:
            raise ValueError("Changed clinical schema")
    con.execute("CREATE SCHEMA research")
    con.execute("CREATE TABLE research.membership(patient_id VARCHAR PRIMARY KEY,role VARCHAR)")
    con.executemany("INSERT INTO research.membership VALUES (?,?)", [(p, "requested") for p in ids])
    con.execute(
        "CREATE VIEW research.patients AS SELECT p.* FROM db.clinical.patients p JOIN research.membership USING(patient_id)"
    )
    con.execute(
        "CREATE VIEW research.events AS SELECT e.*,m.role FROM db.clinical.events e JOIN research.membership m USING(patient_id)"
    )
    con.execute("""CREATE TEMP VIEW history AS SELECT e.*,p.anchor_date,
        date_diff('day',event_date,anchor_date) days_before FROM research.events e
        JOIN research.patients p USING(patient_id)""")
    if con.execute("SELECT count(*) FROM research.patients").fetchone()[0] != len(ids):
        raise ValueError("Missing patient")
    check_history(con)


def check_history(con):
    bad = con.execute("""SELECT count(*) FROM history WHERE event_date IS NULL OR days_before<=0
      OR observed_end>=anchor_date OR observed_end<event_date
      OR base_cost IS NOT NULL OR nullif(reason_code,'') IS NOT NULL
      OR nullif(reason_description,'') IS NOT NULL
      OR (source_table='observations' AND (upper(trim(coalesce(code,''))) IN ('QALY','DALY','QOLS')
          OR upper(trim(coalesce(description,''))) IN ('QALY','DALY','QOLS')))""").fetchone()[0]
    if bad:
        raise ValueError("Unqualified records entered features")


def prepare_extra(con):
    """Long feature representation; no population-fitted preprocessing."""
    check_history(con)
    con.execute(
        "CREATE OR REPLACE TEMP TABLE v18_rows(patient_id VARCHAR,feature VARCHAR,value DOUBLE,zero_default BOOLEAN)"
    )
    con.execute("""INSERT INTO v18_rows WITH s AS (
      SELECT patient_id,source_table,code,
        sum(multiplicity) FILTER(WHERE days_before<=30) n30,
        sum(multiplicity) FILTER(WHERE days_before<=90) n90,
        sum(multiplicity) FILTER(WHERE days_before<=365) n365,
        sum(multiplicity*exp(-ln(2)*days_before/90.)) decay90,
        sum(multiplicity*exp(-ln(2)*days_before/365.)) decay365
      FROM history WHERE nullif(code,'') IS NOT NULL GROUP BY 1,2,3)
      SELECT patient_id,CAST(to_json(['v18','code_recent',source_table,code,stat]) AS VARCHAR),val,true
      FROM s,LATERAL(VALUES ('count_d30',coalesce(n30,0)),('count_d90',coalesce(n90,0)),
        ('decay_d90',decay90),('decay_y1',decay365),
        ('recent_share',coalesce(n90,0)/greatest(coalesce(n365,0),1))) f(stat,val)""")
    con.execute("""INSERT INTO v18_rows WITH s AS (
      SELECT patient_id,coalesce(bodysite_code,'') body,coalesce(modality_code,'') modality,
        coalesce(sop_code,'') sop,count(DISTINCT encounter_id) visits,min(days_before) recency,
        count(DISTINCT event_date) dates,sum(multiplicity*exp(-ln(2)*days_before/365.)) decayed
      FROM history WHERE source_table='imaging_studies' GROUP BY 1,2,3,4)
      SELECT patient_id,CAST(to_json(['v18','imaging_exam',body,modality,sop,stat]) AS VARCHAR),val,z
      FROM s,LATERAL(VALUES ('visits',visits,true),('dates',dates,true),
        ('recency',recency,false),('decay_y1',decayed,true)) f(stat,val,z)""")
    con.execute("""CREATE OR REPLACE TEMP TABLE v18_measured AS
      SELECT patient_id,code,coalesce(description,'') description,coalesce(units,'') units,
        coalesce(event_ts,CAST(event_date AS TIMESTAMP)) stamp,min(days_before) days_before,
        CASE WHEN count(DISTINCT numeric_value)=1 AND NOT bool_or(numeric_conflict)
             THEN min(numeric_value) ELSE NULL END AS value
      FROM history WHERE source_table='observations' AND lower(obs_type)='numeric'
      GROUP BY 1,2,3,4,5""")
    con.execute("""INSERT INTO v18_rows WITH ranked AS (
      SELECT *,row_number() OVER(PARTITION BY patient_id,code,description,units ORDER BY stamp DESC) rn
      FROM v18_measured WHERE value IS NOT NULL), s AS (
      SELECT patient_id,code,description,units,count(*) n,
        max(value) FILTER(WHERE rn=1) latest,
        median(value) FILTER(WHERE rn>1) previous_median,
        avg(value) FILTER(WHERE days_before<=365) recent_mean,
        avg(value) FILTER(WHERE days_before>365 AND days_before<=1826) previous_mean,
        quantile_cont(value,.75)-quantile_cont(value,.25) iqr
      FROM ranked GROUP BY 1,2,3,4)
      SELECT patient_id,CAST(to_json(['v18','measurement_context',code,description,units,stat]) AS VARCHAR),val,false
      FROM s,LATERAL(VALUES ('latest',latest),('latest_minus_previous_median',latest-previous_median),
        ('recent_minus_previous_mean',recent_mean-previous_mean),
        ('latest_relative_deviation',(latest-previous_median)/(abs(previous_median)+greatest(iqr,1e-6)))) f(stat,val)""")
    con.execute("""CREATE OR REPLACE TEMP TABLE v18_states AS
      SELECT patient_id,code,coalesce(description,'') description,
        coalesce(event_ts,CAST(event_date AS TIMESTAMP)) stamp,min(days_before) days_before,
        min(value_raw) AS value,count(DISTINCT value_raw) variants
      FROM history WHERE source_table='observations' AND lower(coalesce(obs_type,''))<>'numeric'
        AND code<>'29303009' AND nullif(value_raw,'') IS NOT NULL AND length(value_raw)<512
      GROUP BY 1,2,3,4""")
    con.execute("""INSERT INTO v18_rows WITH r AS (
      SELECT *,row_number() OVER(PARTITION BY patient_id,code,description ORDER BY stamp DESC) rn,
        lag(value) OVER(PARTITION BY patient_id,code,description ORDER BY stamp) prev,
        lag(variants) OVER(PARTITION BY patient_id,code,description ORDER BY stamp) prev_variants
      FROM v18_states)
      SELECT patient_id,CAST(to_json(['v18','named_state',code,description,value,'latest']) AS VARCHAR),1,true
        FROM r WHERE rn=1 AND variants=1
      UNION ALL
      SELECT patient_id,CAST(to_json(['v18','named_state_transition',code,description,prev,value]) AS VARCHAR),count(*),true
        FROM r WHERE variants=1 AND prev_variants=1 AND value<>prev GROUP BY 1,2""")
    con.execute("""CREATE OR REPLACE TEMP TABLE v18_visit_codes AS
      SELECT patient_id,encounter_id,source_table,code,min(days_before) days_before
      FROM history WHERE source_table IN ('conditions','procedures','medications')
        AND nullif(encounter_id,'') IS NOT NULL AND nullif(code,'') IS NOT NULL GROUP BY 1,2,3,4""")
    con.execute("""INSERT INTO v18_rows WITH pairs AS (
      SELECT a.patient_id,a.encounter_id,
        md5_number_lower(CAST(to_json([a.source_table,a.code,b.source_table,b.code]) AS VARCHAR))%128 bucket,
        greatest(a.days_before,b.days_before) days_before
      FROM v18_visit_codes a JOIN v18_visit_codes b ON a.patient_id=b.patient_id
        AND a.encounter_id=b.encounter_id AND a.source_table<b.source_table)
      SELECT patient_id,CAST(to_json(['v18','same_visit_code_pair',CAST(bucket AS VARCHAR),'full']) AS VARCHAR),count(*),true
        FROM pairs GROUP BY 1,2
      UNION ALL
      SELECT patient_id,CAST(to_json(['v18','same_visit_code_pair',CAST(bucket AS VARCHAR),'y1']) AS VARCHAR),count(*),true
        FROM pairs WHERE days_before<=365 GROUP BY 1,2""")
    con.execute("""INSERT INTO v18_rows WITH exams AS (
      SELECT patient_id,event_date,encounter_id,min(days_before) days_before
      FROM history WHERE source_table='observations' AND code='29303009' GROUP BY 1,2,3), s AS (
      SELECT patient_id,count(*) exams,min(days_before) recency,
        sum(exp(-ln(2)*days_before/365.)) decayed FROM exams GROUP BY 1)
      SELECT patient_id,CAST(to_json(['v18','ecg_exam_fact',stat]) AS VARCHAR),val,z
      FROM s,LATERAL(VALUES('exams',exams,true),('recency',recency,false),('decay_y1',decayed,true)) f(stat,val,z)""")
    con.execute("UPDATE v18_rows SET value=NULL WHERE NOT isfinite(value)")
    if con.execute(
        "SELECT count(*) FROM (SELECT patient_id,feature,count(*) n FROM v18_rows GROUP BY 1,2 HAVING n>1)"
    ).fetchone()[0]:
        raise ValueError("Duplicate feature key")


def fit_extra(con, train_ids, min_support=20):
    if not train_ids or len(train_ids) != len(set(train_ids)) or min_support < 1:
        raise ValueError("Invalid training membership/support")
    con.execute("CREATE OR REPLACE TEMP TABLE extra_fit(patient_id VARCHAR PRIMARY KEY)")
    con.executemany("INSERT INTO extra_fit VALUES (?)", [(p,) for p in train_ids])
    rows = con.execute(
        """SELECT feature,any_value(zero_default),count(DISTINCT patient_id)
        FROM v18_rows JOIN extra_fit USING(patient_id) WHERE value IS NOT NULL
        GROUP BY 1 HAVING count(DISTINCT patient_id)>=? ORDER BY 1""",
        [min_support],
    ).fetchall()
    return [
        dict(name=k, zero_default=bool(z), support=int(n), group=json.loads(k)[1], family="v18")
        for k, z, n in rows
    ]


def transform_extra(con, schema, ids):
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Invalid transform membership")
    con.execute("CREATE OR REPLACE TEMP TABLE extra_score(patient_id VARCHAR PRIMARY KEY,pos INT)")
    con.executemany("INSERT INTO extra_score VALUES (?,?)", [(p, i) for i, p in enumerate(ids)])
    x = np.full((len(ids), len(schema)), np.nan, dtype=np.float32)
    index = {r["name"]: i for i, r in enumerate(schema)}
    if len(index) != len(schema):
        raise ValueError("Duplicate schema")
    for j, r in enumerate(schema):
        if r["zero_default"]:
            x[:, j] = 0.0
    for i, k, v in con.execute(
        "SELECT pos,feature,value FROM v18_rows JOIN extra_score USING(patient_id)"
    ).fetchall():
        if k in index:
            x[i, index[k]] = np.nan if v is None else v
    if np.isinf(x).any():
        raise ValueError("Nonfinite feature")
    return x
