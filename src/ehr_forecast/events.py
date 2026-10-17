"""Official-split-preserving event preparation."""

from __future__ import annotations

from .event_schema import EVENT_COLUMNS


def prepare(con, ids: list[str]) -> dict:
    if not ids or len(ids) != len(set(ids)) or any((not isinstance(p, str) or not p for p in ids)):
        raise ValueError("unique nonempty patient IDs required")
    for view, required in [
        ("e", set(EVENT_COLUMNS)),
        ("a", {"patient_id", "split", "anchor_date", "birth_date", "gender"}),
    ]:
        if not required <= {r[0] for r in con.execute(f"DESCRIBE {view}").fetchall()}:
            raise ValueError(f"missing input schema in {view}")
    con.execute("CREATE OR REPLACE TEMP TABLE requested(patient_id VARCHAR PRIMARY KEY)")
    con.executemany("INSERT INTO requested VALUES (?)", [(p,) for p in ids])
    con.execute("""CREATE OR REPLACE TEMP TABLE people AS
        SELECT a.patient_id,try_cast(a.anchor_date AS DATE) anchor_date,
          try_cast(a.birth_date AS DATE) birth_date,CAST(a.gender AS VARCHAR) gender,a.split AS official_split FROM a
        JOIN requested USING(patient_id) WHERE a.split IN ('train','val','test')""")
    counts = con.execute("""SELECT count(*),count(DISTINCT patient_id),
        count(*) FILTER(WHERE anchor_date IS NULL OR birth_date>anchor_date) FROM people""").fetchone()
    if counts != (len(ids), len(ids), 0):
        raise ValueError("missing, duplicate, or invalid official anchors")
    columns = ",".join(("e." + name for name in EVENT_COLUMNS))
    con.execute(
        f"CREATE OR REPLACE TEMP VIEW selected AS SELECT {columns}\n        FROM e JOIN requested USING(patient_id) JOIN people USING(patient_id) WHERE e.split=people.official_split"
    )
    if con.execute(
        "SELECT count(*) FROM e JOIN requested USING(patient_id) JOIN people USING(patient_id) WHERE e.split IS DISTINCT FROM official_split"
    ).fetchone()[0]:
        raise ValueError("Event partition disagrees with official split")
    violations = con.execute("""SELECT count(*) FROM selected JOIN people USING(patient_id)
        WHERE event_date IS NULL OR event_date>=anchor_date OR event_date<birth_date
        OR (event_ts IS NOT NULL AND (CAST(event_ts AS DATE)<>event_date
             OR event_ts>=CAST(anchor_date AS TIMESTAMP)))""").fetchone()[0]
    if violations:
        raise ValueError(f"{violations} unsafe event dates/timestamps; refusing to prepare")
    con.execute("""CREATE OR REPLACE TEMP TABLE compact AS
        SELECT *,count(*)::BIGINT AS multiplicity FROM selected GROUP BY ALL""")
    con.execute("""CREATE OR REPLACE TEMP TABLE conflicts AS
        SELECT patient_id,code,coalesce(units,'') AS units,event_ts
        FROM compact WHERE source_table='observations' AND lower(obs_type)='numeric'
          AND event_ts IS NOT NULL AND isfinite(try_cast(value_raw AS DOUBLE))
        GROUP BY 1,2,3,4 HAVING count(DISTINCT try_cast(value_raw AS DOUBLE))>1""")
    con.execute("""CREATE OR REPLACE TEMP TABLE clean AS
        SELECT c.*,
          (c.source_table='observations' AND lower(coalesce(c.obs_type,''))='numeric') AS is_numeric,
          (c.event_ts IS NULL) AS date_only,
          (f.patient_id IS NOT NULL) AS numeric_conflict,
          (c.source_table='observations' AND lower(coalesce(c.obs_type,''))='numeric'
           AND c.value_raw IS NOT NULL AND trim(c.value_raw)<>''
           AND NOT coalesce(isfinite(try_cast(c.value_raw AS DOUBLE)),false)) AS invalid_numeric,
          CASE WHEN c.source_table='observations' AND lower(coalesce(c.obs_type,''))='numeric'
             AND f.patient_id IS NULL AND isfinite(try_cast(c.value_raw AS DOUBLE))
             THEN try_cast(c.value_raw AS DOUBLE) ELSE NULL END AS numeric_value
        FROM compact c LEFT JOIN conflicts f ON c.patient_id=f.patient_id
          AND c.code=f.code AND coalesce(c.units,'')=f.units AND c.event_ts=f.event_ts""")
    con.execute("""CREATE OR REPLACE TEMP TABLE measurements AS
        SELECT patient_id,code,coalesce(units,'') AS units,event_ts,
               min(numeric_value) AS value
        FROM clean WHERE is_numeric AND numeric_value IS NOT NULL AND event_ts IS NOT NULL
        GROUP BY 1,2,3,4""")
    con.execute("""CREATE OR REPLACE TEMP TABLE tokens AS
        WITH core AS (
          SELECT patient_id,coalesce(event_ts,CAST(event_date AS TIMESTAMP)) AS stamp,
            source_table AS family,
            CAST(to_json([source_table,coalesce(code,''),
                CASE WHEN is_numeric THEN coalesce(units,'') ELSE '' END]) AS VARCHAR) AS token_key,
            CASE WHEN is_numeric THEN CAST(to_json([code,coalesce(units,'')]) AS VARCHAR) ELSE NULL END AS lab_key,
            numeric_value,is_numeric,invalid_numeric,numeric_conflict,date_only,multiplicity,
            coalesce(encounter_id,'') AS tie_encounter,coalesce(value_raw,'') AS tie_value,
            CAST(to_json([source_file,event_time_raw,modality_code,bodysite_code,sop_code,reason_code]) AS VARCHAR) AS tie_other,
            false AS is_anchor FROM clean
          UNION ALL
          SELECT patient_id,coalesce(event_ts,CAST(event_date AS TIMESTAMP)),family,
            CAST(to_json([family,extra]) AS VARCHAR),NULL,NULL,false,false,false,date_only,multiplicity,
            coalesce(encounter_id,''),coalesce(value_raw,''),
            CAST(to_json([source_file,event_time_raw,code,units,obs_type,modality_code,bodysite_code,sop_code,reason_code]) AS VARCHAR),false
          FROM clean, LATERAL (VALUES
            (source_table||'_reason',reason_code),('imaging_modality',modality_code),
            ('imaging_bodysite',bodysite_code),('imaging_sop',sop_code),
            ('observation_text',CASE WHEN source_table='observations' AND NOT is_numeric
               THEN CAST(to_json([code,value_raw]) AS VARCHAR) ELSE NULL END)
          ) AS extras(family,extra) WHERE extra IS NOT NULL AND extra<>''
          UNION ALL
          SELECT patient_id,CAST(anchor_date AS TIMESTAMP),'demographics',
            CAST(to_json(['demographics','gender',coalesce(gender,'<missing>')]) AS VARCHAR),
            NULL,NULL,false,false,false,false,1,'','','',true FROM people
        )
        SELECT core.*,p.anchor_date,p.birth_date,
          CASE WHEN birth_date IS NULL THEN NULL
            ELSE date_diff('day',birth_date,CAST(stamp AS DATE))/365.2425 END AS age_years,
          date_diff('day',CAST(stamp AS DATE),anchor_date) AS days_before,
          (NOT date_only AND NOT coalesce(lag(date_only) OVER(PARTITION BY patient_id
            ORDER BY stamp,family,token_key,tie_encounter,tie_value,tie_other),true)) AS delta_exact,
          (epoch(stamp)-epoch(lag(stamp) OVER(PARTITION BY patient_id
            ORDER BY stamp,family,token_key,tie_encounter,tie_value,tie_other)))/86400. AS delta_days
        FROM core JOIN people p USING(patient_id)""")
    raw, compact = [
        con.execute(f"SELECT count(*) FROM {v}").fetchone()[0] for v in ("selected", "clean")
    ]
    return dict(
        patients=len(ids),
        input_events=raw,
        compact_events=compact,
        projection_identical_extra_rows=raw - compact,
        multiplicity_sum=con.execute("SELECT coalesce(sum(multiplicity),0) FROM clean").fetchone()[
            0
        ],
        numeric_conflict_groups=con.execute("SELECT count(*) FROM conflicts").fetchone()[0],
        flagged_numeric_rows=con.execute(
            "SELECT count(*) FROM clean WHERE numeric_conflict"
        ).fetchone()[0],
        invalid_numeric_rows=con.execute(
            "SELECT count(*) FROM clean WHERE invalid_numeric"
        ).fetchone()[0],
        tokens=con.execute("SELECT count(*) FROM tokens").fetchone()[0],
        policy="compact safe-field-identical rows with multiplicity; mask conflicting numeric values; retain all patients and history",
    )
