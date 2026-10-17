"""Training-fitted dictionaries and historical EHR summaries."""

from __future__ import annotations

import csv
from datetime import date, datetime
from pathlib import Path

import duckdb
import numpy as np

from .schema import EVENT_HEADERS, EVENT_SPECS, AuditError, date_expr, sql_ident

WINDOWS = ("d90", "y1", "y5", "full")
WINDOW_SQL = {
    "d90": "m.event_date >= a.anchor_date - INTERVAL 90 DAY",
    "y1": "m.event_date >= a.anchor_date - INTERVAL 1 YEAR",
    "y5": "m.event_date >= a.anchor_date - INTERVAL 5 YEAR",
    "full": "TRUE",
}
CODE_STATS = ("ind", "count", "n_enc", "days_since")
LAB_STATS = ("latest", "previous", "mean", "min", "max", "count", "days_since", "delta", "missing")


def column_index(schema: list[dict[str, str]], **wanted: str) -> int:
    for index, column in enumerate(schema):
        if all((column.get(key) == value for key, value in wanted.items())):
            return index
    raise KeyError(wanted)


def _varchar(header: list[str], column: str, alias: str = "e") -> str:
    if column not in header:
        return "CAST(NULL AS VARCHAR)"
    return f"nullif(trim({alias}.{sql_ident(column)}), '')"


def _partition_counts(
    con: duckdb.DuckDBPyConnection, view: str, typed_sql: str, allowed_sql: str
) -> tuple[int, ...]:
    row = con.execute(
        f"\n        WITH typed AS (\n            {typed_sql}\n        ),\n        joined AS (\n            SELECT t.*, a.anchor_date\n            FROM typed t\n            LEFT JOIN anchors a\n                ON t.patient_id = a.patient_id\n               AND a.split IN ({allowed_sql})\n        )\n        SELECT\n            (SELECT count(*) FROM {sql_ident(view)}) AS n_raw,\n            count(*) FILTER (WHERE patient_id IS NULL) AS n_missing_patient,\n            count(*) FILTER (\n                WHERE patient_id IS NOT NULL AND event_date IS NULL\n            ) AS n_missing_time,\n            count(*) FILTER (\n                WHERE patient_id IS NOT NULL AND event_date IS NOT NULL AND anchor_date IS NULL\n            ) AS n_no_anchor,\n            count(*) FILTER (\n                WHERE patient_id IS NOT NULL AND event_date IS NOT NULL\n                  AND anchor_date IS NOT NULL AND event_date >= anchor_date\n            ) AS n_post,\n            count(*) FILTER (\n                WHERE patient_id IS NOT NULL AND event_date IS NOT NULL\n                  AND anchor_date IS NOT NULL AND event_date < anchor_date\n            ) AS n_pre\n        FROM joined\n        "
    ).fetchone()
    if row is None:
        raise AuditError(f"no reconciliation for {view}")
    counts = tuple((int(value) for value in row))
    if counts[0] != sum(counts[1:]):
        raise AuditError(f"event rows were lost for {view}: {counts}")
    return counts


def _typed_select(table: str) -> str:
    header = EVENT_HEADERS[table]
    spec = EVENT_SPECS[table]
    patient = f"nullif(trim(e.{sql_ident(str(spec['patient_key']))}), '')"
    time_column = str(spec["time_col"])
    event_date = date_expr("e", time_column)
    time_raw = f"nullif(trim(e.{sql_ident(time_column)}), '')"
    event_ts = (
        "CASE WHEN "
        + time_raw
        + " IS NULL OR length("
        + time_raw
        + ") < 19 THEN NULL ELSE try_strptime(replace(replace("
        + time_raw
        + ", 'T', ' '), 'Z', ''), '%Y-%m-%d %H:%M:%S') END"
    )
    if table == "encounters":
        encounter = _varchar(header, "Id")
    else:
        encounter = _varchar(header, "ENCOUNTER")
    return f"\n        SELECT\n            {patient} AS patient_id,\n            {event_date} AS event_date,\n            {time_raw} AS event_time_raw,\n            {event_ts} AS event_ts,\n            {_varchar(header, 'CODE')} AS code,\n            {encounter} AS encounter_id,\n            {_varchar(header, 'VALUE')} AS value_raw,\n            {_varchar(header, 'UNITS')} AS units,\n            {_varchar(header, 'TYPE')} AS obs_type,\n            {_varchar(header, 'MODALITY_CODE')} AS modality_code,\n            {_varchar(header, 'BODYSITE_CODE')} AS bodysite_code,\n            {_varchar(header, 'SOP_CODE')} AS sop_code,\n            {_varchar(header, 'REASONCODE')} AS reason_code\n        FROM {sql_ident(f'src_{table}')} e\n    "


def _collect_events(con: duckdb.DuckDBPyConnection, raw: Path) -> list[dict[str, object]]:
    con.execute("""
        CREATE TEMP TABLE safe_events (
            patient_id VARCHAR,
            split VARCHAR,
            source_table VARCHAR,
            source_file VARCHAR,
            event_date DATE,
            event_time_raw VARCHAR,
            event_ts TIMESTAMP,
            code VARCHAR,
            encounter_id VARCHAR,
            value_raw VARCHAR,
            units VARCHAR,
            obs_type VARCHAR,
            modality_code VARCHAR,
            bodysite_code VARCHAR,
            sop_code VARCHAR,
            reason_code VARCHAR
        )
        """)
    reconciliation: list[dict[str, object]] = []
    for source, allowed in (("train_val", ("train", "val")), ("test", ("test",))):
        allowed_sql = ", ".join((f"'{split}'" for split in allowed))
        for table in EVENT_SPECS:
            path = raw / source / f"{table}.csv"
            view = f"src_{source}_{table}"
            from .schema import _register_csv

            _register_csv(con, view, path)
            typed = _typed_select(table).replace(
                f"FROM {sql_ident(f'src_{table}')} e", f"FROM {sql_ident(view)} e"
            )
            counts = _partition_counts(con, view, typed, allowed_sql)
            source_file = f"{source}/{table}.csv"
            con.execute(
                f"\n                INSERT INTO safe_events\n                WITH typed AS (\n                    {typed}\n                )\n                SELECT\n                    t.patient_id,\n                    a.split,\n                    '{table}',\n                    '{source_file}',\n                    t.event_date,\n                    t.event_time_raw,\n                    t.event_ts,\n                    t.code,\n                    t.encounter_id,\n                    t.value_raw,\n                    t.units,\n                    t.obs_type,\n                    t.modality_code,\n                    t.bodysite_code,\n                    t.sop_code,\n                    t.reason_code\n                FROM typed t\n                JOIN anchors a\n                    ON t.patient_id = a.patient_id\n                   AND a.split IN ({allowed_sql})\n                WHERE t.patient_id IS NOT NULL\n                  AND t.event_date IS NOT NULL\n                  AND a.anchor_date IS NOT NULL\n                  AND t.event_date < a.anchor_date\n                "
            )
            inserted = int(
                con.execute(
                    "SELECT count(*) FROM safe_events WHERE source_file = ?", [source_file]
                ).fetchone()[0]
            )
            if inserted != counts[5]:
                raise AuditError(
                    f"safe event insert dropped rows for {source_file}: {inserted} vs {counts[5]}"
                )
            reconciliation.append(
                {
                    "source": source,
                    "table": table,
                    "n_raw": counts[0],
                    "n_missing_patient": counts[1],
                    "n_missing_time": counts[2],
                    "n_no_anchor": counts[3],
                    "n_post": counts[4],
                    "n_pre": counts[5],
                }
            )
            print(f"events {source_file}: pre={counts[5]} raw={counts[0]}", flush=True)
    con.execute("""
        CREATE TEMP VIEW code_mentions AS
        SELECT patient_id, split, event_date, encounter_id, source_table AS family, code
        FROM safe_events
        WHERE code IS NOT NULL
        UNION ALL
        SELECT patient_id, split, event_date, encounter_id, source_table || '_reason', reason_code
        FROM safe_events
        WHERE reason_code IS NOT NULL
        UNION ALL
        SELECT patient_id, split, event_date, encounter_id, 'imaging_modality', modality_code
        FROM safe_events
        WHERE modality_code IS NOT NULL
        UNION ALL
        SELECT patient_id, split, event_date, encounter_id, 'imaging_bodysite', bodysite_code
        FROM safe_events
        WHERE bodysite_code IS NOT NULL
        UNION ALL
        SELECT patient_id, split, event_date, encounter_id, 'imaging_sop', sop_code
        FROM safe_events
        WHERE sop_code IS NOT NULL
        """)
    return reconciliation


def _as_date(value: object) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    raise TypeError(type(value))


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(
            handle,
            lineterminator="""
""",
        )
        writer.writerow(header)
        writer.writerows(rows)


def _cohort_tables(
    con: duckdb.DuckDBPyConnection, raw: Path, processed: Path
) -> dict[str, list[str]]:
    con.execute(
        f"\n        CREATE TEMP TABLE cohort AS\n        SELECT\n            a.patient_id,\n            a.split,\n            a.anchor_date,\n            a.last_encounter_date,\n            a.label_end_date,\n            d.birth_date,\n            d.gender\n        FROM anchors a\n        LEFT JOIN (\n            SELECT 'train_val' AS source,\n                   trim({sql_ident('Id')}) AS patient_id,\n                   {date_expr('p', 'BIRTHDATE')} AS birth_date,\n                   nullif(trim(p.{sql_ident('GENDER')}), '') AS gender\n            FROM patients_train_val p\n            UNION ALL\n            SELECT 'test',\n                   trim({sql_ident('Id')}),\n                   {date_expr('p', 'BIRTHDATE')},\n                   nullif(trim(p.{sql_ident('GENDER')}), '')\n            FROM patients_test p\n        ) d\n            ON d.patient_id = a.patient_id\n           AND (\n                (a.split IN ('train', 'val') AND d.source = 'train_val')\n                OR (a.split = 'test' AND d.source = 'test')\n           )\n        "
    )
    anchor_rows = con.execute("""
        SELECT patient_id, split, anchor_date, last_encounter_date, label_end_date, birth_date, gender
        FROM cohort
        ORDER BY CASE split WHEN 'train' THEN 1 WHEN 'val' THEN 2 ELSE 3 END, patient_id
        """).fetchall()
    _write_csv(
        processed / "anchors.csv",
        [
            "patient_id",
            "split",
            "anchor_date",
            "last_encounter_date",
            "label_end_date",
            "birth_date",
            "gender",
        ],
        [
            [
                row[0],
                row[1],
                _iso(row[2]),
                _iso(row[3]),
                _iso(row[4]),
                _iso(row[5]),
                "" if row[6] is None else row[6],
            ]
            for row in anchor_rows
        ],
    )
    patients: dict[str, list[str]] = {split: [] for split in ("train", "val", "test")}
    anchors: dict[str, tuple[date, date]] = {}
    demo: dict[str, tuple[date | None, str | None]] = {}
    for row in anchor_rows:
        patient_id, split, anchor, _last, label_end, birth, gender = row
        anchor_d = _as_date(anchor)
        label_d = _as_date(label_end)
        if anchor_d is None or label_d is None:
            continue
        patients[split].append(patient_id)
        anchors[patient_id] = (anchor_d, label_d)
        demo[patient_id] = (_as_date(birth), None if gender is None else str(gender))
    con.execute("CREATE TEMP TABLE kept_patients (patient_id VARCHAR, split VARCHAR)")
    con.executemany(
        "INSERT INTO kept_patients VALUES (?, ?)",
        [(patient_id, split) for split, ids in patients.items() for patient_id in ids],
    )
    return {"patients": patients, "anchors": anchors, "demo": demo}


def _iso(value: object) -> str:
    parsed = _as_date(value) if value not in (None, "") else None
    return "" if parsed is None else parsed.isoformat()


def _feature_schema(con: duckdb.DuckDBPyConnection) -> list[dict[str, str]]:
    genders = [
        row[0]
        for row in con.execute("""
            SELECT DISTINCT gender
            FROM cohort
            WHERE split = 'train' AND gender IS NOT NULL
            ORDER BY 1
            """).fetchall()
    ]
    schema: list[dict[str, str]] = [
        {
            "name": "demo|age_years",
            "family": "demographic",
            "key": "age",
            "unit": "",
            "value": "",
            "window": "anchor",
            "stat": "age_years",
            "default": "nan",
        },
        {
            "name": "demo|age_missing",
            "family": "demographic",
            "key": "age",
            "unit": "",
            "value": "",
            "window": "anchor",
            "stat": "age_missing",
            "default": "zero",
        },
        {
            "name": "demo|gender_unknown",
            "family": "demographic",
            "key": "gender",
            "unit": "",
            "value": "",
            "window": "anchor",
            "stat": "gender_unknown",
            "default": "zero",
        },
    ]
    for gender in genders:
        schema.append(
            {
                "name": f"demo|gender|{gender}",
                "family": "demographic",
                "key": "gender",
                "unit": "",
                "value": gender,
                "window": "anchor",
                "stat": "gender",
                "default": "zero",
            }
        )
    for window in WINDOWS:
        for stat, default in (
            ("n_events", "zero"),
            ("n_encounters", "zero"),
            ("history_days", "nan"),
        ):
            schema.append(
                {
                    "name": f"history|{stat}|{window}",
                    "family": "history",
                    "key": stat,
                    "unit": "",
                    "value": "",
                    "window": window,
                    "stat": stat,
                    "default": default,
                }
            )
    code_keys = con.execute("""
        SELECT family, code
        FROM (
            SELECT family, code FROM code_mentions WHERE split = 'train'
        )
        GROUP BY 1, 2
        ORDER BY 1, 2
        """).fetchall()
    for family, code in code_keys:
        for window in WINDOWS:
            for stat in CODE_STATS:
                default = "nan" if stat == "days_since" else "zero"
                schema.append(
                    {
                        "name": f"code|{family}|{code}|{window}|{stat}",
                        "family": str(family),
                        "key": str(code),
                        "unit": "",
                        "value": "",
                        "window": window,
                        "stat": stat,
                        "default": default,
                    }
                )
    lab_keys = con.execute("""
        SELECT code, coalesce(units, '') AS units
        FROM safe_events
        WHERE split = 'train'
          AND source_table = 'observations'
          AND lower(coalesce(obs_type, '')) = 'numeric'
          AND try_cast(value_raw AS DOUBLE) IS NOT NULL
        GROUP BY 1, 2
        ORDER BY 1, 2
        """).fetchall()
    for code, units in lab_keys:
        for window in WINDOWS:
            for stat in LAB_STATS:
                if stat == "missing":
                    default = "one"
                elif stat == "count":
                    default = "zero"
                else:
                    default = "nan"
                schema.append(
                    {
                        "name": f"lab|{code}|{units}|{window}|{stat}",
                        "family": "observation_numeric",
                        "key": str(code),
                        "unit": str(units),
                        "value": "",
                        "window": window,
                        "stat": stat,
                        "default": default,
                    }
                )
    text_keys = con.execute("""
        SELECT code, value_raw
        FROM safe_events
        WHERE split = 'train'
          AND source_table = 'observations'
          AND lower(coalesce(obs_type, '')) <> 'numeric'
          AND code IS NOT NULL
          AND value_raw IS NOT NULL
        GROUP BY 1, 2
        ORDER BY 1, 2
        """).fetchall()
    for code, value in text_keys:
        for window in WINDOWS:
            schema.append(
                {
                    "name": f"text|{code}|{window}|{len(schema)}",
                    "family": "observation_text",
                    "key": str(code),
                    "unit": "",
                    "value": str(value),
                    "window": window,
                    "stat": "count",
                    "default": "zero",
                }
            )
    return schema


def _blank_matrix(n_rows: int, schema: list[dict[str, str]]) -> np.ndarray:
    matrix = np.full((n_rows, len(schema)), np.nan, dtype=np.float32)
    for index, column in enumerate(schema):
        if column["default"] == "zero":
            matrix[:, index] = 0
        elif column["default"] == "one":
            matrix[:, index] = 1
    return matrix


def _fill_demographics(
    matrix: np.ndarray,
    schema: list[dict[str, str]],
    patient_ids: list[str],
    demo: dict[str, tuple[date | None, str | None]],
    anchors: dict[str, tuple[date, date]],
) -> None:
    age_col = column_index(schema, family="demographic", stat="age_years")
    missing_col = column_index(schema, family="demographic", stat="age_missing")
    unknown_col = column_index(schema, family="demographic", stat="gender_unknown")
    gender_cols = {
        column["value"]: index
        for index, column in enumerate(schema)
        if column["stat"] == "gender" and column["family"] == "demographic"
    }
    for row_index, patient_id in enumerate(patient_ids):
        birth, gender = demo[patient_id]
        anchor, _label_end = anchors[patient_id]
        if birth is None:
            matrix[row_index, missing_col] = 1
        else:
            matrix[row_index, age_col] = np.float32((anchor - birth).days / 365.2425)
        if gender in gender_cols:
            matrix[row_index, gender_cols[gender]] = 1
        else:
            matrix[row_index, unknown_col] = 1


def _fill_history(
    con: duckdb.DuckDBPyConnection,
    matrix: np.ndarray,
    schema: list[dict[str, str]],
    patient_index: dict[str, int],
    split: str,
) -> None:
    for window, predicate in WINDOW_SQL.items():
        rows = con.execute(
            f"\n            SELECT m.patient_id,\n                   count(*) AS n_events,\n                   count(*) FILTER (WHERE m.source_table = 'encounters') AS n_encounters,\n                   date_diff('day', min(m.event_date), max(m.event_date)) AS history_days\n            FROM safe_events m\n            JOIN anchors a ON m.patient_id = a.patient_id\n            WHERE m.split = ? AND {predicate}\n            GROUP BY 1\n            ",
            [split],
        ).fetchall()
        event_col = column_index(schema, family="history", stat="n_events", window=window)
        enc_col = column_index(schema, family="history", stat="n_encounters", window=window)
        day_col = column_index(schema, family="history", stat="history_days", window=window)
        for patient_id, n_events, n_encounters, history_days in rows:
            row_index = patient_index.get(patient_id)
            if row_index is None:
                continue
            matrix[row_index, event_col] = n_events
            matrix[row_index, enc_col] = n_encounters
            if history_days is not None:
                matrix[row_index, day_col] = history_days


def _fill_codes(
    con: duckdb.DuckDBPyConnection,
    matrix: np.ndarray,
    schema: list[dict[str, str]],
    patient_index: dict[str, int],
    anchors: dict[str, tuple[date, date]],
    split: str,
) -> None:
    lookup = {
        (column["family"], column["key"], column["window"], column["stat"]): index
        for index, column in enumerate(schema)
        if column["name"].startswith("code|")
    }
    for window, predicate in WINDOW_SQL.items():
        rows = con.execute(
            f"\n            SELECT m.patient_id, m.family, m.code,\n                   count(*) AS n,\n                   count(DISTINCT m.encounter_id) AS n_enc,\n                   max(m.event_date) AS last_date\n            FROM code_mentions m\n            JOIN anchors a ON m.patient_id = a.patient_id\n            WHERE m.split = ? AND {predicate}\n            GROUP BY 1, 2, 3\n            ",
            [split],
        ).fetchall()
        for patient_id, family, code, n, n_enc, last_date in rows:
            row_index = patient_index.get(patient_id)
            if row_index is None:
                continue
            key = (str(family), str(code), window)
            count_col = lookup.get((*key, "count"))
            if count_col is None:
                continue
            matrix[row_index, count_col] = n
            matrix[row_index, lookup[*key, "ind"]] = 1
            matrix[row_index, lookup[*key, "n_enc"]] = n_enc
            last = _as_date(last_date)
            if last is not None:
                matrix[row_index, lookup[*key, "days_since"]] = (anchors[patient_id][0] - last).days


def _fill_labs(
    con: duckdb.DuckDBPyConnection,
    matrix: np.ndarray,
    schema: list[dict[str, str]],
    patient_index: dict[str, int],
    anchors: dict[str, tuple[date, date]],
    split: str,
) -> None:
    lookup = {
        (column["key"], column["unit"], column["window"], column["stat"]): index
        for index, column in enumerate(schema)
        if column["family"] == "observation_numeric"
    }
    if not lookup:
        return
    for window, predicate in WINDOW_SQL.items():
        rows = con.execute(
            f"\n            WITH obs AS (\n                SELECT m.patient_id, m.code, coalesce(m.units, '') AS units,\n                       m.event_date, try_cast(m.value_raw AS DOUBLE) AS val\n                FROM safe_events m\n                JOIN anchors a ON m.patient_id = a.patient_id\n                WHERE m.split = ?\n                  AND m.source_table = 'observations'\n                  AND lower(coalesce(m.obs_type, '')) = 'numeric'\n                  AND {predicate}\n            )\n            SELECT patient_id, code, units, count(val), avg(val), min(val), max(val)\n            FROM obs\n            WHERE val IS NOT NULL\n            GROUP BY 1, 2, 3\n            ",
            [split],
        ).fetchall()
        for patient_id, code, units, n, mean, min_value, max_value in rows:
            row_index = patient_index.get(patient_id)
            base = (str(code), str(units), window)
            if row_index is None or (*base, "count") not in lookup:
                continue
            matrix[row_index, lookup[*base, "count"]] = n
            matrix[row_index, lookup[*base, "missing"]] = 0
            matrix[row_index, lookup[*base, "mean"]] = mean
            matrix[row_index, lookup[*base, "min"]] = min_value
            matrix[row_index, lookup[*base, "max"]] = max_value
        latest_rows = con.execute(
            f"\n            WITH obs AS (\n                SELECT m.patient_id, m.code, coalesce(m.units, '') AS units,\n                       m.event_date, m.event_ts, try_cast(m.value_raw AS DOUBLE) AS val\n                FROM safe_events m\n                JOIN anchors a ON m.patient_id = a.patient_id\n                WHERE m.split = ?\n                  AND m.source_table = 'observations'\n                  AND lower(coalesce(m.obs_type, '')) = 'numeric'\n                  AND {predicate}\n            ),\n            timed AS (\n                SELECT patient_id, code, units, event_ts,\n                       any_value(event_date) AS event_date,\n                       count(DISTINCT val) AS n_distinct,\n                       min(val) AS only_value\n                FROM obs\n                WHERE val IS NOT NULL AND event_ts IS NOT NULL\n                GROUP BY 1, 2, 3, 4\n            ),\n            date_only AS (\n                SELECT patient_id, code, units, event_date,\n                       count(DISTINCT val) AS n_distinct,\n                       min(val) AS only_value\n                FROM obs\n                WHERE val IS NOT NULL AND event_ts IS NULL\n                GROUP BY 1, 2, 3, 4\n            ),\n            days_with_time AS (\n                SELECT DISTINCT patient_id, code, units, event_date\n                FROM obs\n                WHERE val IS NOT NULL AND event_ts IS NOT NULL\n            ),\n            usable AS (\n                SELECT patient_id, code, units, event_ts AS sort_ts, event_date, only_value\n                FROM timed\n                WHERE n_distinct = 1\n                UNION ALL\n                SELECT d.patient_id, d.code, d.units,\n                       CAST(d.event_date AS TIMESTAMP) AS sort_ts,\n                       d.event_date, d.only_value\n                FROM date_only d\n                ANTI JOIN days_with_time t\n                    ON d.patient_id = t.patient_id\n                   AND d.code = t.code\n                   AND d.units = t.units\n                   AND d.event_date = t.event_date\n                WHERE d.n_distinct = 1\n            ),\n            ranked AS (\n                SELECT *,\n                       row_number() OVER (\n                           PARTITION BY patient_id, code, units\n                           ORDER BY sort_ts DESC\n                       ) AS rn\n                FROM usable\n            )\n            SELECT patient_id, code, units,\n                   max(CASE WHEN rn = 1 THEN only_value END),\n                   max(CASE WHEN rn = 2 THEN only_value END),\n                   max(CASE WHEN rn = 1 THEN event_date END)\n            FROM ranked\n            WHERE rn <= 2\n            GROUP BY 1, 2, 3\n            ",
            [split],
        ).fetchall()
        for patient_id, code, units, latest, previous, latest_date in latest_rows:
            row_index = patient_index.get(patient_id)
            base = (str(code), str(units), window)
            if row_index is None or (*base, "latest") not in lookup:
                continue
            if latest is not None:
                matrix[row_index, lookup[*base, "latest"]] = latest
            if previous is not None:
                matrix[row_index, lookup[*base, "previous"]] = previous
            if latest is not None and previous is not None:
                matrix[row_index, lookup[*base, "delta"]] = float(latest) - float(previous)
            last = _as_date(latest_date)
            if last is not None:
                matrix[row_index, lookup[*base, "days_since"]] = (
                    anchors[patient_id][0] - last
                ).days


def _fill_text(
    con: duckdb.DuckDBPyConnection,
    matrix: np.ndarray,
    schema: list[dict[str, str]],
    patient_index: dict[str, int],
    split: str,
) -> None:
    lookup = {
        (column["key"], column["value"], column["window"]): index
        for index, column in enumerate(schema)
        if column["family"] == "observation_text"
    }
    if not lookup:
        return
    for window, predicate in WINDOW_SQL.items():
        rows = con.execute(
            f"\n            SELECT m.patient_id, m.code, m.value_raw, count(*)\n            FROM safe_events m\n            JOIN anchors a ON m.patient_id = a.patient_id\n            WHERE m.split = ?\n              AND m.source_table = 'observations'\n              AND lower(coalesce(m.obs_type, '')) <> 'numeric'\n              AND m.code IS NOT NULL\n              AND m.value_raw IS NOT NULL\n              AND {predicate}\n            GROUP BY 1, 2, 3\n            ",
            [split],
        ).fetchall()
        for patient_id, code, value, n in rows:
            row_index = patient_index.get(patient_id)
            column = lookup.get((str(code), str(value), window))
            if row_index is None or column is None:
                continue
            matrix[row_index, column] = n
