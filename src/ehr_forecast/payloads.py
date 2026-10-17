"""Keep ECG examination facts while excluding unverified payloads."""

from __future__ import annotations

from . import event_schema as events
from .io import ordinary_text


def sanitize_events(con):
    """Keep the ECG examination fact, never its unqualified text payload."""
    if not set(events.EVENT_COLUMNS) <= {
        r[0] for r in con.execute("DESCRIBE source_events").fetchall()
    }:
        raise ValueError("Unexpected event schema")
    con.execute("""CREATE OR REPLACE TEMP VIEW e AS SELECT * REPLACE (
        CASE WHEN source_table='observations' AND code='29303009'
             THEN NULL ELSE value_raw END AS value_raw) FROM source_events""")
    for (value,) in con.execute("""SELECT DISTINCT value_raw FROM e
        WHERE source_table='observations' AND lower(coalesce(obs_type,''))<>'numeric'
        AND value_raw IS NOT NULL""").fetchall():
        ordinary_text(value)
