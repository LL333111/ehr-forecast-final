"""Row-local clinical context, timing and qualified text features."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from urllib.parse import quote

import numpy as np

from .schema import EVENT_HEADERS, PATIENTS_HEADER

TABLES = tuple(sorted(EVENT_HEADERS))
INTERVAL_TABLES = tuple((t for t in TABLES if "STOP" in EVENT_HEADERS[t]))
DEMOGRAPHIC_FIELDS = ("RACE", "ETHNICITY", "BIRTHPLACE")
WINDOWS = ("full", "y5", "y1", "d90")
DESCRIPTION_FIELDS = (
    "DESCRIPTION",
    "REASONDESCRIPTION",
    "BODYSITE_DESCRIPTION",
    "MODALITY_DESCRIPTION",
    "SOP_DESCRIPTION",
)
COST_FIELDS = {
    "encounters": "BASE_ENCOUNTER_COST",
    "medications": "BASE_COST",
    "procedures": "BASE_COST",
    "immunizations": "BASE_COST",
}
GROUPS = ("state", "visit", "demographics", "text", "other")
CONTEXT_FIELDS = ("ORGANIZATION", "PROVIDER", "PAYER")
CONTEXT_HASH_DIM = 128


def _ids(ids):
    ids = list(ids)
    if not ids or len(ids) != len(set(ids)) or any((not isinstance(p, str) or not p for p in ids)):
        raise ValueError("Nonempty unique patient IDs required")
    return ids


def _date(value):
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value).strip()[:10])
    except (TypeError, ValueError):
        return None


def _instant(value):
    """An exact UTC clock when present; date-only observations remain date-only."""
    text = str(value or "").strip()
    if len(text) < 19:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed
    except ValueError:
        return None


def known_interval(start_raw, stop_raw, anchor):
    """Return an observed historical end or an indistinguishable unknown end.

    The audit reason is never a predictive feature. Future, blank, malformed,
    and reverse-order STOP values all produce the same unavailable end values.
    """
    anchor = _date(anchor)
    start = _date(start_raw)
    if anchor is None or start is None or start >= anchor:
        raise ValueError("A strictly pre-anchor START is required")
    end = _date(stop_raw)
    unknown = {
        "observed_end": False,
        "end_date": None,
        "duration_days": None,
        "days_since_end": None,
    }
    if not str(stop_raw or "").strip():
        return {**unknown, "audit_reason": "blank"}
    if end is None:
        return {**unknown, "audit_reason": "malformed"}
    if end >= anchor:
        return {**unknown, "audit_reason": "at_or_after_anchor"}
    start_clock, stop_clock = (_instant(start_raw), _instant(stop_raw))
    duration = (
        (stop_clock - start_clock).total_seconds() / 86400.0
        if start_clock is not None and stop_clock is not None
        else (end - start).days
    )
    if duration < 0 or end < start:
        return {**unknown, "audit_reason": "reverse_order"}
    return {
        "observed_end": True,
        "end_date": end,
        "duration_days": float(duration),
        "days_since_end": float((anchor - end).days),
        "audit_reason": "historical_end",
    }


def _years_before(anchor, years):
    try:
        return anchor.replace(year=anchor.year - years)
    except ValueError:
        return anchor.replace(year=anchor.year - years, day=28)


def _windows(stamp, anchor):
    yield "full"
    if stamp >= _years_before(anchor, 5):
        yield "y5"
    if stamp >= _years_before(anchor, 1):
        yield "y1"
    if stamp >= anchor - timedelta(days=90):
        yield "d90"


def _hash(value, dimension):
    return (
        int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "little")
        % dimension
    )


@lru_cache(maxsize=16384)
def description_hash(table, column, text, dimension):
    """Fixed word/bigram hashing, with no corpus vocabulary or fitted IDF."""
    words = re.findall("\\w+", str(text or "").casefold(), flags=re.UNICODE)
    terms = ["u:" + word for word in words]
    terms.extend(("b:" + left + " " + right for left, right in zip(words, words[1:])))
    prefix = "v10-description-v1\x00" + table + "\x00" + column + "\x00"
    return tuple(sorted(Counter((_hash(prefix + term, dimension) for term in terms)).items()))


def _column_disposition(table, column):
    if table == "patients":
        if column == "Id":
            return ("linkage_only", "Patient linkage; never a learned identifier", None)
        if column in {"BIRTHDATE", "GENDER"}:
            return ("baseline", "Age at anchor/event and gender are in the retained baseline", None)
        if column in DEMOGRAPHIC_FIELDS:
            return (
                "included",
                "Provided demographic/birth-location category; fit-only vocabulary",
                "demographics",
            )
        if column in {"DEATHDATE", "HEALTHCARE_EXPENSES", "HEALTHCARE_COVERAGE"}:
            return (
                "excluded_temporal",
                "Future death or lifetime cumulative totals are not anchor-limited",
                None,
            )
        if column in {"MARITAL", "ADDRESS", "CITY", "STATE", "COUNTY", "ZIP", "LAT", "LON"}:
            return (
                "excluded_temporal",
                "Un-timestamped current snapshot, not demonstrably the anchor state",
                None,
            )
        if column in {"SSN", "DRIVERS", "PASSPORT", "PREFIX", "FIRST", "LAST", "SUFFIX", "MAIDEN"}:
            return (
                "identity_only",
                "Identity/name fields; no raw identity as predictive input",
                None,
            )
    else:
        if column in {"PATIENT", "ENCOUNTER", "Id"}:
            return (
                "linkage_only",
                "Within-patient encounter linkage/counting only; UUID values never encoded",
                "visit",
            )
        if column in {"START", "DATE"}:
            return (
                "baseline_and_included",
                "Strict event date < anchor; baseline time plus addition windows",
                "state",
            )
        if column == "STOP":
            return (
                "included_past_only",
                "Only START<=STOP<anchor; future/blank/invalid ends indistinguishable in inputs",
                "state",
            )
        if column == "ENCOUNTERCLASS":
            return (
                "included",
                "Type attached to a strictly pre-anchor encounter; fit-only categories",
                "visit",
            )
        if column in DESCRIPTION_FIELDS:
            return (
                "included",
                "All provided past description fields use fixed word/bigram hashing",
                "text",
            )
        if column in {
            "CODE",
            "REASONCODE",
            "BODYSITE_CODE",
            "MODALITY_CODE",
            "SOP_CODE",
            "VALUE",
            "UNITS",
            "TYPE",
        }:
            return (
                "baseline",
                "Existing baseline preserves code/reason/measurement/text-value representation; codes also summarize encounters",
                None,
            )
        if column == COST_FIELDS.get(table):
            return (
                "included",
                "Base single-service/prescription unit cost on a past event; excludes cumulative claims/course costs",
                "other",
            )
        if column in {"TOTALCOST", "TOTAL_CLAIM_COST", "PAYER_COVERAGE", "DISPENSES"}:
            return (
                "excluded_temporal",
                "Course/claim/reimbursement aggregate has no pre-anchor accounting cutoff",
                None,
            )
        if column in CONTEXT_FIELDS:
            return (
                "included_context_hash",
                "Strictly past event context: per-field distinct counts and fixed field-prefixed hashes; raw identifiers never named",
                "visit",
            )
        if column == "UDI":
            return (
                "included_aggregate",
                "Strictly past observed distinct devices and repeat counts only; raw device identifier is neither hashed nor named",
                "visit",
            )
    raise ValueError(f"Unclassified source field {table}.{column}; explicit review required")


def coverage_ledger(headers):
    if set(headers) != set(TABLES) | {"patients"}:
        raise ValueError("All ten event tables and patients must be present")
    result = []
    for table, columns in sorted(headers.items()):
        expected = PATIENTS_HEADER if table == "patients" else EVENT_HEADERS[table]
        if len(columns) != len(set(columns)) or set(columns) != set(expected):
            raise ValueError(f"Unexpected source schema for {table}")
        for column in columns:
            disposition, reason, group = _column_disposition(table, column)
            result.append(
                dict(
                    table=table, column=column, disposition=disposition, reason=reason, group=group
                )
            )
    return result


@dataclass
class Extracted:
    ids: list[str]
    text_hash_dim: int
    rows: dict[str, dict[str, float]]
    catalog: dict[str, dict]
    fixed_names: set[str]
    ledger: list[dict]
    audit: dict


@dataclass
class _Stats:
    count: int = 0
    total: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def add(self, value):
        self.count += 1
        self.total += value
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)


def extract_from_rows(event_rows, patient_rows, anchors, ids, text_hash_dim=512, headers=None):
    """Pure extraction API for prevalidated schemas and patient-local row streams."""
    ids = _ids(ids)
    if not isinstance(text_hash_dim, int) or not 1 <= text_hash_dim <= 8192:
        raise ValueError("Hash dimension must be between 1 and 8192")
    if set(event_rows) != set(TABLES):
        raise ValueError("All ten event streams required")
    selected = set(ids)
    dates = {p: _date(anchors.get(p)) for p in ids}
    if any((value is None for value in dates.values())):
        raise ValueError("Missing/invalid requested anchor")
    headers = headers or {**EVENT_HEADERS, "patients": PATIENTS_HEADER}
    ledger = coverage_ledger(headers)
    rows = {p: {} for p in ids}
    catalog, fixed = ({}, set())
    audit = {
        "patients": len(ids),
        "past_rows": Counter(),
        "invalid_start_rows": Counter(),
        "stop_reasons": defaultdict(Counter),
        "invalid_cost_rows": Counter(),
        "unlinked_event_rows": Counter(),
    }

    def put(
        p,
        group,
        source,
        key,
        stat,
        value,
        window="full",
        zero=True,
        static=False,
        category_field=None,
        add=False,
    ):
        parts = (group, source, key, stat, window)
        name = "v10|" + "|".join((quote(str(part), safe="") for part in parts))
        if name not in catalog:
            catalog[name] = dict(
                name=name,
                group=group,
                family="v10_" + group,
                source=source,
                key=key,
                stat=stat,
                window=window,
                default="zero" if zero else "nan",
                category_field=category_field,
            )
        if static:
            fixed.add(name)
        if p is not None:
            if not math.isfinite(float(value)):
                raise ValueError("Nonfinite extracted feature")
            rows[p][name] = rows[p].get(name, 0.0) + float(value) if add else float(value)
        return name

    for source in DEMOGRAPHIC_FIELDS + ("ENCOUNTERCLASS",):
        group = "demographics" if source in DEMOGRAPHIC_FIELDS else "visit"
        for window in ("full",) if group == "demographics" else WINDOWS:
            put(
                None,
                group,
                source,
                "<unknown>",
                "count",
                0,
                window,
                static=True,
                category_field=source,
            )
    for window in WINDOWS:
        for index in range(CONTEXT_HASH_DIM):
            put(None, "visit", "event_context", str(index), "hash_count", 0, window, static=True)
        for column in CONTEXT_FIELDS:
            for stat in ("observed_count", "distinct_count"):
                put(None, "visit", "event_context", column, stat, 0, window, static=True)
        for stat in ("observed_count", "distinct_count", "repeat_count"):
            put(None, "visit", "devices", "UDI", stat, 0, window, static=True)
        for index in range(text_hash_dim):
            put(None, "text", "descriptions", str(index), "hash_count", 0, window, static=True)
            put(None, "visit", "code_adjacency", str(index), "hash_count", 0, window, static=True)
        for key in ("visits", "without_past_encounter_row"):
            put(None, "visit", "composition", "<all>", key, 0, window, static=True)
        for key in ("events_per_visit", "families_per_visit", "codes_per_visit") + tuple(
            ("events_" + t for t in TABLES)
        ):
            for stat in ("mean", "max"):
                put(None, "visit", "composition", key, stat, 0, window, zero=False, static=True)
        for left, right in itertools.combinations(TABLES, 2):
            put(
                None,
                "visit",
                "family_pair",
                left + "+" + right,
                "cooccurring_visits",
                0,
                window,
                static=True,
            )
        for table, column in COST_FIELDS.items():
            for stat in (
                "base_cost_count",
                "base_cost_sum",
                "base_cost_mean",
                "base_cost_min",
                "base_cost_max",
            ):
                put(
                    None,
                    "other",
                    table,
                    column,
                    stat,
                    0,
                    window,
                    zero=stat in {"base_cost_count", "base_cost_sum"},
                    static=True,
                )
    for table in INTERVAL_TABLES:
        for window in WINDOWS:
            put(None, "state", table, "<all>", "observed_end_count", 0, window, static=True)
        put(None, "state", table, "<all>", "no_observed_end_count", 0, static=True)
        for stat in (
            "known_duration_mean_days",
            "known_duration_max_days",
            "known_duration_min_days",
            "days_since_last_observed_end",
            "unended_observed_elapsed_mean_days",
        ):
            put(None, "state", table, "<all>", stat, 0, zero=False, static=True)
    seen = set()
    for row in patient_rows:
        p = str(row.get("Id", "")).strip()
        if p not in selected:
            continue
        if p in seen:
            raise ValueError("Duplicate requested patient demographic row")
        seen.add(p)
        for source in DEMOGRAPHIC_FIELDS:
            raw_value = str(row.get(source, "") or "").strip()
            value = "value:" + raw_value if raw_value else "<missing>"
            put(p, "demographics", source, value, "count", 1, category_field=source)
    if seen != selected:
        raise ValueError("Missing requested patient demographics")
    state = defaultdict(
        lambda: {
            "ended": 0,
            "unknown": 0,
            "durations": _Stats(),
            "recency": math.inf,
            "unended_elapsed": _Stats(),
            "ended_windows": Counter(),
        }
    )
    costs = defaultdict(_Stats)
    contexts = defaultdict(Counter)
    devices = defaultdict(Counter)
    visits = defaultdict(
        lambda: {"counts": Counter(), "codes": set(), "dates": [], "parent": False}
    )
    for table in TABLES:
        stamp_column = "START" if "START" in EVENT_HEADERS[table] else "DATE"
        for row in event_rows[table]:
            p = str(row.get("PATIENT", "")).strip()
            if p not in selected:
                continue
            stamp_raw = row.get(stamp_column)
            stamp = _date(stamp_raw)
            if stamp is None:
                audit["invalid_start_rows"][table] += 1
                continue
            if stamp >= dates[p]:
                continue
            audit["past_rows"][table] += 1
            windows = tuple(_windows(stamp, dates[p]))
            for column in CONTEXT_FIELDS:
                if column not in EVENT_HEADERS[table]:
                    continue
                value = str(row.get(column, "") or "").strip()
                if value:
                    index = _hash(
                        "v10-event-context-v1\x00" + column + "\x00" + value, CONTEXT_HASH_DIM
                    )
                    for window in windows:
                        contexts[p, column, window][value] += 1
                        put(
                            p,
                            "visit",
                            "event_context",
                            str(index),
                            "hash_count",
                            1,
                            window,
                            static=True,
                            add=True,
                        )
            if "UDI" in EVENT_HEADERS[table]:
                device = str(row.get("UDI", "") or "").strip()
                if device:
                    for window in windows:
                        devices[p, window][device] += 1
            raw_code = str(row.get("CODE", "") or "").strip()
            code = "code:" + raw_code if raw_code else "<missing-code>"
            if table in INTERVAL_TABLES:
                interval = known_interval(stamp_raw, row.get("STOP"), dates[p])
                audit["stop_reasons"][table][interval["audit_reason"]] += 1
                for key in ("<all>", code):
                    aggregate = state[p, table, key]
                    if interval["observed_end"]:
                        aggregate["ended"] += 1
                        aggregate["durations"].add(interval["duration_days"])
                        aggregate["recency"] = min(aggregate["recency"], interval["days_since_end"])
                        aggregate["ended_windows"].update(_windows(interval["end_date"], dates[p]))
                    else:
                        aggregate["unknown"] += 1
                        aggregate["unended_elapsed"].add(float((dates[p] - stamp).days))
            if table == "encounters":
                raw_value = str(row.get("ENCOUNTERCLASS", "") or "").strip()
                value = "value:" + raw_value if raw_value else "<missing>"
                for window in windows:
                    put(
                        p,
                        "visit",
                        "ENCOUNTERCLASS",
                        value,
                        "count",
                        1,
                        window,
                        category_field="ENCOUNTERCLASS",
                        add=True,
                    )
            for column in DESCRIPTION_FIELDS:
                if column not in EVENT_HEADERS[table]:
                    continue
                text = row.get(column, "")
                if text:
                    for index, count in description_hash(table, column, text, text_hash_dim):
                        for window in windows:
                            put(
                                p,
                                "text",
                                "descriptions",
                                str(index),
                                "hash_count",
                                count,
                                window,
                                static=True,
                                add=True,
                            )
            cost_column = COST_FIELDS.get(table)
            if cost_column:
                raw_cost = str(row.get(cost_column, "") or "").strip()
                if raw_cost:
                    try:
                        cost = float(raw_cost)
                    except ValueError:
                        cost = math.nan
                    if math.isfinite(cost):
                        for window in windows:
                            costs[p, table, window].add(cost)
                    else:
                        audit["invalid_cost_rows"][table] += 1
            encounter = str(
                row.get("Id" if table == "encounters" else "ENCOUNTER", "") or ""
            ).strip()
            if not encounter:
                audit["unlinked_event_rows"][table] += 1
                put(p, "visit", table, "<all>", "unlinked_events", 1, static=True, add=True)
                continue
            visit = visits[p, encounter]
            visit["counts"][table] += 1
            visit["dates"].append(stamp)
            visit["parent"] |= table == "encounters"
            for column in ("CODE", "REASONCODE", "BODYSITE_CODE", "MODALITY_CODE", "SOP_CODE"):
                value = str(row.get(column, "") or "").strip()
                if value:
                    visit["codes"].add(table + "\x00" + column + "\x00" + value)
    for (p, column, window), values in contexts.items():
        put(
            p,
            "visit",
            "event_context",
            column,
            "observed_count",
            sum(values.values()),
            window,
            static=True,
        )
        put(p, "visit", "event_context", column, "distinct_count", len(values), window, static=True)
    for (p, window), values in devices.items():
        observed = sum(values.values())
        for stat, value in [
            ("observed_count", observed),
            ("distinct_count", len(values)),
            ("repeat_count", observed - len(values)),
        ]:
            put(p, "visit", "devices", "UDI", stat, value, window, static=True)
    for (p, table, key), values in state.items():
        static = key == "<all>"
        put(p, "state", table, key, "observed_end_count", values["ended"], static=static)
        put(p, "state", table, key, "no_observed_end_count", values["unknown"], static=static)
        for window in WINDOWS[1:]:
            put(
                p,
                "state",
                table,
                key,
                "observed_end_count",
                values["ended_windows"][window],
                window,
                static=static,
            )
        if values["ended"]:
            duration = values["durations"]
            for stat, value in [
                ("known_duration_mean_days", duration.total / duration.count),
                ("known_duration_max_days", duration.maximum),
                ("known_duration_min_days", duration.minimum),
                ("days_since_last_observed_end", values["recency"]),
            ]:
                put(p, "state", table, key, stat, value, zero=False, static=static)
        if values["unknown"]:
            elapsed = values["unended_elapsed"]
            put(
                p,
                "state",
                table,
                key,
                "unended_observed_elapsed_mean_days",
                elapsed.total / elapsed.count,
                zero=False,
                static=static,
            )
    for (p, table, window), values in costs.items():
        for stat, value, zero in [
            ("base_cost_count", values.count, True),
            ("base_cost_sum", values.total, True),
            ("base_cost_mean", values.total / values.count, False),
            ("base_cost_min", values.minimum, False),
            ("base_cost_max", values.maximum, False),
        ]:
            put(p, "other", table, COST_FIELDS[table], stat, value, window, zero=zero, static=True)
    visit_summary = defaultdict(lambda: defaultdict(_Stats))
    for (p, _encounter), visit in visits.items():
        visit_date = max(visit["dates"])
        families = sorted(visit["counts"])
        codes = sorted(visit["codes"])
        for window in _windows(visit_date, dates[p]):
            summaries = visit_summary[p, window]
            summaries["events_per_visit"].add(float(sum(visit["counts"].values())))
            summaries["families_per_visit"].add(float(len(families)))
            summaries["codes_per_visit"].add(float(len(codes)))
            put(p, "visit", "composition", "<all>", "visits", 1, window, static=True, add=True)
            if not visit["parent"]:
                put(
                    p,
                    "visit",
                    "composition",
                    "<all>",
                    "without_past_encounter_row",
                    1,
                    window,
                    static=True,
                    add=True,
                )
            for table in TABLES:
                summaries["events_" + table].add(float(visit["counts"][table]))
            for left, right in itertools.combinations(families, 2):
                put(
                    p,
                    "visit",
                    "family_pair",
                    left + "+" + right,
                    "cooccurring_visits",
                    1,
                    window,
                    static=True,
                    add=True,
                )
            pairs = zip(codes, codes[1:])
            for left, right in pairs:
                index = _hash("v10-visit-adjacency-v1\x00" + left + "\x01" + right, text_hash_dim)
                put(
                    p,
                    "visit",
                    "code_adjacency",
                    str(index),
                    "hash_count",
                    1,
                    window,
                    static=True,
                    add=True,
                )
    for (p, window), summaries in visit_summary.items():
        for key, values in summaries.items():
            put(
                p,
                "visit",
                "composition",
                key,
                "mean",
                values.total / values.count,
                window,
                zero=False,
                static=True,
            )
            put(
                p,
                "visit",
                "composition",
                key,
                "max",
                values.maximum,
                window,
                zero=False,
                static=True,
            )
    for table in TABLES:
        put(None, "visit", table, "<all>", "unlinked_events", 0, static=True)
    for table in INTERVAL_TABLES:
        for stat in ("observed_end_count", "no_observed_end_count"):
            put(None, "state", table, "<all>", stat, 0, static=True)
    audit["past_rows"] = {table: audit["past_rows"][table] for table in TABLES}
    for key in ("invalid_start_rows", "invalid_cost_rows", "unlinked_event_rows"):
        audit[key] = dict(audit[key])
    audit["stop_reasons"] = {key: dict(value) for key, value in audit["stop_reasons"].items()}
    audit.update(
        event_tables=len(TABLES),
        labels_consumed=False,
        post_anchor_inputs=False,
        statistics_fitted=False,
        text_hash_policy="fixed unigrams+bigrams; no vocabulary/IDF",
        relationship_policy="within-patient encounter linkage; O(k) sorted-code adjacency hash; no patient/encounter UUID encoding",
        context_policy="strictly past organization/provider/payer counts and fixed shared field-prefixed hash; UDI distinct/repeat counts only",
        stop_policy="future/blank/malformed/reversed end has identical unavailable-end inputs",
    )
    return Extracted(ids, text_hash_dim, rows, catalog, fixed, ledger, audit)


def fit_protocol(extracted, fit_ids):
    fit_ids = _ids(fit_ids)
    if not set(fit_ids) <= set(extracted.ids):
        raise ValueError("Fit IDs not in extracted rows")
    names = set(extracted.fixed_names)
    for p in fit_ids:
        names.update(extracted.rows[p])
    schema = [dict(extracted.catalog[name]) for name in sorted(names)]
    return dict(
        version=10,
        text_hash_dim=extracted.text_hash_dim,
        context_hash_dim=CONTEXT_HASH_DIM,
        fitting_patients=len(fit_ids),
        fitting_ids_sha256=hashlib.sha256(
            json.dumps(sorted(fit_ids), separators=(",", ":")).encode()
        ).hexdigest(),
        schema=schema,
        fit_policy="All fit-supported feature keys plus fixed hashes/aggregate columns; no label/support screening",
        scaling="none; row-local counts/statistics; missing numeric summaries remain NaN",
    )


def transform(extracted, ids, protocol):
    ids = _ids(ids)
    if (
        protocol.get("version") != 10
        or protocol.get("text_hash_dim") != extracted.text_hash_dim
        or protocol.get("context_hash_dim") != CONTEXT_HASH_DIM
    ):
        raise ValueError("Feature protocol/hash contract mismatch")
    if not set(ids) <= set(extracted.ids):
        raise ValueError("Scoring IDs outside extracted rows")
    schema = protocol["schema"]
    names = [column["name"] for column in schema]
    if len(names) != len(set(names)) or any(
        (column.get("group") not in GROUPS for column in schema)
    ):
        raise ValueError("Invalid feature schema")
    lookup = {name: i for i, name in enumerate(names)}
    unknown = {
        (c["category_field"], c["window"]): i
        for i, c in enumerate(schema)
        if c.get("category_field") and c["key"] == "<unknown>"
    }
    defaults = np.array(
        [0.0 if c["default"] == "zero" else np.nan for c in schema], dtype=np.float32
    )
    matrix = np.tile(defaults, (len(ids), 1))
    for i, p in enumerate(ids):
        for name, value in extracted.rows[p].items():
            j = lookup.get(name)
            if j is not None:
                matrix[i, j] = value
            elif extracted.catalog[name].get("category_field"):
                column = extracted.catalog[name]
                matrix[i, unknown[column["category_field"], column["window"]]] += value
    if np.isinf(matrix).any():
        raise ValueError("Float32 feature overflow")
    return matrix
