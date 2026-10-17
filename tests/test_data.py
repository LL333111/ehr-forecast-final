import json
from datetime import date

import duckdb
import pytest

from ehr_forecast import data as final
from ehr_forecast import temporal as trajectories
from ehr_forecast.events import prepare
from ehr_forecast.schema import EVENT_HEADERS, PATIENTS_HEADER

from .helpers import connection, row, write_csv


@pytest.mark.parametrize("ids", [["v"], ["t"], ["p", "v"], ["p", "p"], []])
def test_fit_rejects_heldout_duplicate_empty(ids):
    with pytest.raises(ValueError):
        final.assert_train_ids(ids, {"p": "train", "q": "train", "v": "val", "t": "test"}, True)


def test_final_fit_requires_all_train():
    with pytest.raises(ValueError):
        final.assert_train_ids(["p"], {"p": "train", "q": "train"}, True)
    final.assert_train_ids(["q", "p"], {"p": "train", "q": "train", "v": "val"}, True)


def test_fit_gate_precedes_any_data_operation():
    with pytest.raises(ValueError, match="official train"):
        final.fit_representation({"splits": {"v": "val"}}, ["v"])


def test_row_local_adapter_preserves_official_split():
    with connection([row(), row(patient="v", split="val")]) as c:
        prepare(c, ["p1", "v"])
        assert dict(c.execute("SELECT patient_id,official_split FROM people").fetchall()) == {
            "p1": "train",
            "v": "val",
        }
        assert c.execute("SELECT split FROM clean WHERE patient_id='v'").fetchone()[0] == "val"


def test_row_local_adapter_rejects_partition_mismatch():
    with connection([row(patient="v", split="train")]) as c:
        with pytest.raises(ValueError, match="partition"):
            prepare(c, ["v"])


@pytest.mark.parametrize("bad", [row(when="2021-01-01 00:00:00"), row(when="1970-01-01 00:00:00")])
def test_temporal_boundary_kept(bad):
    with connection([bad]) as c:
        with pytest.raises(ValueError, match="unsafe"):
            prepare(c, ["p1"])


def test_vocabulary_and_transform_keep_val_separate():
    rows = [
        row(code="common", value="10"),
        row(patient="v", split="val", code="common", value="20"),
        row(patient="v", split="val", code="heldout_only", value="99999"),
    ]
    with connection(rows) as c:
        prepare(c, ["p1", "v"])
        trajectories.prepare_tabular(c)
        c.execute(
            "CREATE TABLE base_people AS SELECT patient_id,official_split split,anchor_date,birth_date,gender FROM people"
        )
        c.execute("CREATE VIEW base_events AS SELECT * FROM e")
        splits = {"p1": "train", "v": "val"}
        dates = {p: (date(2021, 1, 1), date(2026, 1, 1)) for p in splits}
        final.baseline_views(c, ["p1"], dates, splits)
        schema = final.base._feature_schema(c)
        assert not any(("heldout_only" in json.dumps(s) for s in schema))
        x = final.baseline_transform(
            c, ["v"], dates, {p: (date(1980, 1, 1), "F") for p in splits}, splits, schema
        )
        i = next(
            (
                i
                for i, s in enumerate(schema)
                if s["family"] == "observation_numeric"
                and s["key"] == "common"
                and (s["stat"] == "latest")
                and (s["window"] == "full")
            )
        )
        assert x[0, i] == 20
        assert c.execute("SELECT split FROM anchors").fetchone()[0] == "val"


def raw_fixture(tmp):
    raw = tmp / "raw"
    for part in ["train_val", "test"]:
        ids = ["p", "v"] if part == "train_val" else ["t"]
        write_csv(
            raw / part / "patients.csv",
            PATIENTS_HEADER,
            [dict(Id=p, BIRTHDATE="1980-01-01", GENDER="F") for p in ids],
        )
        for table, fields in EVENT_HEADERS.items():
            rows = []
            if table == "encounters":
                rows = [
                    dict(
                        Id=p + "visit",
                        PATIENT=p,
                        START="2019-01-01",
                        STOP="2019-01-01",
                        CODE="visit",
                        BASE_ENCOUNTER_COST="999",
                    )
                    for p in ids
                ]
                if part == "train_val":
                    rows += [
                        dict(
                            Id=p + "last",
                            PATIENT=p,
                            START="2025-01-01",
                            STOP="2025-01-01",
                            CODE="future",
                        )
                        for p in ids
                    ]
            elif table == "conditions":
                rows = [dict(PATIENT=p, START="2019-01-01", CODE="past") for p in ids]
                if part == "train_val":
                    rows += [dict(PATIENT=p, START="2021-01-01", CODE="target") for p in ids]
            elif table == "observations":
                rows = [
                    dict(
                        PATIENT=p,
                        DATE="2019-01-01",
                        CODE="lab",
                        TYPE="numeric",
                        VALUE="1",
                        UNITS="u",
                    )
                    for p in ids
                ]
                rows += [
                    dict(PATIENT=p, DATE="2019-01-01", CODE="QALY", TYPE="numeric", VALUE="999")
                    for p in ids
                ]
            write_csv(raw / part / (table + ".csv"), fields, rows)
    write_csv(
        raw / "patient_splits.csv",
        ["Id", "split"],
        [dict(Id=p, split=s) for p, s in [("p", "train"), ("v", "val"), ("t", "test")]],
    )
    write_csv(
        raw / "test_anchors.csv", ["Id", "anchor_date"], [dict(Id="t", anchor_date="2020-01-01")]
    )
    write_csv(
        raw / "target_conditions.csv",
        ["CODE", "DESCRIPTION"],
        [dict(CODE="target", DESCRIPTION="synthetic target")],
    )
    (raw / "ReadMe.md").write_text("Synthetic test only", encoding="utf-8")
    return raw


def test_raw_cache_end_to_end_never_fits_or_reads_labels(tmp_path):
    raw = raw_fixture(tmp_path)
    cache = tmp_path / "cache"
    before = final.raw_hashes(raw)
    result = final.build_raw_cache(raw, cache)
    assert not result["labels_opened"] and (not result["preprocessing_fitted"])
    assert result["official_counts"] == {"train": 1, "val": 1, "test": 1}
    assert final.raw_hashes(raw) == before
    with duckdb.connect(str(cache / "database/ehr.duckdb"), read_only=True) as c:
        assert (
            c.execute(
                "SELECT count(*) FROM clinical.events WHERE code IN ('QALY','target','future') OR base_cost IS NOT NULL OR reason_code IS NOT NULL"
            ).fetchone()[0]
            == 0
        )
        assert c.execute(
            "SELECT anchor_date FROM clinical.patients WHERE patient_id='t'"
        ).fetchone()[0] == date(2020, 1, 1)
    y, prior = final.train_labels(raw, cache, ["p"], ["target"])
    assert y.tolist() == [[1]] and prior.tolist() == [[0]]
    with pytest.raises(ValueError):
        final.train_labels(raw, cache, ["v"], ["target"])
    assert final.build_raw_cache(raw, cache) == result
    with feature_context_for_test(raw, cache) as ctx:
        protocol = final.fit_representation(ctx, ["p"])
        assert protocol["fit_count"] == 1
        assert final.transform(ctx, protocol, ["t"]).shape == (1, len(protocol["catalog"]))


def feature_context_for_test(raw, cache):
    return final.feature_context(raw, cache)
