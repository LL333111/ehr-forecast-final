import csv

import numpy as np
import pytest

from ehr_forecast.inference import predict_bundle, verify_bundle, write_predictions
from ehr_forecast.io import write


def test_submission_format_mask_and_roundtrip(tmp_path):
    p = np.array([[0.12345678912345678, 0.999], [0.1, 0.2]])
    prior = np.array([[0, 1], [0, 0]])
    path = tmp_path / "predictions.csv"
    masked = write_predictions(path, ["test-b", "test-a"], ["42", "08"], p, prior)
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["patient_id", "42", "08"] and [r[0] for r in rows[1:]] == [
        "test-b",
        "test-a",
    ]
    assert np.array_equal(np.array([r[1:] for r in rows[1:]], float), masked)
    assert masked[0, 1] == 0 and p[0, 1] == 0.999
    with pytest.raises(FileExistsError):
        write_predictions(path, ["test-b", "test-a"], ["42", "08"], p, prior)


@pytest.mark.parametrize(
    "change", ["duplicate_id", "duplicate_code", "nan", "range", "shape", "prior"]
)
def test_invalid_submission_is_rejected_without_output(tmp_path, change):
    ids = ["a", "b"]
    codes = ["1", "2"]
    p = np.ones((2, 2)) * 0.3
    prior = np.zeros((2, 2))
    if change == "duplicate_id":
        ids = ["a", "a"]
    if change == "duplicate_code":
        codes = ["1", "1"]
    if change == "nan":
        p[0, 0] = np.nan
    if change == "range":
        p[0, 0] = 1.01
    if change == "shape":
        p = p[:1]
    if change == "prior":
        prior[0, 0] = 0.5
    out = tmp_path / "p.csv"
    with pytest.raises(ValueError):
        write_predictions(out, ids, codes, p, prior)
    assert not out.exists()


def test_inference_cannot_be_used_for_fitting(tmp_path):
    with pytest.raises(ValueError, match="held-out"):
        predict_bundle(tmp_path, tmp_path, tmp_path, tmp_path / "x", "train")


def test_bundle_escape_rejected(tmp_path):
    write(
        tmp_path / "manifest.json",
        dict(training_data_excluded=True, raw_data_excluded=True, files={"../outside": "bad"}),
    )
    with pytest.raises(ValueError, match="Unsafe"):
        verify_bundle(tmp_path)
