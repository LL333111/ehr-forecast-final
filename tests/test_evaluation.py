import numpy as np
import pytest

from ehr_forecast.data import build_raw_cache
from ehr_forecast.evaluation import evaluate, metrics
from ehr_forecast.inference import write_predictions
from ehr_forecast.io import sha, write

from .test_data import raw_fixture


@pytest.mark.parametrize("change", ["test", "label_read", "changed"])
def test_outcomes_not_opened_without_valid_prediction_receipt(tmp_path, change):
    prediction = tmp_path / "prediction.csv"
    prediction.write_text("patient_id,1\nv,0.2\n", encoding="utf-8")
    write(
        str(prediction) + ".receipt.json",
        dict(
            complete=True,
            split="test" if change == "test" else "val",
            labels_read=change == "label_read",
            fitting_performed=False,
            prediction_sha256="changed" if change == "changed" else sha(prediction),
        ),
    )
    with pytest.raises(ValueError, match="before outcomes"):
        evaluate(tmp_path / "missing_raw", tmp_path / "cache", prediction, tmp_path / "result.json")
    assert not (tmp_path / "result.json").exists()


def test_metrics_match_task_prior_and_undefined_target_rules():
    y = np.array([[1, 0], [0, 0], [0, 0]])
    p = np.array([[0.4, 0.9], [0.99, 0.2], [0.1, 0.4]])
    prior = np.array([[0, 0], [1, 0], [0, 0]])
    result = metrics(y, p, prior)
    assert result["ap"] == [1.0, 0.0]
    assert result["macro_ap"] == 0.5
    assert result["auroc"] == [1.0, None]
    assert result["macro_auroc"] == 1.0


def test_validation_score_reads_only_the_official_population(tmp_path):
    raw = raw_fixture(tmp_path)
    cache = tmp_path / "cache"
    build_raw_cache(raw, cache)
    prediction = tmp_path / "val.csv"
    write_predictions(prediction, ["v"], ["target"], np.array([[0.3]]), np.array([[0]]))
    write(
        str(prediction) + ".receipt.json",
        dict(
            complete=True,
            split="val",
            labels_read=False,
            fitting_performed=False,
            prediction_sha256=sha(prediction),
        ),
    )
    result = evaluate(raw, cache, prediction, tmp_path / "result.json")
    assert result["positives"] == [1] and result["patients"] == 1
    assert result["macro_ap"] == 1.0 and not result["fitting_performed"]
    with pytest.raises(FileExistsError):
        evaluate(raw, cache, prediction, tmp_path / "result.json")
