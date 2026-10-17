import inspect

import numpy as np
import pytest

from ehr_forecast.engineering import fit_quality
from ehr_forecast.model import predict_native, train_native
from ehr_forecast.selection import (
    binned_mutual_information,
    scale_fit,
    scale_transform,
    select_features,
)
from ehr_forecast.training import cloud_guard


def test_local_formal_fit_rejected(monkeypatch):
    monkeypatch.delenv("RUNPOD_POD_ID", raising=False)
    with pytest.raises(RuntimeError, match="RunPod"):
        cloud_guard()


def test_native_loss_replay_and_constants(tmp_path):
    rng = np.random.default_rng(54)
    x = rng.normal(size=(60, 7)).astype(np.float32)
    x[::4, 2] = np.nan
    y = np.column_stack([x[:, 0] > 0, np.zeros(60)]).astype(np.float32)
    result = train_native(x[:50], y[:50], tmp_path, quick=True, threads=1)
    assert (
        result["fit_rows"] == 50
        and len(result["training_loss"]) == 5
        and (not result["early_stopping"])
    )
    p = predict_native(x[50:], tmp_path)
    assert p.shape == (10, 2) and np.all(p[:, 1] == 0)
    assert np.isfinite(result["target_training_loss"]).all()
    assert np.array_equal(
        predict_native(x[:50], tmp_path), np.load(tmp_path / "train_predictions.npy")
    )
    with pytest.raises(FileExistsError):
        train_native(x[:50], y[:50], tmp_path, quick=True)
    with pytest.raises(ValueError):
        predict_native(x[:, :-1], tmp_path)
    with pytest.raises(ValueError):
        predict_native(x, tmp_path, iteration=6)
    (tmp_path / "target_00.txt").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        predict_native(x, tmp_path)


def toy():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(72, 12)).astype(np.float32)
    x[::4, 2] = np.nan
    y = np.stack(
        [x[:, 0] > 0, x[:, 1] + x[:, 3] > 0.2, np.zeros(len(x), dtype=bool)], axis=1
    ).astype(np.int8)
    return (x, y)


def test_scaler_does_not_learn_from_scoring_and_keeps_missing():
    x, _ = toy()
    state = scale_fit(x[:50])
    before = {k: v.copy() for k, v in state.items()}
    z = scale_transform(x[50:] * 100000, state)
    assert np.isfinite(z).all()
    for k, v in state.items():
        np.testing.assert_array_equal(v, before[k])
    assert z.shape[1] > x.shape[1]


def test_mi_identifies_signal_and_constant_targets():
    x, y = toy()
    scores = binned_mutual_information(x, y)
    assert scores[:, 0].argmax() == 0
    assert np.all(scores[:, 2] == 0)
    assert np.isfinite(scores).all()


def test_train_only_selector_artifacts(tmp_path):
    x, y = toy()
    cols, _ = fit_quality(x)
    result = select_features(x, y, cols, tmp_path, quick=True, threads=1)
    assert not result["scoring_labels_used"]
    with np.load(tmp_path / "selection.npz", allow_pickle=False) as z:
        assert len(z["compact"]) <= len(cols)
        assert np.all(z["stability_votes"][np.isin(z["quality"], z["stable"])] >= 2)
    assert "score_y" not in inspect.signature(select_features).parameters
