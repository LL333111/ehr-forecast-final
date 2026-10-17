"""Fixed LightGBM training and native-weight prediction."""

from __future__ import annotations

import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .io import read, sha, write
from .selection import binary_model, validate_fit


def train_native(x, y, folder, *, seed=20261016, quick=False, threads=2):
    """Called on official train by the guarded runner, or tiny synthetic tests."""
    import lightgbm as lgb

    validate_fit(x, y)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "model.json").exists():
        raise FileExistsError("Final model is immutable")
    x = np.asarray(x, np.float32)
    iterations = 5 if quick else 150
    started = time.monotonic()

    def one(j):
        if len(np.unique(y[:, j])) == 1:
            return (dict(constant=float(y[:, j].mean()), seed=seed + j), [0.0] * iterations, [])
        model = binary_model(seed + j, quick, threads)
        evaluation = {}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(
                x,
                y[:, j],
                eval_set=[(x, y[:, j])],
                eval_names=["training"],
                eval_metric="binary_logloss",
                callbacks=[lgb.record_evaluation(evaluation)],
            )
        path = folder / f"target_{j:02d}.txt"
        model.booster_.save_model(str(path))
        history = evaluation["training"]["binary_logloss"]
        if len(history) != iterations:
            raise ValueError("Missing training loss observations")
        return (
            dict(
                file=path.name,
                sha256=sha(path),
                seed=seed + j,
                trees=model.booster_.current_iteration(),
            ),
            history,
            [str(w.message) for w in caught],
        )

    with ThreadPoolExecutor(max_workers=1 if quick else 4) as pool:
        results = list(pool.map(one, range(y.shape[1])))
    losses = np.asarray([v[1] for v in results], dtype=np.float64)
    result = dict(
        format="LightGBM native text",
        family="lightgbm",
        fit_rows=len(x),
        width=x.shape[1],
        targets=y.shape[1],
        seed=seed,
        trees_requested=iterations,
        early_stopping=False,
        external_weights=False,
        models=[r[0] for r in results],
        training_loss=losses.mean(axis=0).tolist(),
        target_training_loss=losses.tolist(),
        warnings=sorted(set((w for r in results for w in r[2]))),
        parameters=binary_model(seed, quick, threads).get_params(),
        seconds=time.monotonic() - started,
    )
    write(folder / "model.json", result)
    p = predict_native(x, folder)
    if not np.isfinite(p).all():
        raise ValueError("Invalid native training replay")
    np.save(folder / "train_predictions.npy", p, allow_pickle=False)
    return result


def predict_native(x, folder, iteration=None):
    import lightgbm as lgb

    folder = Path(folder)
    meta = read(folder / "model.json")
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != meta["width"] or np.isinf(x).any():
        raise ValueError("Wrong feature dimensions/values")
    if iteration is not None and (
        not isinstance(iteration, int) or not 1 <= iteration <= meta["trees_requested"]
    ):
        raise ValueError("Invalid monitoring iteration")
    if len(meta["models"]) != meta["targets"]:
        raise ValueError("Incomplete native targets")
    values = []
    for row in meta["models"]:
        if "constant" in row:
            values.append(np.full(len(x), row["constant"]))
            continue
        if Path(row["file"]).name != row["file"]:
            raise ValueError("Unsafe native model path")
        path = folder / row["file"]
        if sha(path) != row["sha256"]:
            raise ValueError("Native model changed")
        model = lgb.Booster(model_file=str(path))
        values.append(model.predict(x, num_iteration=iteration, num_threads=2))
    result = np.stack(values, axis=1).astype(np.float64)
    if not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
        raise ValueError("Invalid probabilities")
    return result
