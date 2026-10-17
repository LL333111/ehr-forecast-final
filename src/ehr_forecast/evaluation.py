"""Score locked official-validation predictions without fitting or checkpoint selection."""

import csv
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .data import build_raw_cache, priors, references
from .io import read, sha, write
from .labels import labels, shift_years
from .schema import EVENT_HEADERS


def metrics(y, probabilities, prior):
    if (
        y.ndim != 2
        or y.shape != probabilities.shape
        or y.shape != prior.shape
        or not np.isin(y, [0, 1]).all()
        or not np.isin(prior, [0, 1]).all()
        or np.any(y[prior == 1])
        or not np.isfinite(probabilities).all()
        or np.any((probabilities < 0) | (probabilities > 1))
    ):
        raise ValueError("Invalid labels, probabilities or prior alignment")
    p = np.where(prior == 1, 0.0, probabilities)
    ap = [
        float(average_precision_score(y[:, j], p[:, j])) if y[:, j].any() else 0.0
        for j in range(y.shape[1])
    ]
    auc = [
        float(roc_auc_score(y[:, j], p[:, j])) if len(np.unique(y[:, j])) == 2 else None
        for j in range(y.shape[1])
    ]
    defined = [a for a in auc if a is not None]
    return {
        "macro_ap": float(np.mean(ap)),
        "macro_auroc": float(np.mean(defined)) if defined else None,
        "ap": ap,
        "auroc": auc,
        "positives": y.sum(axis=0).astype(int).tolist(),
        "targets": len(ap),
    }


def evaluate(raw, cache, prediction, output):
    """Validate the prediction receipt before opening validation outcomes."""
    raw, cache, prediction, output = map(Path, (raw, cache, prediction, output))
    if output.exists():
        raise FileExistsError("Evaluation output is immutable; choose a new path")
    receipt = read(str(prediction) + ".receipt.json")
    if (
        not receipt["complete"]
        or receipt["split"] != "val"
        or receipt["labels_read"]
        or receipt["fitting_performed"]
        or sha(prediction) != receipt["prediction_sha256"]
    ):
        raise ValueError("Validation predictions must be locked before outcomes are read")
    _, roles, targets = references(raw)
    ids, codes = roles["val"], [t["CODE"] for t in targets]
    with prediction.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    if rows[0] != ["patient_id", *codes] or [r[0] for r in rows[1:]] != ids:
        raise ValueError("Validation patient or target order differs from the task")
    p = np.array([r[1:] for r in rows[1:]], dtype=float)
    if p.shape != (len(ids), len(codes)) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid validation probabilities")
    build_raw_cache(raw, cache)
    allowed, first = set(ids), {}
    with (raw / "train_val/conditions.csv").open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(EVENT_HEADERS["conditions"]):
            raise ValueError("Unreviewed validation label schema")
        for row in reader:
            if row["PATIENT"] not in allowed or row["CODE"] not in codes:
                continue
            key, onset = (row["PATIENT"], row["CODE"]), date.fromisoformat(row["START"][:10])
            if key not in first or onset < first[key]:
                first[key] = onset
    with duckdb.connect(str(cache / "database/ehr.duckdb"), read_only=True) as con:
        dates = {
            pid: (anchor, shift_years(anchor, 5))
            for pid, anchor in con.execute(
                "SELECT patient_id,anchor_date FROM clinical.patients WHERE split='val'"
            ).fetchall()
        }
    y, prior = labels(first, ids, codes, dates)
    if not np.array_equal(prior, priors(cache, ids, codes)):
        raise ValueError("Validation prior mismatch")
    result = metrics(y, p, prior)
    result.update(
        patients=len(ids),
        official_split="val",
        checkpoint_selection=False,
        fitting_performed=False,
        test_metrics_available=False,
        independent_virgin_holdout=False,
        prediction_sha256=receipt["prediction_sha256"],
    )
    if sha(prediction) != receipt["prediction_sha256"]:
        raise ValueError("Predictions changed during evaluation")
    write(output, result)
    return result
