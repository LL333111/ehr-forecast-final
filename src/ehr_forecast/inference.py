"""Validated raw-input prediction with no fitting."""

from __future__ import annotations

import csv
from pathlib import Path, PurePosixPath

import numpy as np

from .data import build_raw_cache, feature_context, priors, references, transform
from .io import read, sha, write
from .model import predict_native


def verify_bundle(folder):
    folder = Path(folder)
    manifest = read(folder / "manifest.json")
    if not manifest.get("training_data_excluded") or not manifest.get("raw_data_excluded"):
        raise ValueError("Invalid portable model manifest")
    for name, digest in manifest["files"].items():
        p = PurePosixPath(name)
        if p.is_absolute() or ".." in p.parts or ":" in name or ("\\" in name) or (str(p) != name):
            raise ValueError("Unsafe model member path")
        path = folder / name
        if (
            not path.resolve().is_relative_to(folder.resolve())
            or path.is_symlink()
            or sha(path) != digest
        ):
            raise ValueError("Changed model bundle: " + name)
    for required in [
        "bundle.json",
        "model/model.json",
        "model/columns.npy",
        "preprocessing/representation.json",
    ]:
        if required not in manifest["files"]:
            raise ValueError("Missing portable model component")
    info = read(folder / "bundle.json")
    if not info["official_train_only"] or info["fit_patients"] != 2791 or info["features"] != 512:
        raise ValueError("Wrong final training scope")
    return info


def write_predictions(path, ids, codes, probabilities, prior):
    """Validate the exact task format and apply the first-diagnosis prior mask."""
    path = Path(path)
    p = np.asarray(probabilities, dtype=np.float64)
    prior = np.asarray(prior)
    if (
        not ids
        or len(ids) != len(set(ids))
        or any((not isinstance(v, str) or not v for v in ids))
        or (not codes)
        or (len(codes) != len(set(codes)))
        or ("patient_id" in codes)
        or any((not isinstance(v, str) or not v for v in codes))
        or (p.shape != (len(ids), len(codes)))
        or (prior.shape != p.shape)
        or (not np.isfinite(p).all())
        or np.any((p < 0) | (p > 1))
        or (not np.isin(prior, [0, 1]).all())
    ):
        raise ValueError("Invalid submission probabilities/order")
    if path.exists():
        raise FileExistsError("Prediction output is immutable; choose a new path")
    p = np.where(prior == 1, 0.0, p)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", *codes])
        w.writerows([[pid, *[format(v, ".17g") for v in row]] for pid, row in zip(ids, p)])
    return p


def predict_bundle(raw, folder, cache, output, split="test"):
    if split not in ["val", "test"]:
        raise ValueError("Portable entry point only scores held-out splits")
    folder = Path(folder)
    info = verify_bundle(folder)
    _, roles, targets = references(raw)
    codes = [t["CODE"] for t in targets]
    if codes != info["target_codes"]:
        raise ValueError("Task target order differs from trained model")
    protocol = read(folder / "preprocessing" / "representation.json")
    if protocol["fit_count"] != 2791 or protocol["fit_hash"] != info["fit_patient_hash"]:
        raise ValueError("Preprocessing training scope mismatch")
    columns = np.load(folder / "model" / "columns.npy", allow_pickle=False)
    if (
        columns.ndim != 1
        or columns.dtype.kind not in "iu"
        or len(columns) != 512
        or (len(set(columns)) != 512)
        or (columns.min() < 0)
        or (columns.max() >= len(protocol["catalog"]))
    ):
        raise ValueError("Invalid final feature order")
    raw_report = build_raw_cache(raw, cache)
    with feature_context(raw, cache, roles[split]) as ctx:
        x = transform(ctx, protocol, roles[split])
    if x.shape != (len(roles[split]), len(protocol["catalog"])):
        raise ValueError("Raw feature width mismatch")
    p = predict_native(x[:, columns], folder / "model")
    prior = priors(cache, roles[split], codes)
    write_predictions(output, roles[split], codes, p, prior)
    receipt = dict(
        complete=True,
        split=split,
        patients=len(p),
        targets=len(codes),
        features=512,
        fitting_performed=False,
        labels_read=False,
        raw_data_modified=False,
        model_manifest_sha256=sha(folder / "manifest.json"),
        prediction_sha256=sha(output),
        prior_zero_entries=int(prior.sum()),
        raw_cache_verified=raw_report["complete"],
    )
    write(str(output) + ".receipt.json", receipt)
    return receipt
