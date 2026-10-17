"""One fixed final recipe; no validation-based selection or checkpoint search."""

import csv
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .data import (
    build_raw_cache,
    feature_context,
    fit_representation,
    id_hash,
    references,
    train_labels,
    transform,
)
from .engineering import fit_quality
from .io import read, sha, write
from .model import train_native
from .selection import select_features, validate_fit

ARRAYS = ("train.npy", "labels.npy", "prior.npy", "quality.npy")


def prepare(raw, output):
    """Fit dictionaries on official train locally and save a numeric training package."""
    raw, output = Path(raw), Path(output)
    if output.exists():
        raise FileExistsError("Choose a fresh preparation directory")
    task = (raw / "ReadMe.md").read_text(encoding="utf-8")
    for rule in (
        "Do not train on validation or test",
        "Do not use external datasets or pretrained weights",
    ):
        if rule not in task:
            raise ValueError("Task rules changed; review the supplied README")
    _, roles, targets = references(raw)
    if {k: len(v) for k, v in roles.items()} != {"train": 2791, "val": 365, "test": 358} or len(
        targets
    ) != 40:
        raise ValueError("This recipe expects the supplied official task splits")
    cache = output / "cache"
    build_raw_cache(raw, cache)
    # Keep the accepted row-local extraction context; fitting still uses train only.
    with feature_context(raw, cache) as context:
        representation = fit_representation(context, roles["train"])
        x = transform(context, representation, roles["train"])
    codes = [row["CODE"] for row in targets]
    y, prior = train_labels(raw, cache, roles["train"], codes)
    quality, exclusions = fit_quality(x)
    training = output / "training"
    training.mkdir(parents=True)
    for name, array in zip(ARRAYS, (x, y, prior, quality)):
        np.save(training / name, array, allow_pickle=False)
    metadata = {
        "fit_patients": len(x),
        "target_codes": codes,
        "features": x.shape[1],
        "quality_features": len(quality),
        "official_train_only": True,
        "fit_patient_hash": id_hash(roles["train"]),
        "validation_or_test_fitted": False,
        "seed": 20261016,
        "trees": 150,
    }
    write(training / "metadata.json", metadata)
    write(
        training / "manifest.json",
        {
            "files": {name: sha(training / name) for name in (*ARRAYS, "metadata.json")},
        },
    )
    write(output / "preprocessing/representation.json", representation)
    write(output / "preprocessing/quality_audit.json", exclusions)
    return metadata


def cloud_guard():
    if not os.environ.get("RUNPOD_POD_ID"):
        raise RuntimeError("Formal fitting is restricted to the authorized RunPod host")


def load_training(prepared):
    prepared = Path(prepared)
    manifest = read(prepared / "manifest.json")
    if set(manifest["files"]) != {*ARRAYS, "metadata.json"}:
        raise ValueError("Unexpected training package members")
    if {p.name for p in prepared.iterdir()} != {*ARRAYS, "metadata.json", "manifest.json"}:
        raise ValueError("Only the numeric training package may be transferred")
    for name, digest in manifest["files"].items():
        if sha(prepared / name) != digest:
            raise ValueError("Training package changed")
    metadata = read(prepared / "metadata.json")
    if (
        metadata["fit_patients"] != 2791
        or not metadata["official_train_only"]
        or metadata["validation_or_test_fitted"]
        or len(metadata["target_codes"]) != 40
        or metadata["seed"] != 20261016
        or metadata["trees"] != 150
    ):
        raise ValueError("Invalid official training scope")
    x, y, prior, quality = [np.load(prepared / name, allow_pickle=False) for name in ARRAYS]
    validate_fit(x, y)
    if (
        x.shape != (2791, metadata["features"])
        or y.shape != (2791, 40)
        or prior.shape != y.shape
        or not np.isin(prior, [0, 1]).all()
        or np.any(y[prior == 1])
        or len(quality) != metadata["quality_features"]
        or quality.ndim != 1
        or quality.dtype.kind not in "iu"
        or len(quality) < 512
        or len(set(quality)) != len(quality)
        or quality.min() < 0
        or quality.max() >= x.shape[1]
    ):
        raise ValueError("Invalid training array dimensions or prior mask")
    return x, y, quality


def train(prepared, output):
    """Run the frozen recipe and save actual offline W&B training records."""
    cloud_guard()
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a new training output directory")
    x, y, quality = load_training(prepared)
    import wandb

    output.mkdir(parents=True)
    run = wandb.init(
        project="ehr-forecast",
        name="final-lightgbm512",
        mode="offline",
        dir=str(output),
        config={"seed": 20261016, "trees": 150, "features": 512, "fit_patients": 2791},
        settings=wandb.Settings(disable_code=True, disable_git=True, init_timeout=60),
    )
    try:
        select_features(x, y, quality, output / "selector", threads=2)
        with np.load(output / "selector/selection.npz") as selection:
            columns = selection["compact"]
        if len(columns) != 512:
            raise ValueError("Final selector did not produce 512 features")
        (output / "model").mkdir()
        np.save(output / "model/columns.npy", columns, allow_pickle=False)
        model = train_native(x[:, columns], y, output / "model", seed=20261016, threads=2)
        with (output / "training_curve.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["iteration", "training_binary_logloss"])
            writer.writeheader()
            for step, loss in enumerate(model["training_loss"], 1):
                row = {"iteration": step, "training_binary_logloss": loss}
                writer.writerow(row)
                run.log(row, step=step)
        result = {
            "complete": True,
            "utc": datetime.now(timezone.utc).isoformat(),
            "fit_patients": 2791,
            "targets": 40,
            "features": 512,
            "validation_or_test_fitted": False,
            "wandb_run_id": run.id,
            "wandb_mode": "offline",
            "model_sha256": sha(output / "model/model.json"),
        }
        write(output / "training_complete.json", result)
        return result
    finally:
        run.finish()
