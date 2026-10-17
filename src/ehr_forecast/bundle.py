"""Portable model archives with explicit contents and integrity checks."""

import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np

from .inference import verify_bundle
from .io import read, sha, write


def unpack(archive, destination):
    """Reject unsafe archives before creating any output."""
    archive, destination = Path(archive), Path(destination)
    if destination.exists():
        verify_bundle(destination)
        if read(destination / "archive.json")["sha256"] != sha(archive):
            raise ValueError("Cached model came from a different archive")
        return destination
    with zipfile.ZipFile(archive) as source:
        members = source.infolist()
        names = [item.filename for item in members]
        if len(names) != len(set(names)) or sum(item.file_size for item in members) > 100_000_000:
            raise ValueError("Unexpected model archive size or duplicate members")
        for item in members:
            path = PurePosixPath(item.filename)
            if (
                item.is_dir()
                or item.orig_filename != item.filename
                or path.is_absolute()
                or ".." in path.parts
                or ":" in item.filename
                or "\\" in item.filename
                or str(path) != item.filename
                or stat.S_ISLNK(item.external_attr >> 16)
            ):
                raise ValueError("Unsafe model archive member")
        import json

        manifest = json.loads(source.read("manifest.json"))
        if set(names) != {"manifest.json", *manifest["files"]}:
            raise ValueError("Unexpected model archive contents")
        destination.mkdir(parents=True)
        for item in members:
            path = destination / item.filename
            path.parent.mkdir(parents=True, exist_ok=True)
            with source.open(item) as src, path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    verify_bundle(destination)
    write(destination / "archive.json", {"sha256": sha(archive)})
    return destination


def export(prepared, trained, output):
    """Combine local preprocessing and native fitted weights; exclude patient rows."""
    prepared, trained, output = map(Path, (prepared, trained, output))
    if output.exists():
        raise FileExistsError("Choose a new model archive path")
    metadata = read(prepared / "training/metadata.json")
    representation = read(prepared / "preprocessing/representation.json")
    native = read(trained / "model/model.json")
    if (
        metadata["fit_patients"] != 2791
        or not metadata["official_train_only"]
        or metadata["validation_or_test_fitted"]
        or representation["fit_count"] != 2791
        or representation["fit_hash"] != metadata["fit_patient_hash"]
        or native["fit_rows"] != 2791
        or native["width"] != 512
        or native["targets"] != 40
        or native["trees_requested"] != 150
        or native["early_stopping"]
        or native["external_weights"]
        or len(native["models"]) != 40
        or len(metadata["target_codes"]) != 40
    ):
        raise ValueError("Model and preprocessing scope do not match")
    columns = np.load(trained / "model/columns.npy", allow_pickle=False)
    if (
        columns.ndim != 1
        or columns.dtype.kind not in "iu"
        or len(columns) != 512
        or len(set(columns)) != 512
        or columns.min() < 0
        or columns.max() >= len(representation["catalog"])
    ):
        raise ValueError("Invalid selected feature indices")
    bundle = {
        "version": 19,
        "model": "LightGBM, 40 independent binary targets",
        "features": 512,
        "fit_patients": 2791,
        "candidate_features": len(representation["catalog"]),
        "quality_features": metadata["quality_features"],
        "official_train_only": True,
        "fit_patient_hash": metadata["fit_patient_hash"],
        "target_codes": metadata["target_codes"],
        "seed": native["seed"],
        "trees_requested": native["trees_requested"],
        "external_data_or_weights": False,
        "prior_rule": "Zero probability for a condition diagnosed before the anchor.",
        "model_use": "Synthetic EHR research; not validated for clinical use.",
    }
    import json

    contents = {
        "bundle.json": json.dumps(bundle, indent=2).encode(),
        "preprocessing/representation.json": (
            prepared / "preprocessing/representation.json"
        ).read_bytes(),
        "preprocessing/selected_features.json": json.dumps(
            [representation["catalog"][int(i)] for i in columns], indent=2
        ).encode(),
    }
    for name in [
        "model.json",
        "columns.npy",
        *[r["file"] for r in native["models"] if "file" in r],
    ]:
        path = trained / "model" / name
        if path.name != name:
            raise ValueError("Unsafe native model filename")
        contents["model/" + name] = path.read_bytes()
    for row in native["models"]:
        if "file" in row and sha(trained / "model" / row["file"]) != row["sha256"]:
            raise ValueError("Native model changed before export")
    import hashlib

    manifest = {
        "training_data_excluded": True,
        "raw_data_excluded": True,
        "credentials_excluded": True,
        "files": {name: hashlib.sha256(content).hexdigest() for name, content in contents.items()},
    }
    contents["manifest.json"] = json.dumps(manifest, indent=2).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(contents.items()):
            archive.writestr(name, content)
    return {"file": output.name, "sha256": sha(output), "members": len(contents)}
