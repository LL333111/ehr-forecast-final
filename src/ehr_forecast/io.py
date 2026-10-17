"""Small file utilities and text-payload validation."""

from __future__ import annotations

import hashlib
import json
import math
import time as clock
from pathlib import Path


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2)
        + """
""",
        encoding="utf-8",
    )
    for attempt in range(8):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            clock.sleep(0.05 * (attempt + 1))


def finite(value):
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (ValueError, TypeError):
        return None


def ordinary_text(value):
    """Unknown opaque payloads fail closed, instead of becoming vocabulary IDs."""
    value = str(value or "").strip()
    if (
        len(value) > 512
        or value.startswith(("iVBOR", "/9j/", "data:", "{", "[", "<"))
        or (len(value.split()) > 8 and all((finite(x) is not None for x in value.split())))
    ):
        raise ValueError("Unreviewed non-language payload")
    return value
