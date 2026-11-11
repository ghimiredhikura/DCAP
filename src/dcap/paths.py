"""Repository paths and small reproducibility utilities."""

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("DCAP_ROOT", Path(__file__).resolve().parents[2])).resolve()
if (
    not (ROOT / "configs/study.json").exists()
    and (Path.cwd() / "configs/study.json").exists()
):
    ROOT = Path.cwd().resolve()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def data_root(value=None):
    return Path(value or os.environ.get("DCAP_DATA", ROOT / "data")).resolve()


def model_root(value=None):
    return Path(value or os.environ.get("DCAP_MODELS", ROOT / "models")).resolve()


def study():
    return read(ROOT / "configs/study.json")
