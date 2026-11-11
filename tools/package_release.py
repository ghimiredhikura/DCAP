#!/usr/bin/env python3
"""Build the source manifest and ZIP; omit installed assets and generated files."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOP = {"src", "configs", "metadata", "results", "data", "docs", "tests", "tools"}
FILES = {
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "pyproject.toml",
    "requirements-tested.txt",
    ".gitignore",
    ".gitattributes",
}


def included(path):
    rel = path.relative_to(ROOT)
    if not path.is_file() or path.is_symlink():
        return False
    if any(p.startswith(".") for p in rel.parts) and len(rel.parts) > 1:
        return False
    if "__pycache__" in rel.parts or any(p.endswith(".egg-info") for p in rel.parts):
        return False
    if path.suffix in {".pyc", ".pt", ".zip", ".log", ".part", ".tmp"}:
        return False
    if (
        rel.parts[0] == "data"
        and len(rel.parts) > 1
        and rel.parts[1] not in {"example.npz", "DATA_NOTICES.txt", "licenses"}
    ):
        return False
    return rel.parts[0] in TOP or str(rel) in FILES


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output")
    a = p.parse_args()
    manifest = ROOT / "metadata/files.json"
    entries = sorted(f for f in ROOT.rglob("*") if included(f) and f != manifest)
    manifest.write_text(
        json.dumps({str(f.relative_to(ROOT)): digest(f) for f in entries}, indent=2)
        + "\n"
    )
    if a.output:
        target = Path(a.output).resolve()
        if target.is_relative_to(ROOT):
            raise ValueError("Archive output must be outside the source checkout")
        target.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(
            target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for f in sorted([*entries, manifest]):
                archive.write(f, "dcap/" + str(f.relative_to(ROOT)))
        target.with_suffix(target.suffix + ".sha256").write_text(
            digest(target) + "  " + target.name + "\n"
        )
    print(json.dumps(dict(files=len(entries), archive=a.output)))


if __name__ == "__main__":
    main()
