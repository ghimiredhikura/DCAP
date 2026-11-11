"""Hash-checked downloads of the separate data/model assets."""

import urllib.request
from pathlib import Path

from .paths import ROOT, read, sha256


def records_for(kind, group="main"):
    notices = read(ROOT / "metadata/notices.json")
    if kind == "data":
        return [*read(ROOT / "metadata/data_assets.json")["archives"], notices["data"]]
    records = [
        m
        for m in read(ROOT / "metadata/models.json")["models"]
        if group == "all" or m["asset_group"] == group
    ]
    if group in ("main", "all"):
        records.append(notices["models"])
    if group in ("comparators", "all"):
        records.extend([notices["comparator_models"], notices["segformer_license"]])
    return records


def download(repository, tag, kind, output, group="main"):
    if repository.count("/") != 1 or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-./"
        for c in repository
    ):
        raise ValueError("Use OWNER/REPOSITORY")
    records = records_for(kind, group)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for r in records:
        target = output / r["asset"]
        if target.exists():
            if sha256(target) == r["sha256"]:
                continue
            raise FileExistsError("Existing file differs: " + str(target))
        temporary = target.with_suffix(target.suffix + ".part")
        release = r.get("release", tag or "v1.1.0")
        url = (
            f'https://github.com/{repository}/releases/download/{release}/{r["asset"]}'
        )
        urllib.request.urlretrieve(url, temporary)
        if sha256(temporary) != r["sha256"]:
            raise ValueError("Downloaded asset checksum mismatch: " + target.name)
        temporary.rename(target)
    return dict(downloaded=len(records), directory=str(output))
