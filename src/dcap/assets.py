"""Hash-checked downloads of the separate data/model assets."""

import urllib.request
from pathlib import Path

from .paths import ROOT, read, sha256


def download(repository, tag, kind, output):
    if repository.count("/") != 1 or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-./"
        for c in repository
    ):
        raise ValueError("Use OWNER/REPOSITORY")
    records = (
        read(ROOT / "metadata/data_assets.json")["archives"]
        if kind == "data"
        else read(ROOT / "metadata/models.json")["models"]
    )
    notice = read(ROOT / "metadata/notices.json")[kind]
    records = [*records, notice]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for r in records:
        target = output / r["asset"]
        if target.exists():
            if sha256(target) == r["sha256"]:
                continue
            raise FileExistsError("Existing file differs: " + str(target))
        temporary = target.with_suffix(target.suffix + ".part")
        url = f'https://github.com/{repository}/releases/download/{tag}/{r["asset"]}'
        urllib.request.urlretrieve(url, temporary)
        if sha256(temporary) != r["sha256"]:
            raise ValueError("Downloaded asset checksum mismatch: " + target.name)
        temporary.rename(target)
    return dict(downloaded=len(records), directory=str(output))
