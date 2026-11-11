"""Install and validate the fixed 1,320-chip study dataset."""

import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from .paths import ROOT, data_root, read, sha256


def split_rows(split=None):
    frame = pd.read_csv(ROOT / "metadata/split.csv")
    return frame if split is None else frame[frame.split == split]


def load_cache(directory=None, splits=("train", "validation"), limit=None):
    rows = split_rows()
    selected = rows[rows.split.isin(splits)].to_dict("records")
    if limit:
        selected = selected[:limit]
    root = data_root(directory)
    with ThreadPoolExecutor(max_workers=8) as pool:
        cache = dict(pool.map(lambda row: read_chip(root, row), selected))
    return cache, rows, read(ROOT / "metadata/normalization.json")


def install(source, directory=None):
    root = data_root(directory)
    catalog = read(ROOT / "metadata/data_assets.json")
    for record in catalog["archives"]:
        archive = Path(source) / record["asset"]
        if sha256(archive) != record["sha256"]:
            raise ValueError(f"Asset checksum mismatch: {archive.name}")
        with zipfile.ZipFile(archive) as zipped:
            for entry in zipped.infolist():
                target = (root / entry.filename).resolve()
                if not target.is_relative_to(root):
                    raise ValueError("Unsafe archive path")
                if target.exists():
                    # Existing rasters are never silently replaced.
                    expected = {**catalog["files"], **catalog.get("notices", {})}[
                        entry.filename
                    ]["sha256"]
                    if sha256(target) != expected:
                        raise FileExistsError(f"Existing file differs: {target}")
                    continue
                zipped.extract(entry, root)
    return verify(root)


def verify(directory=None, full_hash=True):
    root = data_root(directory)
    catalog = read(ROOT / "metadata/data_assets.json")
    for name, record in catalog["files"].items():
        path = root / name
        if not path.is_file() or path.stat().st_size != record["bytes"]:
            raise ValueError(f"Missing or wrong-size raster: {path}")
        if full_hash and sha256(path) != record["sha256"]:
            raise ValueError(f"Changed raster: {path}")
    frame = split_rows()
    assert frame.sample_id.is_unique
    assert frame.groupby("split").size().to_dict() == {
        "test": 198,
        "train": 990,
        "validation": 132,
    }
    assert frame.groupby("spatial_group").split.nunique().max() == 1
    return dict(
        passed=True,
        chips=len(frame),
        raster_files=len(catalog["files"]),
        split_counts=frame.groupby("split").size().to_dict(),
    )


# Deterministic raster reading, input construction and training augmentation.

import zlib

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

CROP_SIZE = 224


def read_chip(dataset_dir, row):
    with rasterio.open(dataset_dir / row["image_path"]) as image_source:
        raw = image_source.read().astype(np.float32)
        image_transform, image_crs = (image_source.transform, image_source.crs)
        names = list(image_source.descriptions)
    with rasterio.open(dataset_dir / row["mask_path"]) as mask_source:
        target = mask_source.read(1)
        if image_transform != mask_source.transform or image_crs != mask_source.crs:
            raise ValueError(f"Image/mask alignment mismatch: {row['sample_id']}")
    expected = [
        f"{date}_{band}"
        for date in ("pre", "post")
        for band in ("blue", "green", "red", "nir", "swir16", "swir22")
    ] + ["dem_m", "slope_deg"]
    if raw.shape != (14, 256, 256) or target.shape != (256, 256) or names != expected:
        raise ValueError(f"Shape/band mismatch: {row['sample_id']}: {names}")
    if (
        not np.isin(target, [0, 1, 255]).all()
        or not np.isfinite(raw[:, target != 255]).all()
    ):
        raise ValueError(f"Invalid chip values: {row['sample_id']}")
    if int((target == 1).sum()) != int(row["positive_pixels"]):
        raise ValueError(f"Mask/source positive count mismatch: {row['sample_id']}")
    return (row["sample_id"], (raw, target))


class CachedDataset(TorchDataset):
    def __init__(self, cache, rows, norm, augment=False, seed=42):
        self.cache, self.rows, self.augment, self.seed = (cache, rows, augment, seed)
        self.epoch = 0
        self.mean = torch.tensor(norm["mean"])[:, None, None]
        self.std = torch.tensor(norm["standard_deviation"])[:, None, None]
        self.cm = torch.tensor(norm["reflectance_change_mean"])[:, None, None]
        self.cs = torch.tensor(norm["reflectance_change_standard_deviation"])[
            :, None, None
        ]

    def __len__(self):
        return len(self.rows)

    def rng(self, sid, salt):
        return torch.Generator().manual_seed(
            zlib.crc32(sid.encode()) + self.seed * 1000003 + self.epoch * 1000033 + salt
        )

    def __getitem__(self, i):
        sid = self.rows[i]["sample_id"]
        raw, target = self.cache[sid]
        raw = torch.from_numpy(raw).clone()
        target = torch.from_numpy(target.astype(np.int64))
        if self.augment:
            rng = self.rng(sid, 11)
            gains = torch.rand((12, 1, 1), generator=rng) * 0.1 + 0.95
            offsets = torch.rand((12, 1, 1), generator=rng) * 0.01 - 0.005
            noise = torch.randn((12, 256, 256), generator=self.rng(sid, 23)) * 0.001
            raw[:12] = raw[:12] * gains + offsets + noise
        standardized = (raw - self.mean) / self.std
        image = torch.cat(
            (
                standardized[:12],
                (raw[6:12] - raw[:6] - self.cm) / self.cs,
                standardized[12:14],
            )
        )
        valid = target != 255
        image[:, ~valid] = 0
        image = torch.cat((image, valid[None].float()))
        if self.augment:
            rng = self.rng(sid, 37)
            for dim in (-1, -2):
                if torch.rand((), generator=rng) < 0.5:
                    image, target = (image.flip(dim), target.flip(dim))
            k = int(torch.randint(0, 4, (), generator=rng))
            image, target = (
                torch.rot90(image, k, (-2, -1)),
                torch.rot90(target, k, (-2, -1)),
            )
            rng = self.rng(sid, 47)
            crop = CROP_SIZE
            loc = torch.nonzero(target == 1)
            if len(loc) and torch.rand((), generator=rng) < 0.9:
                y, x = loc[int(torch.randint(len(loc), (), generator=rng))].tolist()
                top = int(
                    torch.randint(
                        max(0, y - crop + 1), min(y, 256 - crop) + 1, (), generator=rng
                    )
                )
                left = int(
                    torch.randint(
                        max(0, x - crop + 1), min(x, 256 - crop) + 1, (), generator=rng
                    )
                )
            else:
                top, left = [
                    int(torch.randint(0, 257 - crop, (), generator=rng))
                    for _ in range(2)
                ]
            image, target = (
                image[:, top : top + crop, left : left + crop],
                target[top : top + crop, left : left + crop],
            )
        return (image, target, sid)


def loader(dataset, shuffle=False, seed=42):
    return DataLoader(
        dataset,
        batch_size=4,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        generator=torch.Generator().manual_seed(seed),
    )


def input_indices(mode):
    base = mode.replace("_no_terrain", "")
    spectral = {
        "post_only": list(range(6, 12)),
        "raw_bitemporal": list(range(12)),
        "reflectance_change": list(range(18)),
    }[base]
    return spectral + ([] if mode.endswith("_no_terrain") else [18, 19]) + [20]


class Dataset(CachedDataset):
    def __init__(
        self,
        cache,
        rows,
        norm,
        mode="reflectance_change",
        recipe="enhanced",
        training=False,
        seed=42,
    ):
        super().__init__(
            cache, rows, norm, augment=training and recipe == "enhanced", seed=seed
        )
        self.mode, self.recipe, self.training = (mode, recipe, training)

    def __getitem__(self, i):
        image, target, sid = super().__getitem__(i)
        if self.training and self.recipe == "standard":
            rng = self.rng(sid, 37)
            for dim in (-1, -2):
                if torch.rand((), generator=rng) < 0.5:
                    image, target = (image.flip(dim), target.flip(dim))
            k = int(torch.randint(0, 4, (), generator=rng))
            image, target = (
                torch.rot90(image, k, (-2, -1)),
                torch.rot90(target, k, (-2, -1)),
            )
        return (image[input_indices(self.mode)], target, sid)
