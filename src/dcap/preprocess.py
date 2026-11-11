"""Rebuild the frozen chip geometry from attributed inventories and public imagery."""

import gzip
import json
import shutil
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import requests
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from requests.adapters import HTTPAdapter
from shapely.geometry import box, mapping, shape
from shapely.ops import transform as transform_geometry
from shapely.strtree import STRtree
from urllib3.util.retry import Retry

from .paths import ROOT, data_root, sha256, write

BANDS = ("blue", "green", "red", "nir", "swir16", "swir22")
BAD_SCL = [0, 1, 3, 8, 9, 10, 11]
SESSION = requests.Session()
SESSION.mount(
    "https://",
    HTTPAdapter(
        max_retries=Retry(
            total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504]
        )
    ),
)


def metadata(name):
    return json.loads(gzip.decompress((ROOT / "metadata" / name).read_bytes()))


@lru_cache(maxsize=512)
def signed(href):
    response = SESSION.get(
        "https://planetarycomputer.microsoft.com/api/sas/v1/sign",
        params={"href": href},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["href"]


def read_grid(href, epsg, transform, nearest=False):
    with rasterio.open(href) as source:
        with WarpedVRT(
            source,
            crs=f"EPSG:{epsg}",
            transform=transform,
            width=256,
            height=256,
            resampling=Resampling.nearest if nearest else Resampling.bilinear,
            src_nodata=source.nodata if source.nodata is not None else 0,
            nodata=0,
        ) as vrt:
            raw = vrt.read(1, out_dtype="float32")
    return raw, raw != 0


def bounds_wgs84(row):
    transformer = Transformer.from_crs(int(row["epsg"]), 4326, always_xy=True)
    points = [
        transformer.transform(x, y)
        for x in [row["left"], row["right"]]
        for y in [row["bottom"], row["top"]]
    ]
    return [
        min(p[0] for p in points),
        min(p[1] for p in points),
        max(p[0] for p in points),
        max(p[1] for p in points),
    ]


def dem_for(row, transform):
    bounds = bounds_wgs84(row)
    response = SESSION.post(
        "https://earth-search.aws.element84.com/v1/search",
        json={"collections": ["cop-dem-glo-30"], "bbox": bounds, "limit": 100},
        timeout=120,
    )
    response.raise_for_status()
    items = response.json()["features"]
    chip = box(*bounds)
    dem = np.full((256, 256), np.nan, dtype="float32")
    for item in items:
        if not shape(item["geometry"]).intersects(chip):
            continue
        href = item["assets"]["data"]["href"].replace(
            "s3://copernicus-dem-30m/", "https://copernicus-dem-30m.s3.amazonaws.com/"
        )
        values, valid = read_grid(href, int(row["epsg"]), transform)
        fill = np.isnan(dem) & valid
        dem[fill] = values[fill]
    if np.isnan(dem).any():
        raise ValueError("Incomplete DEM coverage: " + row["sample_id"])
    return dem, [i["id"] for i in items]


def write_raster(path, array, row, transform, image=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(
        driver="GTiff",
        width=256,
        height=256,
        count=14 if image else 1,
        dtype="float32" if image else "uint8",
        crs=f'EPSG:{int(row["epsg"])}',
        transform=transform,
        compress="deflate",
        tiled=True,
        blockxsize=128,
        blockysize=128,
    )
    if image:
        profile["predictor"] = 3
    else:
        profile["nodata"] = 255
    temp = path.with_suffix(".tmp.tif")
    with rasterio.open(temp, "w", **profile) as dst:
        if image:
            dst.write(array.astype("float32"))
            for index, name in enumerate(
                [f"{date}_{band}" for date in ["pre", "post"] for band in BANDS]
                + ["dem_m", "slope_deg"],
                1,
            ):
                dst.set_band_description(index, name)
        else:
            dst.write(array.astype("uint8"), 1)
            dst.set_band_description(1, "landslide_mask")
    temp.replace(path)


def prepare(directory=None, limit=None, mask_only=False, imagery_root=None):
    root = data_root(directory)
    if limit is not None and limit < 1:
        raise ValueError("--limit must be positive")
    if mask_only and not imagery_root:
        raise ValueError(
            "--mask-only requires --imagery-root with existing study masks"
        )
    rows = pd.read_csv(ROOT / "metadata/chips.csv").to_dict("records")
    if limit:
        rows = rows[:limit]
    features = metadata("source_inventory.geojson.gz")["features"]
    scenes = metadata("source_scenes.json.gz")
    features = sorted(features, key=lambda f: f["properties"]["polygon_id"])
    indices = {}
    completed = []
    # A fresh destination makes partial and source-rebuilt data explicit.
    if (root / "images").exists() or (root / "masks").exists():
        raise FileExistsError("Use a fresh preparation destination")
    root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "data/DATA_NOTICES.txt", root / "DATA_NOTICES.txt")
    shutil.copytree(ROOT / "data/licenses", root / "licenses", dirs_exist_ok=True)
    for row in rows:
        epsg = int(row["epsg"])
        year = int(row["year"])
        key = (year, epsg)
        if key not in indices:
            transformer = Transformer.from_crs(4326, epsg, always_xy=True)
            geometries = [
                transform_geometry(transformer.transform, shape(f["geometry"]))
                for f in features
                if int(f["properties"]["year"]) == year
            ]
            indices[key] = (geometries, STRtree(geometries))
        geometries, tree = indices[key]
        transform = from_origin(row["left"], row["top"], 10, 10)
        dem_ids = []
        if mask_only:
            with rasterio.open(Path(imagery_root) / row["mask_path"]) as source:
                if source.transform != transform or source.crs.to_epsg() != epsg:
                    raise ValueError("Source grid mismatch")
                ignore = source.read(1) == 255
        else:
            arrays = []
            ignore = np.zeros((256, 256), dtype=bool)
            for date in ["pre", "post"]:
                scene = scenes[row[f"{date}_item_id"]]["assets"]
                scl, _ = read_grid(signed(scene["scl"]), epsg, transform, True)
                ignore |= np.isin(scl, BAD_SCL)
                for band in BANDS:
                    raw, valid = read_grid(signed(scene[band]), epsg, transform)
                    arrays.append(raw * np.float32(0.0001))
                    ignore |= ~valid
            dem, dem_ids = dem_for(row, transform)
            slope = np.degrees(
                np.arctan(np.hypot(*np.gradient(dem, 10.0, 10.0)))
            ).astype("float32")
            write_raster(
                root / row["image_path"],
                np.stack([*arrays, dem, slope]),
                row,
                transform,
                True,
            )
        chosen = tree.query(
            box(row["left"], row["bottom"], row["right"], row["top"]),
            predicate="intersects",
        )
        mask = rasterize(
            [(mapping(geometries[int(i)]), 1) for i in chosen],
            out_shape=(256, 256),
            transform=transform,
            fill=0,
            dtype="uint8",
            all_touched=True,
        )
        mask[ignore] = 255
        write_raster(root / row["mask_path"], mask, row, transform)
        completed.append(
            dict(
                sample_id=row["sample_id"],
                positive_pixels=int((mask == 1).sum()),
                expected_positive_pixels=int(row["positive_pixels"]),
                dem_tiles=dem_ids,
            )
        )
        if len(completed) % 25 == 0:
            print(f"Prepared {len(completed)}/{len(rows)} chips", flush=True)
    write(
        root / "preparation.json",
        dict(
            mask_only=mask_only,
            source_inventory_sha256=sha256(
                ROOT / "metadata/source_inventory.geojson.gz"
            ),
            chips=completed,
        ),
    )
    return dict(
        prepared=len(completed),
        directory=str(root),
        mask_only=mask_only,
        positive_counts_match=all(
            r["positive_pixels"] == r["expected_positive_pixels"] for r in completed
        ),
    )
