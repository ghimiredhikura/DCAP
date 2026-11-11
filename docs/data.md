# Data preparation and source terms

For a folder summary, see [`data/README.md`](../data/README.md).

## 1. Install the processed dataset

Use the three archives from the [v1.1.0 release](https://github.com/ghimiredhikura/DCAP/releases/tag/v1.1.0):

```bash
python -m dcap download --repository ghimiredhikura/DCAP --tag v1.1.0 --kind data --output downloads/data
python -m dcap data install --source downloads/data
python -m dcap data verify
```

- **Contents:** 1,320 georeferenced image/mask pairs; about 3.43 GiB compressed.
- **Checks:** archive SHA-256, safe extraction paths and all 2,640 raster hashes. Different existing files are preserved.
- **Storage:** allow roughly 8 GB for archives and extracted files; no further conversion is needed.
- **Offline transfer:** pass the archive folder to `data install --source` instead of downloading.

### Stored data and model inputs

| Item | Representation |
|---|---|
| Image | Losslessly compressed 14-band float32 GeoTIFF |
| Spectral bands | Blue, green, red, NIR, SWIR1, SWIR2, before and after the monsoon |
| Terrain | Elevation and slope |
| Mask | 0 background; 1 landslide; 255 ignored |
| Full model input | 12 reflectances + 6 post-minus-pre changes + 2 terrain channels + 1 validity indicator |
| Normalization | Fitting chips only; invalid features zeroed before adding validity |

[`chips.csv`](../metadata/chips.csv) records scenes, polygons, grid bounds and positive counts. The loader also supports the channel combinations in the factorial design.

## 2. Obtain the source data (optional)

The included compressed metadata preserves inventory geometry, identifiers, selected scene URLs and quality decisions. The accepted set has **2,063 polygons**: all four two-date polygon/context bad-pixel fractions must be ≤0.20. Only accepted geometries contribute to labels.

| Source | Access and role |
|---|---|
| Monsoon inventory | [BGS accession 166966](https://webapps.bgs.ac.uk/services/ngdc/accessions/index.html#item166966); the [metadata API](https://webservices.bgs.ac.uk/accessions/item/166966) gives license/attribution |
| Western Nepal inventories | [Zenodo 7970874](https://doi.org/10.5281/zenodo.7970874), Burrows, Marc and Andermann; identifiers connect polygons to the chip manifest |
| Sentinel-2 L2A | [Planetary Computer](https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a); fixed scene choices in `source_scenes.json.gz` and `chips.csv`; fresh public signatures obtained when reading |
| Copernicus DEM GLO-30 | [Earth Search STAC](https://earth-search.aws.element84.com/v1/collections/cop-dem-glo-30); resampled to the 10 m chip grid, without creating native 10 m elevation detail |

No private account token or temporary signed URL is packaged.

## 3. Rebuild chips (optional)

Try one chip, then rebuild the fixed set in a fresh destination:

```bash
python -m dcap data prepare --data-root generated/source_sample --limit 1
python -m dcap data prepare --data-root generated/source_data
```

| Preparation step | Rule |
|---|---|
| Reprojection | Bilinear for spectral bands; nearest neighbour for SCL |
| Reflectance | DN × 0.0001 |
| Slope | Elevation gradient after DEM resampling to 10 m |
| Labels | Accepted polygons; `all_touched=True` |
| Ignored pixels | SCL 0, 1, 3, 8, 9, 10, 11, or missing spectral values on either date |
| Provenance | Fixed scene choices; `preparation.json` records count agreement and DEM tile identities |

Upstream products or encoding can change. Use the frozen processed archives for exact raster replay; rebuilding never silently substitutes a newer scene.

To check inventory-to-mask reconstruction using existing ignored pixels:

```bash
python -m dcap data prepare --mask-only --imagery-root data --data-root generated/masks
```

This checks label rasterization only; it does not redownload imagery or independently reconstruct cloud masks.

## Attribution and redistribution

Processed data may be redistributed with the supplied notices retained. **Dataset derivatives are not MIT licensed.**

| Component | Source terms | Attribution |
|---|---|---|
| BGS inventory | [Open Government Licence](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/) | “Contains data supplied by UKRI.” |
| Western Nepal inventories | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Authors, source DOI and modifications |
| Sentinel-2 derivatives | [Copernicus Sentinel notice](https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice) | “Contains modified Copernicus Sentinel data (2017–2019).” |
| Copernicus DEM derivatives | [Full DEM license](../data/licenses/Copernicus-DEM-GLO30.pdf) | Modified-product attribution, liability statement and downstream obligations |

- Keep [`DATA_NOTICES.txt`](../data/DATA_NOTICES.txt) and the DEM license when redistributing chips, masks, inventory derivatives or the example fixture. Both are embedded in every data archive.
- [`license_sources.json`](../metadata/license_sources.json) records the checked sources and document hashes.
- Raw third-party archives and administrative boundaries are not included.
