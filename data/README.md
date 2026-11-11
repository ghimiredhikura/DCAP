# Data

## Included in this folder

| File | Purpose |
|---|---|
| [`example.npz`](example.npz) | Normalized fitting-chip input and CPU reference predictions for inference checks; no full dataset needed |
| [`DATA_NOTICES.txt`](DATA_NOTICES.txt) | Source attribution, modifications and redistribution terms |
| [`licenses/Copernicus-DEM-GLO30.pdf`](licenses/Copernicus-DEM-GLO30.pdf) | Full DEM license |

## Full processed dataset

Download the three year-specific archives from the [v1.1.0 release](https://github.com/ghimiredhikura/DCAP/releases/tag/v1.1.0): `chips_2017.zip`, `chips_2018.zip`, `chips_2019.zip` (**3.43 GiB** total).

- **1,320 image/mask pairs:** 990 fitting, 132 validation, 198 test.
- **Images:** georeferenced, losslessly compressed 14-band float32 GeoTIFFs—12 pre/post Sentinel-2 reflectance bands, elevation and slope.
- **Masks:** 0 = background, 1 = landslide, 255 = ignored.
- The loader constructs 21 input channels by adding six temporal differences and a validity indicator.

```bash
python -m dcap download --repository ghimiredhikura/DCAP --tag v1.1.0 --kind data --output downloads/data
python -m dcap data install --source downloads/data
python -m dcap data verify
```

Installation creates `images/` and `masks/` here; these rasters are excluded from Git. Allow roughly 8 GB for archives and extracted data. Splits and checksums are in [`metadata/`](../metadata/).

Dataset derivatives retain their source licenses, **not MIT**. Keep the supplied notices. See [data preparation](../docs/data.md) for source downloads, rebuilding and attribution details.
