# Data preparation and source terms

## Processed assets

The three separate `chips_2017.zip`, `chips_2018.zip` and `chips_2019.zip` archives contain 1,320 image/mask pairs. Each image is a losslessly compressed, georeferenced 14-band float32 GeoTIFF: six pre-monsoon and six post-monsoon Sentinel-2 bands (blue, green, red, NIR, SWIR1, SWIR2), elevation and slope. Masks use 0 background, 1 landslide and 255 ignored. `metadata/chips.csv` identifies scenes, polygons, grid bounds and positive counts. No lossy quantization was introduced for distribution.

```bash
python -m dcap download --repository ghimiredhikura/DCAP --tag v1.1.0 --kind data --output downloads/data
python -m dcap data install --source downloads/data
python -m dcap data verify
```

The installer verifies archive SHA-256 values, rejects unsafe paths, preserves different existing files, extracts the notices and verifies all 2,640 raster hashes. No additional data conversion is required. Allow roughly 8 GB when retaining both archives and extracted data. During model training, data caching also needs host RAM.

Assets are attached to the [v1.1.0 prerelease](https://github.com/ghimiredhikura/DCAP/releases/tag/v1.1.0). For models, use `--kind models --output downloads/models`, followed by `models install --source downloads/models`. If the archives were transferred offline, pass their folder to `data install --source` instead; no download is required.

At runtime, the package constructs the 21-channel full input from the 12 reflectances, six post-minus-pre changes, two terrain channels and one validity indicator. Normalization comes only from fitting chips. All invalid feature pixels are zeroed before adding the validity channel. The dataset class also provides the other channel combinations in the factorial design.

## Source access and rebuilding

The source inventory records and selected scene URLs are included as small gzip-compressed metadata. They preserve polygon identifiers and geometry; local filesystem paths and temporary signed URLs are excluded. `source_quality.json.gz` records the two-date polygon/context quality checks. The fixed accepted set has 2,063 inventory polygons. All four bad-pixel fractions must be ≤0.20. Only these accepted geometries contribute to labels; source datasets contain additional polygons.

1. Obtain the monsoon landslide inventory from [BGS accession 166966](https://webapps.bgs.ac.uk/services/ngdc/accessions/index.html#item166966). The [BGS metadata API](https://webservices.bgs.ac.uk/accessions/item/166966) records the Open Government Licence and the required UKRI attribution.
2. Obtain the western Nepal source inventories from [Zenodo record 7970874](https://doi.org/10.5281/zenodo.7970874), credited to Katy Burrows, Odin Marc and Christoff Andermann. The record is licensed CC BY 4.0. Polygon identifiers in the included GeoJSON connect these sources to the chip manifest.
3. The frozen Sentinel-2 L2A scenes are listed in `source_scenes.json.gz` and `chips.csv`. They are read from [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a) with a fresh public asset signature at read time. No private account token is packaged.
4. Copernicus DEM GLO-30 tiles are obtained through the public [Earth Search STAC endpoint](https://earth-search.aws.element84.com/v1/collections/cop-dem-glo-30). GLO-30 is resampled onto the 10 m chip grids; this does not create native 10 m elevation detail.

Rebuild a sample, then the full fixed chip set in a fresh destination:

```bash
python -m dcap data prepare --data-root generated/source_sample --limit 1
python -m dcap data prepare --data-root generated/source_data
```

This reads the attributed inventory geometries and selected source scenes, reprojects spectral bands with bilinear resampling and SCL with nearest-neighbour resampling, applies the study reflectance conversion (DN ×0.0001), and forms DEM slope from the 10 m resampled elevation gradient. It rasterizes the quality-approved polygons with `all_touched=True`. SCL classes 0, 1, 3, 8, 9, 10 and 11, or missing spectral pixels on either date, become ignored. The exact per-chip scene choice is retained, rather than rerunning cloud-based scene selection. `preparation.json` records count agreement and DEM tile identities.

Remote access requires internet and available upstream assets. A provider may change a product or its encoding; use the frozen processed archives for exact raster replay. Source rebuilding does not silently substitute a newer scene. To check mask reconstruction separately using existing ignored pixels:

```bash
python -m dcap data prepare --mask-only --imagery-root data --data-root generated/masks
```

That command verifies the inventory-to-raster step; it does not redownload imagery or independently reconstruct cloud masks.

## Redistribution and attribution

The inspected source terms permit this processed-data distribution with the notices retained. **The dataset is not relicensed under MIT.** The software license covers author code and designated author-trained weights only.

| Component | Source terms | Required attribution |
|---|---|---|
| BGS inventory | [Open Government Licence](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/) | “Contains data supplied by UKRI.” |
| Western Nepal inventories | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Authors, source DOI and indication of modifications |
| Sentinel-2 derivatives | [Copernicus Sentinel legal notice](https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice) | “Contains modified Copernicus Sentinel data (2017–2019).” |
| Copernicus DEM GLO-30 derivatives | [Copernicus DEM license](../data/licenses/Copernicus-DEM-GLO30.pdf) | Exact modified-product attribution, liability statement and downstream obligations in the supplied notice |

`data/DATA_NOTICES.txt` carries the complete attributions and descriptions of modifications; it and the DEM license are also embedded in every archive. Preserve them when redistributing chips, masks, derived inventory geometries or the example fixture. `metadata/license_sources.json` records the checked authoritative sources and downloaded document hashes. Raw third-party archives and administrative boundary layers are not included.
