# Release layout and publication

The source is hosted at [ghimiredhikura/DCAP](https://github.com/ghimiredhikura/DCAP). The [v1.1.0 prerelease](https://github.com/ghimiredhikura/DCAP/releases/tag/v1.1.0) contains the processed data and representative model ensembles. RTX 4080/Jetson Orin deployment measurements remain pending and are not included in the reported results. No software DOI is assigned.

## Asset layout

Git history contains `src/`, `configs/`, `metadata/`, `results/`, the `data/` notices/example, `docs/`, `tests/` and `tools/`. Generated outputs, installed rasters, model weights, virtual environments and local caches are excluded.

Release assets contain six `.pt` files and `MODEL_NOTICES.txt`, three year-specific data ZIPs and `DATA_NOTICES.txt`, the data catalog and DEM license, a portable source ZIP, and checksums. Each data ZIP includes its attribution and DEM license. Catalogs in `metadata/` record asset filenames, sizes and SHA-256 hashes. The release's `SHA256SUMS` uses the flat asset filenames; after manually downloading the assets to one folder, run `sha256sum -c SHA256SUMS` there.

The download helper verifies the required assets against the catalogs:

```bash
python -m dcap download --repository ghimiredhikura/DCAP --tag v1.1.0 --kind models --output downloads/models
python -m dcap download --repository ghimiredhikura/DCAP --tag v1.1.0 --kind data --output downloads/data
python -m dcap models install --source downloads/models
python -m dcap data install --source downloads/data
```

Data archive installation also extracts the full DEM license. The portable source ZIP extracts to `dcap/`; it contains the same files as the tagged source checkout, excluding Git metadata.

## Preparing later versions

1. Update the version and date in `CITATION.cff`, `pyproject.toml` and `metadata/provenance.json`, and the documented asset tag.
2. Run `python tools/package_release.py --output ../dcap-source.zip`, `python -m dcap verify` and `pytest -q` from the repository root. The packaging tool rebuilds the source manifest and excludes generated results and installed assets.
3. Inspect the resulting source archive and attach the separately catalogued assets. Preserve `LICENSE`, `DATA_NOTICES.txt`, the DEM license and `MODEL_NOTICES.txt`.
4. Check the public links from a fresh download, then place the repository/version link in the manuscript Data Availability statement. A DOI may be added after archival deposition; none is claimed here.

Author code and selected author-trained ResUNet weights are MIT licensed. Dependencies are installed from their upstream distributions and retain their licenses. No ImageNet pretrained parameters, SegFormer weights, vendor source, administrative boundaries or unrelated results are bundled. The main pruning baselines use the installed Torch-Pruning package. The data notices, rather than the software MIT license, govern all included dataset derivatives.

## File map

| Location | Purpose |
|---|---|
| `src/dcap/` | Reusable model, loss, pruning, data, inference and analysis implementation |
| `configs/` | Main protocol and factorial cell definitions |
| `metadata/` | Splits, provenance, source geometries/scenes, asset checksums |
| `results/` | Only the evidence required for Figures 6–8 and the dense/DCAP example |
| `data/` | Small inference fixture and source notices; installed rasters are ignored |
| `docs/` | Reproduction, data, benchmark and validation instructions |
| `tools/package_release.py` | Source manifest and portable ZIP creation |
| `tests/` | Frozen-evidence, threshold-provenance and structural-pruning regression checks |
