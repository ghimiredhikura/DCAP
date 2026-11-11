# Release assets and packaging

[Repository](https://github.com/ghimiredhikura/DCAP) · [v1.1.0 prerelease](https://github.com/ghimiredhikura/DCAP/releases/tag/v1.1.0)

RTX 4080/Jetson Orin measurements are pending. No software DOI is assigned.

## Where files live

| Location | Contents |
|---|---|
| Git repository | Code, configuration, metadata, selected results, example fixture, notices and guides |
| Model assets | Six `.pt` checkpoints and `MODEL_NOTICES.txt` |
| Data assets | Three year-specific ZIPs, `DATA_CATALOG.json`, `DATA_NOTICES.txt` and DEM license |
| Source asset | Portable ZIP extracting to `dcap/`; matches the tagged checkout without Git metadata |
| Checksums | `SHA256SUMS` for all attached files; separate source-ZIP checksum |

Installed rasters, weights, generated outputs, environments and caches stay outside Git history. Each data ZIP contains its notices and full DEM license.

## Download and verify

Follow [model evaluation](reproduction.md#2-evaluate-released-models) or [data installation](data.md#1-install-the-processed-dataset). The download helper checks filenames and SHA-256 values against the catalogs in `metadata/`.

For manual downloads, place all attached assets in one folder and run:

```bash
sha256sum -c SHA256SUMS
```

This manifest uses flat asset filenames.

## Prepare a later version

1. Update the version/date in `CITATION.cff`, `pyproject.toml` and `metadata/provenance.json`, plus the documented asset tag.
2. Rebuild and check the source package from the repository root:

   ```bash
   python tools/package_release.py --output ../dcap-source.zip
   python -m dcap verify
   python -m pytest -q
   ```

3. Inspect the source ZIP and separately catalogued assets. Keep all notices and licenses.
4. Publish under a new tag, then verify anonymous downloads and checksums before citing the version.

Documentation updates on `main` may follow a release; an existing tag and its archive retain their published snapshot.

## License boundaries

- **MIT:** author code and the selected author-trained ResUNet checkpoints.
- **Source terms:** dataset derivatives, including the example fixture; see [data notices](../data/DATA_NOTICES.txt).
- **Dependencies:** installed from upstream distributions under their own licenses; main pruning baselines use Torch-Pruning.
- **Not bundled:** ImageNet pretrained parameters, SegFormer weights, vendor code, administrative boundaries or unrelated results.

See the [repository overview](../README.md#repository-guide) for the source-folder map.
