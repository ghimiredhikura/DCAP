# DCAP for landslide segmentation

Code and selected evidence for **Detail–Context Adaptive Pruning (DCAP)**: dependency-shared channel gates, physical structured pruning and task-aware recovery. The Nepal study uses paired Sentinel-2 imagery, terrain and spatially separated fitting, validation and test locations.

[**v1.1.0 prerelease**](https://github.com/ghimiredhikura/DCAP/releases/tag/v1.1.0): code, processed data, representative checkpoints and Figures 6–8 evidence are available. RTX 4080 and Jetson Orin deployment measurements are pending.

| Three-model ensemble | MAC reduction | Test AP | Test Dice | Small-scar recall | Joint support |
|---|---:|---:|---:|---:|---:|
| Dense reference | 0.00% | 0.4333 | 0.4760 | 0.2895 | — |
| DCAP | 70.03% | 0.4268 | 0.4791 | 0.3158 | 0.9484 |

The compression level was selected on validation data. Test joint support is slightly below the specified 0.95 cutoff; these results do not establish uniform superiority over the baselines. Test locations belong to the sampled 2017–2019 years.

## Quick start

Clone the repository, then activate an environment with Python and PyTorch. For a new environment, follow the short [setup guide](docs/setup.md) (conda or venv, CPU or NVIDIA GPU). Run subsequent commands from the repository root.

```bash
git clone https://github.com/ghimiredhikura/DCAP.git
cd DCAP
python -m pip install -e .
python -m pip check
python -m dcap verify
python -m dcap plot --figures 6 7 8
OPENBLAS_NUM_THREADS=2 python -m dcap replay --grid
```

Plotting and statistical replay need no model or raster downloads. Outputs go to `generated/`.

For inference, download and install the separate release assets (about 97 MiB of checkpoints and 3.43 GiB of data). Downloads and installation verify their checksums; see [data instructions](docs/data.md) for source preparation and licensing.

```bash
python -m dcap download --repository ghimiredhikura/DCAP --tag v1.1.0 --kind models --output downloads/models
python -m dcap download --repository ghimiredhikura/DCAP --tag v1.1.0 --kind data --output downloads/data
python -m dcap models install --source downloads/models
python -m dcap data install --source downloads/data
python -m dcap evaluate --ensemble dcap70 --split test
python -m dcap benchmark --output generated/ubuntu_benchmark
```

## Contents and guides

- [`src/dcap/`](src/dcap/): shared implementation for preparation, training, inference, statistics and plots.
- [`configs/`](configs/) and [`metadata/`](metadata/): study settings, fixed splits, source geometries, thresholds and checksums.
- [`results/`](results/): the data and PDFs for manuscript Figures 6–8 and compact dense/DCAP spatial evidence.
- [Environment setup](docs/setup.md): existing environments or a separate conda/venv installation.
- [Reproduction](docs/reproduction.md): checkpoint evaluation, training and regeneration of the compression grid.
- [Data and licenses](docs/data.md): processed data, source downloads and rebuilding the chips.
- [Ubuntu benchmark](docs/benchmark.md): timing protocol and host verification.
- [Release guide](docs/release.md): versioned assets, attribution and source packaging.

Only six checkpoints are distributed: the three dense and three DCAP members. Data and model assets remain separate from Git history. This release focuses on the main ResUNet compression study; it does not bundle every auxiliary comparator or regional experiment.

Author code and the six author-trained checkpoints use the [MIT license](LICENSE). Dataset components retain their source terms in [DATA_NOTICES](data/DATA_NOTICES.txt). Please cite the accompanying manuscript, this software via [CITATION.cff](CITATION.cff), and the source datasets.
