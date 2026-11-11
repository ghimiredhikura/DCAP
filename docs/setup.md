# Environment setup

Run the commands from the repository root, where `pyproject.toml` is located. The tested combination is **Python 3.13, PyTorch 2.13.0 and CUDA 13.0** on Ubuntu. Training and benchmarking need an NVIDIA GPU; plotting, statistical replay and the regression tests also run with CPU PyTorch.

## If you already have an environment

Activate your existing conda environment or virtual environment. Check its Python and PyTorch installation, then install this checkout:

```bash
python --version
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
python -m pip install -e '.[dev]'
python -m pip check
python -m dcap --help
```

This uses the existing PyTorch installation when it satisfies the package requirements. The package supports Python 3.10+ and PyTorch 2.4+, but the exact dependency pins below were tested with Python 3.13. Do not apply those pins to an unrelated environment unless you intend to change its package versions.

## Create a separate environment

With conda installed:

```bash
conda create -n dcap python=3.13 -y
conda activate dcap
```

Alternatively, if Python 3.13 is already installed:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

Choose **one** PyTorch installation. For an Ubuntu NVIDIA desktop with a driver supporting CUDA 13.0:

```bash
nvidia-smi
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
```

For CPU-only plotting, evidence replay and tests:

```bash
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
```

For another GPU/driver combination, use the matching command from the [official PyTorch installer](https://pytorch.org/get-started/locally/). Then install the tested non-PyTorch dependencies and the local package:

```bash
python -m pip install -r requirements-tested.txt
python -m pip install -e '.[dev]'
python -m pip check
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
python -m dcap verify
```

`False` for CUDA is expected in the CPU environment. Use `CUDA_VISIBLE_DEVICES=0` to select a GPU for training or timing. If your shell injects other Python packages through `PYTHONPATH`, clear it before setup (`unset PYTHONPATH`).

## First checks

```bash
python -m dcap plot --figures 6 7 8
OPENBLAS_NUM_THREADS=2 python -m dcap replay --grid
OPENBLAS_NUM_THREADS=2 python -m pytest -q
```

These commands need no checkpoint or full-dataset download. Outputs are placed in `generated/`; installation requires no administrator privileges. Keep the source checkout: settings and evidence live beside `src/`, so use the editable installation shown above. If the checkout moves, repeat `pip install -e '.[dev]'` from its new location.

The full dataset uses about 3.5 GiB for the archives and another 3.5 GiB when extracted. Full training caches fitting/validation rasters in RAM; 16 GB host RAM is a practical starting point. Keep at least 8 GiB disk space free during training, in addition to the space needed for outputs. The 63-model campaign writes optimizer states and requires substantially more storage than inference.

The optional DeepLabV3+ factorial cells also require a torchvision wheel matching PyTorch. For the tested CUDA 13.0 combination, install `torchvision==0.28.0` from the same PyTorch index, then run `python -m pip install -e '.[factorial]'`. ResUNet/DCAP training, plotting and the released ensembles do not require torchvision.
