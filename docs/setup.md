# Environment setup

Run commands from the repository root. Tested on Ubuntu with **Python 3.13, PyTorch 2.13.0 and CUDA 13.0**. CPU PyTorch is sufficient for plots, statistical replay and tests; training and benchmarking use an NVIDIA GPU.

## 1. Choose an environment

### Use an existing environment

Activate your conda environment or venv, then check it:

```bash
python --version
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

Requirements: Python 3.10+ and PyTorch 2.4+. If these are already installed, continue to Step 3 and retain your current dependency versions.

### Or create a separate environment

With conda:

```bash
conda create -n dcap python=3.13 -y
conda activate dcap
```

Alternatively, with Python 3.13 already installed:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

## 2. Install PyTorch if needed

Choose **one** option.

**Ubuntu NVIDIA desktop with CUDA 13.0 driver support:**

```bash
nvidia-smi
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
```

**CPU only:**

```bash
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
```

For another GPU/driver combination, use the [official PyTorch installer](https://pytorch.org/get-started/locally/).

## 3. Install DCAP

For a **new Python 3.13 environment**, install the tested non-PyTorch dependencies first:

```bash
python -m pip install -r requirements-tested.txt
```

For either environment route:

```bash
python -m pip install -e '.[dev]'
python -m pip check
python -m dcap verify
```

The editable installation keeps configurations and evidence beside the source. Keep the checkout; if it moves, repeat the editable install from its new location.

## 4. Check the installation

No checkpoints or full raster dataset are needed:

```bash
python -m dcap plot --figures 6 7 8
OPENBLAS_NUM_THREADS=2 python -m dcap replay --grid
OPENBLAS_NUM_THREADS=2 python -m pytest -q
```

Outputs go to `generated/`. Continue with the [reproduction guide](reproduction.md).

## Resources and common settings

| Item | Guidance |
|---|---|
| Data storage | About 3.5 GiB each for archives and extracted rasters; allow roughly 8 GB total |
| Training | 16 GB host RAM is a practical starting point; full-data caching uses RAM |
| Campaign storage | Reserve additional space for 63 models, optimizer states and outputs |
| GPU selection | Use `CUDA_VISIBLE_DEVICES=0` before the command |
| CPU environment | `torch.cuda.is_available()` returning `False` is expected |
| Package conflicts | If the shell injects unrelated packages, run `unset PYTHONPATH` |

Do not apply the exact Python 3.13 dependency pins to an unrelated environment unless you intend to change its packages. No administrator privileges are needed.

## Optional DeepLabV3+ cells

Install a torchvision wheel matching PyTorch. For the tested CUDA 13.0 combination:

```bash
python -m pip install torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -e '.[factorial]'
```

ResUNet/DCAP and SIRFP training/evaluation, and plotting, do not require torchvision. SegFormer requires the optional `comparators` dependencies; see [the comparator guide](comparators.md).
