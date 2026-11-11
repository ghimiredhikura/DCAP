# Reproducing the main results

Complete [environment setup](setup.md), then run commands from the repository root. Choose the route that matches what you want to verify.

| Route | What it checks | Requirements |
|---|---|---|
| [1. Replay supplied evidence](#1-replay-supplied-evidence) | Reported statistics and Figures 6–8 | CPU; no asset downloads |
| [2. Evaluate released models](#2-evaluate-released-models) | Predictions from the dense/DCAP checkpoints | Six checkpoints, processed data; NVIDIA GPU recommended |
| [3. Retrain the main comparison](#3-retrain-the-main-comparison) | Training, pruning, recovery and model selection | Processed data, NVIDIA GPU, substantial runtime and output storage |
| [4. SIRFP and SegFormer-B2](comparators.md) | Evaluate or retrain the comparison methods | Optional dependencies and checkpoint download |

**Scope:** the main ResUNet compression study, with optional [SIRFP and SegFormer-B2 reproduction](comparators.md). Dense, DCAP70, SIRFP25, SIRFP70 and SegFormer-B2 checkpoints are available. Other main grid methods can be retrained; geographic outer-fold retraining is outside this compact package.

## 1. Replay supplied evidence

### Run

```bash
python -m dcap verify
OPENBLAS_NUM_THREADS=2 python -m dcap replay --grid
python -m dcap plot --figures 6 7 8 10
```

### Inspect

| Output | Expected content |
|---|---|
| `verify` output | `"passed": true`; fitting/validation/test counts 990/132/198 |
| `generated/replay/representative_replay.json` | Dense/DCAP endpoints and paired spatial intervals; DCAP joint support 1.0000 on validation and 0.9484 on test |
| `generated/replay/figure8.csv` | All 72 grid/margin sensitivity rows |
| `generated/figures/` | Figures 6–8 as PDF and PNG |

`replay` recomputes statistics from `results/spatial_evidence.npz`; it does not run model inference. It covers the dense/DCAP metrics, their Figure 7 intervals and all Figure 8 values. The other Figure 6/7 curves use supplied CSV values; regenerating those from predictions requires Route 3.

To check agreement with the frozen evidence and the package's integrity rules:

```bash
python -m pip install -e '.[dev]'
OPENBLAS_NUM_THREADS=2 python -m pytest -q
```

See [validation](validation.md) for the checks performed and their limits.

For the verified target-device comparison, see [deployment results](benchmark.md). `plot --figures 10` also rebuilds `table13.csv` and `figure10.csv` from recorded timing summaries.

## 2. Evaluate released models

### Download and install

About 97 MiB of weights and 3.43 GiB of archives; allow roughly 8 GB for downloads and extracted data. See [data preparation](data.md) for offline installation or rebuilding from source.

```bash
python -m dcap download --repository ghimiredhikura/DCAP --tag v1.1.0 --kind models --output downloads/models
python -m dcap download --repository ghimiredhikura/DCAP --tag v1.1.0 --kind data --output downloads/data
python -m dcap models install --source downloads/models
python -m dcap data install --source downloads/data
```

### Evaluate both ensembles

```bash
for model in dense dcap70; do
  for split in validation test; do
    CUDA_VISIBLE_DEVICES=0 python -m dcap evaluate --ensemble "$model" --split "$split"
  done
done
```

### Compare outputs

- `generated/evaluation/{model}_{split}.json`: endpoints, checkpoint hashes and threshold provenance.
- Matching `.npz`: per-chip statistics, with its checksum recorded in the JSON.
- Reference values: [`results/representative.json`](../results/representative.json) and the [README table](../README.md#main-results).

The evaluator uses the stored validation thresholds for these exact checkpoints. For a CPU functional check, add `--device cpu --precision fp32`. CPU/FP32 and different GPU/library combinations can differ slightly from the AMP study results.

## 3. Retrain the main comparison

### Inspect the plan, then run each phase

The campaign trains **63 models**: three dense references plus three members at five compression targets for DCAP, Group Taylor, Group Magnitude and direct compact models. Jobs run sequentially on one visible GPU.

```bash
python -m dcap campaign plan --output generated/campaign
CUDA_VISIBLE_DEVICES=0 python -m dcap campaign train --data-root data --output generated/campaign
CUDA_VISIBLE_DEVICES=0 python -m dcap campaign evaluate --data-root data --output generated/campaign
OPENBLAS_NUM_THREADS=2 python -m dcap campaign collect --output generated/campaign
python -m dcap plot --source generated/campaign/results --output generated/campaign/figures
```

### Outputs and continuation

- Training folders retain settings, measured structures, selected weights, history and optimizer/RNG resume state.
- Collection writes result tables, paired comparisons and `selections.json`; figures use these newly generated estimates.
- Repeat a campaign phase to verify/reuse completed outputs and resume unfinished training. Interrupted pruning preparation reuses a complete initial checkpoint or repeats preparation.
- Evaluation requires all planned fits. Keep `complete.json`, `configuration.json`, `structure.json` and selected checkpoints until collection finishes.
- Reserve more storage than inference requires: the campaign retains optimizer states for many fits.

Selection uses validation AP and joint support only. Direct compact models remain controls; Figures 7–8 use the matched 70% target. Across-method selection covers the three included pruning methods. The full 63-model campaign was not repeated during package validation.

## Optional: individual runs

### Train a dense parent and a DCAP model

```bash
python -m dcap train --method dense --seed 42 --output generated/dense42
python -m dcap train --method dcap --seed 42 --target 0.70 \
  --teacher generated/dense42/checkpoint.pt --output generated/dcap42
```

- Other methods: `taylor`, `magnitude`, `compact`, `dense_recovery`.
- Recovery controls: `--recovery supervised` or a single `--omit-loss`.
- Resume one run: repeat its original command with `--resume`, preserving options and output path.
- Functional smoke check: `train --method dense --smoke --output generated/smoke` (eight fitting chips, two epochs). Fresh DCAP calibration still requires the full fitting set.
- The 16 cells in [`configs/factorial.json`](../configs/factorial.json) reuse the same input/loss code. DeepLabV3+ cells need the [optional torchvision setup](setup.md#optional-deeplabv3-cells) and download official ImageNet encoder weights at training time.

### Evaluate newly trained checkpoints

After training all three members, fit validation thresholds **before** evaluating test. Substitute your actual checkpoint paths:

```bash
python -m dcap evaluate --ensemble custom_dcap --split validation \
  --checkpoints generated/dcap42/checkpoint.pt generated/dcap123/checkpoint.pt generated/dcap2026/checkpoint.pt \
  --output generated/custom_dcap_validation.json
python -m dcap evaluate --ensemble custom_dcap --split test \
  --checkpoints generated/dcap42/checkpoint.pt generated/dcap123/checkpoint.pt generated/dcap2026/checkpoint.pt \
  --validation-result generated/custom_dcap_validation.json \
  --output generated/custom_dcap_test.json
```

Do not reuse released thresholds for new weights. The evaluator rejects validation records from different checkpoint hashes or precision; collection rejects missing/changed statistics.

## Protocol reference

| Item | Fixed setting |
|---|---|
| Configuration | [`study.json`](../configs/study.json): optimizer, recovery loss, margins, stopping rules and budgets; [`design.json`](../configs/design.json): subsets and fitting-only normalization |
| Partitions | 990 fitting / 132 validation / 198 test chips in [`split.csv`](../metadata/split.csv) |
| Replication | Seeds 42, 123, 2026; equally weighted three-model ensembles |
| Calibration | 128 fitting chips for gates; 32 for Group Taylor importance |
| Model fitting | Dense/direct compact from scratch; pruned models inherit the corresponding dense parent and use the same full recovery loss |
| Compression | Measured MAC reduction, rather than parameter-count reduction |
| Predictions | AMP forwards; sigmoid probabilities averaged in FP32 |
| Thresholds | Leave-one-spatial-block-out fits on validation; global validation threshold on test |
| Metrics | AP: 1,001 score bins; small-component recall: fitting-derived 41-pixel area cutoff and 50% coverage; boundary F1: two-pixel tolerance |
| Spatial inference | Paired 20 km blocks, 10,000 shared bootstrap draws; undefined constraint draws fail joint support |
| Selection | Validation AP ≥ dense AP − 0.01 and joint support ≥ 0.95; greatest passing MAC reduction, otherwise dense fallback |
| Figure 8 | 10/20/30 km blocks; 0/E/N/EN denote unshifted/half-block east/north/both shifts; multipliers scale preservation tolerances |

Training trajectories need not be bitwise identical across machines. The frozen checkpoints, processed rasters and compact statistics provide the direct route to replaying the reported values.

## Code map

Start with [`cli.py`](../src/dcap/cli.py), then [`experiment.py`](../src/dcap/experiment.py) for campaign orchestration.

| Files | Role |
|---|---|
| `data.py`, `preprocess.py` | Inputs and source-data preparation |
| `training.py`, `training_helpers.py`, `losses.py` | Fitting, recovery and shared objectives |
| `algorithm.py`, `models.py` | Shared gates, physical pruning, inheritance and model loading |
| `evaluation.py`, `analysis.py`, `figures.py` | Predictions, spatial inference and plots |
| `benchmark.py` | Timing with the same model loader and ensemble forward function |

Keep the editable checkout. Use `DCAP_ROOT` to identify it when invoking the package elsewhere.


## Published result tables

- [Complete pruning operating points](../results/operating_points.csv): all five targets for DCAP, Group Taylor and Group Magnitude under both recovery formulations, plus the SIRFP adaptation under task-aware recovery; validation/test endpoints, computation and joint support (70 rows). `mac_reduction` is a fraction; `target_percent` is a percentage. These are ensemble results, not individual model scores.
- [Deployment comparison](benchmark.md): Table 13 and Figure 10 from the verified RTX 4080/Orin timing summaries. Run `python -m dcap plot --figures 10` to reproduce the tables and plot without a GPU.

Only dense/DCAP checkpoints are distributed. The shared campaign regenerates the main ResUNet comparison; SIRFP/SegFormer implementations and regional retraining are outside this compact package.
