# Reproducing the main results

## Setup and scope

Complete [environment setup](setup.md) first and run commands from the checkout root. Keep this source checkout because configurations and evidence live beside `src/`. `DCAP_ROOT` can explicitly identify it when invoking the package elsewhere. The recorded environment is in [tested_environment.json](tested_environment.json).

The release contains the main dense and DCAP ensembles, 20 compression operating points for Figure 6, matched 70%-target paired intervals for Figure 7, and 72 sensitivity rows for Figure 8. It includes no optimizer histories or full prediction arrays. SIRFP, SegFormer, spatial outer-fold studies and other auxiliary experiments are outside this compact release's execution matrix. Their checkpoint archives are not required for Figures 6–8. Third-party comparator source is not redistributed here.

## Fixed protocol

`configs/study.json` gives the optimizer, recovery weights, margins, stopping rules and compression targets. `configs/design.json` fixes fitting, validation, calibration subsets and fitting-only normalization; `metadata/split.csv` records the 990/132/198 partitions. Seeds 42, 123 and 2026 define equally weighted three-model ensembles. Gate training uses 128 fitting chips; Taylor importance uses 32. Channel removal targets measured MACs rather than parameter counts. Dense and compact models train from scratch; pruned models recover from the corresponding dense seed. The main Group Taylor and Group Magnitude comparisons use the same full recovery loss as DCAP.

Evaluation averages sigmoid probabilities in FP32, using AMP forwards for manuscript replay. Validation endpoints use leave-one-spatial-block-out threshold fits; test uses the global validation threshold. AP uses 1,001 score bins. Small-component recall uses the fitting-derived 41-pixel area threshold, 50% predicted coverage and connected components; boundary F1 uses a two-pixel tolerance. Paired bootstrap inference uses the same draws for both models, 10,000 draws and a 20 km primary grid. Undefined constraint draws fail the joint decision. A pruning candidate qualifies when validation AP is at least dense AP minus 0.01 and validation joint support is ≥0.95; the greatest measured MAC reduction among qualifying candidates is selected, with the dense reference as fallback. Test results never select the budget.

## Replay supplied evidence

```bash
python -m dcap verify
OPENBLAS_NUM_THREADS=2 python -m dcap replay --grid
python -m dcap plot --figures 6 7 8
```

`replay` recomputes endpoints, intervals and support from per-chip additive statistics in `results/spatial_evidence.npz`. It reproduces the dense/DCAP metrics, their Figure 7 intervals and all Figure 8 values without inference. Figure 6 and the remaining Figure 7 methods plot the supplied small CSV tables; obtaining their statistics anew requires training the corresponding models. In Figure 8, 0 is the unshifted grid and E, N and EN shift its origin by half a block east, north or both. Margin multipliers scale the four preservation tolerances, not the probabilities.

## Evaluate the released checkpoints

Download the model and data assets once, then install and evaluate them:

```bash
python -m dcap download --repository ghimiredhikura/DCAP --tag v1.1.0 --kind models --output downloads/models
python -m dcap download --repository ghimiredhikura/DCAP --tag v1.1.0 --kind data --output downloads/data
python -m dcap models install --source downloads/models
python -m dcap data install --source downloads/data
for model in dense dcap70; do
  for split in validation test; do
    CUDA_VISIBLE_DEVICES=0 python -m dcap evaluate --ensemble "$model" --split "$split"
  done
done
```

The evaluator checks weight hashes and uses the stored validation thresholds for those exact released ensembles. Each result comprises JSON endpoints/provenance and an NPZ of per-chip additive statistics. The JSON records the NPZ checksum; collection rejects a missing or changed statistics file. For a CPU functional check use `--device cpu --precision fp32`; CPU/FP32 endpoints are not expected to be bitwise identical to the AMP study results. Different GPU/library combinations can produce small numerical differences.

## Train and regenerate the main grid

Inspect the plan before starting a substantial run:

```bash
python -m dcap campaign plan --output generated/campaign
CUDA_VISIBLE_DEVICES=0 python -m dcap campaign train --data-root data --output generated/campaign
CUDA_VISIBLE_DEVICES=0 python -m dcap campaign evaluate --data-root data --output generated/campaign
OPENBLAS_NUM_THREADS=2 python -m dcap campaign collect --output generated/campaign
python -m dcap plot --source generated/campaign/results --output generated/campaign/figures
```

The campaign trains 63 models: three dense references and three members at five targets for DCAP, Group Taylor, Group Magnitude and direct compact models. It runs sequentially on one visible GPU. Repeating a completed phase verifies and reuses matching outputs. Training continues from local optimizer/RNG state after interruption; if interrupted during pruning preparation, it reuses a complete initial checkpoint or repeats preparation. Evaluation requires all planned fits to have completed. Do not delete `complete.json`, `configuration.json`, `structure.json` or the selected checkpoints before collection. Reserve substantially more storage and time than needed for the compact release. Each training directory contains settings, structure measurements, history, selected tensor-only weights and local resume state. Collection applies the AP guard and joint rule using validation data, then writes plot-ready tables, `selections.json` and paired comparisons. Direct compact models remain controls. Figures 7 and 8 retain the matched 70% target; budget-selection results are recorded separately. The across-method decision in this compact campaign covers the three included pruning methods, not the additional SIRFP comparison. The supplied published values remain unchanged; a new training campaign produces its own estimates.

For an individual run:

```bash
python -m dcap train --method dense --seed 42 --output generated/dense42
python -m dcap train --method dcap --seed 42 --target 0.70 \
  --teacher generated/dense42/checkpoint.pt --output generated/dcap42
```

The same command supports `taylor`, `magnitude`, `compact`, `dense_recovery`, `--recovery supervised`, a single `--omit-loss`, and the 16 design cells in `configs/factorial.json`. These reuse the same input and loss functions. For factorial cells using DeepLabV3+ with a ResNet-34 encoder, install the optional dependency with `python -m pip install -e '.[factorial]'` (a torchvision wheel matching PyTorch). Those cells download the official torchvision ImageNet encoder weights at training time; the weights are not bundled. A short training check is `train --method dense --smoke --output generated/smoke`; smoke mode trains on eight chips for two epochs and is not scientific evidence. Fresh DCAP calibration requires the complete fitting set, so it is not silently shortened by smoke mode.

For three newly trained DCAP checkpoints, evaluate validation before test (substitute your actual run paths):

```bash
python -m dcap evaluate --ensemble custom_dcap --split validation \
  --checkpoints generated/dcap42/checkpoint.pt generated/dcap123/checkpoint.pt generated/dcap2026/checkpoint.pt \
  --output generated/custom_dcap_validation.json
python -m dcap evaluate --ensemble custom_dcap --split test \
  --checkpoints generated/dcap42/checkpoint.pt generated/dcap123/checkpoint.pt generated/dcap2026/checkpoint.pt \
  --validation-result generated/custom_dcap_validation.json \
  --output generated/custom_dcap_test.json
```

The evaluator rejects a validation result from different checkpoint hashes or precision. Do not use released thresholds for newly trained models. For an individual interrupted run, repeat its original `train` command with `--resume`; retain the same options and output directory.

## Verification

```bash
python -m pip install -e '.[dev]'
OPENBLAS_NUM_THREADS=2 pytest -q
```

Tests check the frozen scientific evidence and sensitivity values, threshold provenance and source-mask lineage. Host execution details and limits are recorded in `validation.md`. GPU training trajectories are not promised to be bitwise identical across machines.

## Reading the code

Start at `cli.py` for the commands, then `experiment.py` for the four campaign phases. The main path is:

`data.py` → `training.py` → `algorithm.py` → `models.py` → `evaluation.py` → `analysis.py` → `figures.py`.

`algorithm.py` contains shared-gate learning, physical channel removal and inheritance selection. `losses.py` and `training_helpers.py` provide the common losses, moving averages and validation metrics. `preprocess.py` handles source-data preparation; `benchmark.py` uses the same model loader and forward function as evaluation. Functions use direct loops and ordinary PyTorch/NumPy operations; no separate implementation is maintained for each experiment.
