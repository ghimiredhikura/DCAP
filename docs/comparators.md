# SIRFP and SegFormer-B2

Use the same environment, processed data and evaluation code as the [main study](reproduction.md). These are the manuscript's **21-channel binary segmentation adaptations**. All checkpoints are tensor-only files, with three members per ensemble.

## 1. Install and download

Activate your prepared environment. For SegFormer, install a torchvision wheel matching your PyTorch installation, as described in [setup](setup.md#optional-deeplabv3-cells), then:

```bash
python -m pip install -e '.[comparators]'
python -m pip check
python -m dcap download --repository ghimiredhikura/DCAP --kind models --group comparators --output downloads/comparators
python -m dcap models install --group comparators --source downloads/comparators
```

The nine additional checkpoints total approximately **348 MiB**. No SIRFP source download is needed to evaluate them. Asset URLs are pinned in the catalog: new comparators use `v1.2.0`; unchanged dense/DCAP and data assets remain at `v1.1.0`.

| Ensemble | Purpose | Test AP | Test Dice |
|---|---|---:|---:|
| `sirfp25` | SIRFP validation-selected operating point | 0.4345 | 0.4788 |
| `sirfp70` | Common 70% MAC-reduction target | 0.3643 | 0.4065 |
| `segformer_b2` | Pretrained transformer comparison | 0.4175 | 0.4498 |

Seeds are 42, 123 and 2026, with equal probability averaging. Exact endpoints are in [`results/comparators.json`](../results/comparators.json); frozen thresholds and checkpoint hashes are in [`metadata/`](../metadata/).

## 2. Evaluate the released ensembles

Install the processed data using [Route 2](reproduction.md#2-evaluate-released-models), then:

```bash
for model in sirfp25 sirfp70 segformer_b2; do
  for split in validation test; do
    CUDA_VISIBLE_DEVICES=0 python -m dcap evaluate --ensemble "$model" --split "$split"
  done
done
```

Inspect `generated/evaluation/{model}_{split}.json`. Validation uses the frozen spatial cross-fitted thresholds; test uses each ensemble's global validation threshold. Compare all five endpoints with `results/comparators.json`. Hardware/library changes can produce small numerical differences.

## 3. Retrain SIRFP

### Obtain the pinned scoring dependency

The upstream repository has no explicit redistribution license at the pinned commit. Its source is **not bundled** here; obtain it separately and respect the upstream authors' terms. This package loads only its hash-checked scoring method, avoiding the upstream training framework.

```bash
git clone https://github.com/dywu98/SIRFP.git downloads/SIRFP
git -C downloads/SIRFP checkout 367176e45e5001e7ae1983ac9587b873ceda1e11
```

Install the released dense models using the main guide, or supply a newly trained dense checkpoint for each seed with `--teacher`.

### Train the pruning grid

```bash
for target in 0.25 0.40 0.55 0.70 0.85; do
  for seed in 42 123 2026; do
    CUDA_VISIBLE_DEVICES=0 python -m dcap train --method sirfp --target "$target" --seed "$seed" --sirfp-source downloads/SIRFP
  done
done
```

For just the released operating points, use `0.25 0.70` in the first loop. `--plan` prints a job without training; `--resume` continues an interrupted output directory.

**Adaptation:** unpooled L1-normalized Gram statistics and spatial-edge scores are accumulated over the fixed 128 fitting chips, with decay 0.999. Encoder/decoder edge weights are 0.4/0.0. Linked channel scores are averaged, encoder/decoder thresholds remain separate, and physical pruning searches for the requested MAC reduction within 0.012. At least 2% of each layer's channels (minimum one) are retained. Recovery uses the shared full objective and checkpoint-selection procedure. The scoring variant follows the pinned upstream implementation; these commands reproduce the study adaptation, not the original SIRFP benchmark datasets.

## 4. Retrain SegFormer-B2

### Prepare ImageNet initialization

```bash
python -m dcap pretraining --output downloads/pretraining
```

This downloads the pinned NVIDIA MiT-B2 checkpoint, verifies its SHA-256, converts its key layout and verifies all 332 backbone tensors. For an existing download, add `--source /path/to/pytorch_model.bin`. No pretrained download occurs when loading a released study checkpoint.

### Train the three models

```bash
for seed in 42 123 2026; do
  CUDA_VISIBLE_DEVICES=0 python -m dcap train --method dense --architecture segformer_b2 --seed "$seed" --pretraining downloads/pretraining/mit_b2_original_backbone.pt
done
```

**Configuration:** original B2 backbone, 768-wide decoder, one output channel; RGB stem kernels initialize post-event channels 8/7/6 and the remaining kernels start at zero. AdamW uses encoder/decoder learning rates 0.00006/0.0006, weight decay 0.01 and zero decay for normalization parameters. The shared study loss, augmentation, EMA, stopping and checkpoint-selection rules apply. MAC counts include both attention matrix multiplications. Pins and settings are in [`configs/comparators.json`](../configs/comparators.json).

## 5. Evaluate newly trained checkpoints

Re-estimate thresholds from the new validation predictions; do not reuse thresholds from the released weights. Example for SIRFP25:

```bash
python -m dcap evaluate --ensemble sirfp25_retrained --split validation --checkpoints generated/training/sirfp_mac25_seed42/checkpoint.pt generated/training/sirfp_mac25_seed123/checkpoint.pt generated/training/sirfp_mac25_seed2026/checkpoint.pt
python -m dcap evaluate --ensemble sirfp25_retrained --split test --checkpoints generated/training/sirfp_mac25_seed42/checkpoint.pt generated/training/sirfp_mac25_seed123/checkpoint.pt generated/training/sirfp_mac25_seed2026/checkpoint.pt --validation-result generated/evaluation/sirfp25_retrained_validation.json
```

Use the corresponding `segformer_b2_seed{seed}/checkpoint.pt` paths for SegFormer. Each evaluation also saves per-chip sufficient statistics in an `.npz` file. Geographic fold retraining remains outside this compact package.

## Licenses and references

- **SIRFP:** [paper](https://arxiv.org/abs/2412.12672), [upstream source](https://github.com/dywu98/SIRFP). Our integration and trained ResUNet weights are covered by the repository's MIT license; upstream source is separately obtained and not relicensed.
- **SegFormer:** [paper](https://arxiv.org/abs/2105.15203), [upstream source](https://github.com/NVlabs/SegFormer). Derived code and weights retain the [NVIDIA Source Code License](../licenses/SegFormer-LICENSE.txt), including its noncommercial research/evaluation restriction.

Retain [the model notices](../licenses/COMPARATOR_MODEL_NOTICES.txt) with redistributed checkpoints. Dataset terms remain separate.
