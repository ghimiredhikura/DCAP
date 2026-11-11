# Ubuntu GPU inference benchmark

Measures the released dense/DCAP ensembles using the same loader and forward function as evaluation. Full raster downloads are unnecessary.

## 1. Prepare models

Activate the environment from [setup](setup.md), then run from the repository root:

```bash
python -m dcap download --repository ghimiredhikura/DCAP --tag v1.1.0 --kind models --output downloads/models
python -m dcap models install --source downloads/models
```

The included fitting-chip fixture supplies normalized input and CPU reference predictions for all six models, including a horizontal-flip check. It retains the dataset's source terms.

## 2. Run timing

Select an idle GPU and a fresh output directory:

```bash
CUDA_VISIBLE_DEVICES=0 python -m dcap benchmark --output generated/ubuntu_benchmark
```

For a quick functional check, add `--smoke`; its shorter timings are labelled accordingly. The command does not change clocks or power modes. Record these settings when comparing devices.

## 3. Read outputs

| Output | Contents |
|---|---|
| `summary.json` | Results for all 12 model/mode/batch combinations |
| Per-job JSON | Raw timings, block medians, throughput, peak PyTorch allocated/reserved memory and parity checks |
| Provenance | Checkpoint hashes, software versions and `nvidia-smi` snapshot |

## Measurement protocol

| Item | Setting |
|---|---|
| Jobs | Dense/DCAP × eager FP32/CUDA-graph FP32/AMP × batch 1/4; isolated workers |
| Timed work | Three sequential forwards, sigmoid and FP32 probability averaging; GPU-resident input |
| Excluded | Disk I/O, preprocessing, host transfers, graph capture and thresholding |
| Timing | CUDA-synchronized wall clock; TF32 and cuDNN benchmarking disabled; deterministic kernels requested |
| FP32 sampling | Three blocks, each after 30 warmups; at least 100 calls and two seconds per block |
| AMP sampling | 30 warmups and 100 calls |

### Numerical checks before timing

| Check | Tolerance |
|---|---|
| FP32 vs CPU reference | Mean absolute error ≤1e-5; maximum ≤0.001 |
| AMP vs CPU reference | Mean absolute error ≤0.002; maximum ≤0.02 |
| CUDA graph vs eager | `atol=1e-6`, `rtol=1e-5`; changed-input replay required |

All 12 configurations passed on an Ubuntu RTX 5090 host. These are functional checks, not the pending RTX 4080/Jetson Orin manuscript measurements. This tool reports PyTorch timings and allocator memory, not TensorRT performance or whole-process memory. The separate target-device package retains the full manuscript hardware protocol.
