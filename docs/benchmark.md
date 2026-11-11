# Ubuntu GPU inference benchmark

Activate the environment described in [setup.md](setup.md) before running these commands. The same `dcap.models` loader and probability-averaging function serve evaluation and benchmarking. No second inference implementation is included.

```bash
python -m pip install -e .
python -m dcap download --repository ghimiredhikura/DCAP --tag v1.1.0 --kind models --output downloads/models
python -m dcap models install --source downloads/models
CUDA_VISIBLE_DEVICES=0 python -m dcap benchmark --output generated/ubuntu_benchmark
```

No full raster download is required. The included fitting-chip fixture contains the normalized input and CPU reference probabilities for the six released models on the original and horizontally flipped inputs. The fixture remains subject to the data notices.

The command runs 12 isolated jobs: dense/DCAP ensembles × eager FP32/CUDA-graph FP32/AMP × batch 1/4. Each ensemble executes three sequential forwards, sigmoid and FP32 probability averaging. Input is already resident on the GPU. Timing excludes disk access, preprocessing, host transfers, graph capture and downstream thresholding. CUDA synchronization brackets wall-clock timing. TF32 is disabled, cuDNN benchmarking is disabled and deterministic kernels are requested. CUDA graphs are checked with changed input so a stale captured output cannot pass.

Each member must pass CPU-reference parity before timing. FP32 tolerances are mean absolute error ≤1e-5 and maximum ≤0.001; AMP tolerances are ≤0.002 and ≤0.02. Graph output must match eager output to atol 1e-6/rtol 1e-5. FP32 timing uses three blocks, each after 30 warmups with at least 100 calls and two seconds. AMP uses 30 warmups and 100 calls. `--smoke` reduces timing samples for a functional check and labels the output accordingly.

Per-job JSON retains raw times, block medians, throughput, peak PyTorch allocated/reserved memory, numerical checks, checkpoint hashes, versions and an `nvidia-smi` snapshot. Summary results are saved to `summary.json`. Use a fresh output directory. `CUDA_VISIBLE_DEVICES` selects the device; the benchmark does not change power modes or clocks. Avoid competing GPU workloads and record device settings when comparing machines.

This package was checked on an Ubuntu RTX 5090 host; all 12 jobs passed. These timings are a local execution example, not measurements for the manuscript's RTX 4080/Jetson Orin comparison. They are not TensorRT timings or whole-process memory measurements. The separate target-device measurement package retains the full manuscript hardware protocol.
