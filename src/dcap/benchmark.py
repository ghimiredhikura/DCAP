"""Ubuntu CUDA benchmark, using the same loader and forward path as evaluation."""

import gc
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .models import load_ensemble, probability_mean
from .paths import ROOT, read, sha256, write


def run(output, models=None, smoke=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    completed = []
    for model in ["dense", "dcap70"]:
        for mode in ["fp32_eager", "fp32_graph", "amp"]:
            for batch in [1, 4]:
                target = output / f"{model}_{mode}_b{batch}.json"
                if target.exists():
                    raise FileExistsError(
                        "Choose a fresh output directory: " + str(target)
                    )
                cmd = [
                    sys.executable,
                    "-m",
                    "dcap",
                    "benchmark",
                    "--worker",
                    "--ensemble",
                    model,
                    "--mode",
                    mode,
                    "--batch",
                    str(batch),
                    "--output",
                    str(target),
                ]
                if models:
                    cmd += ["--models-root", str(models)]
                if smoke:
                    cmd += ["--smoke"]
                env = os.environ.copy()
                env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
                env["DCAP_ROOT"] = str(ROOT)
                with target.with_suffix(".log").open("w") as stream:
                    p = subprocess.run(
                        cmd, env=env, stdout=stream, stderr=subprocess.STDOUT
                    )
                if p.returncode:
                    raise RuntimeError(
                        "Benchmark failed; see " + str(target.with_suffix(".log"))
                    )
                completed.append(read(target))
                print(target.name, "passed", flush=True)
    write(
        output / "summary.json",
        dict(
            device_scope="Current host only; not manuscript RTX 4080/Orin evidence",
            smoke=smoke,
            results=completed,
        ),
    )
    return dict(jobs=len(completed), output=str(output), all_passed=True)


def worker(name, mode, batch, output, models=None, smoke=False):
    if not torch.cuda.is_available():
        raise RuntimeError("Ubuntu benchmark requires CUDA-enabled PyTorch")
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    members, records = load_ensemble(name, models, "cuda")
    with np.load(ROOT / "data/example.npz", allow_pickle=False) as f:
        cpu = torch.from_numpy(f["input"].copy())
        refs = {k: f[k].copy() for k in f.files if k.startswith(name + "_")}
    x = cpu.repeat(batch, 1, 1, 1).cuda()
    parity = []

    def check(actual, expected, mean, maxerr, label):
        delta = np.abs(actual.detach().float().cpu().numpy() - expected)
        result = dict(label=label, mean=float(delta.mean()), maximum=float(delta.max()))
        if (
            not np.isfinite(delta).all()
            or result["mean"] > mean
            or result["maximum"] > maxerr
        ):
            raise RuntimeError("Parity failed: " + str(result))
        parity.append(result)

    with torch.inference_mode():
        for model, record in zip(members, records):
            for i, c in enumerate([x, x.flip(-1).contiguous()]):
                ref = np.repeat(
                    refs[f'{name}_seed{record["seed"]}_case{i}'], batch, axis=0
                )
                check(model(c).sigmoid(), ref, 1e-5, 0.001, "FP32 member")
                if mode == "amp":
                    with torch.autocast("cuda", dtype=torch.float16):
                        p = model(c).sigmoid()
                    check(p, ref, 0.002, 0.02, "AMP member")
                    del p
        del c, ref, model
        fn = lambda: probability_mean(members, x, "amp" if mode == "amp" else "fp32")
        expected = fn().detach().cpu()
        changed = x.flip(-1).contiguous()
        expected_changed = probability_mean(
            members, changed, "amp" if mode == "amp" else "fp32"
        ).cpu()
        graph = None
        if mode == "fp32_graph":
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    fn()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                static = fn()

            def once():
                graph.replay()
                return static

        else:
            once = fn
        torch.testing.assert_close(once().cpu(), expected, atol=1e-6, rtol=1e-5)
        saved = x.clone()
        x.copy_(changed)
        torch.testing.assert_close(once().cpu(), expected_changed, atol=1e-6, rtol=1e-5)
        if (expected - expected_changed).abs().max() < 1e-5:
            raise RuntimeError("Changed input has no observable effect")
        x.copy_(saved)
        del saved, changed, expected, expected_changed, refs
        gc.collect()
        torch.cuda.empty_cache()
        nblocks = 1 if smoke or mode == "amp" else 3
        n = 3 if smoke else 100
        warmup = 2 if smoke else 30
        duration = 0 if smoke or mode == "amp" else 2
        samples = []
        allocated = []
        reserved = []
        for _ in range(nblocks):
            for _ in range(warmup):
                once()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            block = []
            start = time.perf_counter()
            while len(block) < n or time.perf_counter() - start < duration:
                torch.cuda.synchronize()
                t = time.perf_counter_ns()
                once()
                torch.cuda.synchronize()
                block.append((time.perf_counter_ns() - t) / 1e6)
            samples.append(block)
            allocated.append(torch.cuda.max_memory_allocated())
            reserved.append(torch.cuda.max_memory_reserved())
    times = [t for b in samples for t in b]
    median = statistics.median(times)
    try:
        smi = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,temperature.gpu,power.limit,clocks.current.sm,clocks.current.memory",
                "--format=csv",
            ],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        smi = None
    result = dict(
        ensemble=name,
        mode=mode,
        batch=batch,
        smoke=smoke,
        device=torch.cuda.get_device_name(),
        python=platform.python_version(),
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
        TF32=False,
        checkpoint_hashes=[r["sha256"] for r in records],
        fixture_sha256=sha256(ROOT / "data/example.npz"),
        numpy=np.__version__,
        platform=platform.platform(),
        nvidia_smi=smi,
        parity=parity,
        changed_input_check_passed=True,
        protocol=dict(
            warmup=warmup, blocks=nblocks, minimum_samples=n, minimum_seconds=duration
        ),
        samples_ms=samples,
        block_medians_ms=[statistics.median(b) for b in samples],
        median_ms=median,
        images_per_second=batch * 1000 / median,
        torch_peak_allocated_bytes=max(allocated),
        torch_peak_reserved_bytes=max(reserved),
        scope="Three sequential forwards, sigmoid and FP32 probability mean; GPU-resident input, loading/transfers/capture excluded; synchronized wall time.",
    )
    write(output, result)
    return dict(passed=True, output=str(output), median_ms=median)
