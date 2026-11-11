"""SIRFP integration with fitting-only calibration and dependency-group pruning.

The greedy scorer is loaded from the separately downloaded pinned upstream
source. Vectorized Gram/edge statistics and physical pruning implement the
study's ResUNet adaptation; see docs/comparators.md for its scope.
"""

from pathlib import Path
import copy
import time
import torch
from torch import nn
from torch.nn import functional as F
import torch_pruning as tp
from .importance import LearnedGateGroupImportance
from .data import Dataset, loader
from .paths import sha256


@torch.no_grad()
def spatial_statistics(features):
    x = features.detach().float()
    flat = x.flatten(2)
    normalized = flat / (flat.abs().sum(-1, keepdim=True) + 1e-5)
    gram = torch.bmm(normalized, normalized.transpose(1, 2)).mean(0)
    kernel = x.new_tensor([[-0.5, 0, 0.5], [-1, 0, 1], [-0.5, 0, 0.5]])
    kx = kernel[None, None].repeat(x.shape[1], 1, 1, 1)
    ky = kernel.T[None, None].repeat(x.shape[1], 1, 1, 1)
    gx = F.conv2d(x, kx, padding=1, groups=x.shape[1])
    gy = F.conv2d(x, ky, padding=1, groups=x.shape[1])
    magnitude = (gx.square() + gy.square()).sqrt()
    variance = magnitude.square().mean((2, 3)) - magnitude.mean((2, 3)).square()
    # Nonnegative mathematical variance; guard floating-point cancellation.
    edge = variance.clamp_min(0).sqrt().mean(0)
    return gram, edge


def load_scorer(source):
    """Load only the scoring method from the user's pinned upstream checkout.

    This avoids importing the upstream training framework and does not copy
    its unlicensed source into this distribution. The file hash is checked
    before any source is compiled.
    """
    import ast
    from functools import partial
    from .paths import ROOT, read

    expected = read(ROOT / "configs/comparators.json")["sirfp"]["source_sha256"]
    path = Path(source) / "pruners/graph_pruner.py"
    if sha256(path) != expected:
        raise ValueError(
            "SIRFP source differs from the pinned version; see docs/comparators.md"
        )
    tree = ast.parse(path.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "GraphEdgePruner"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_para_score"
    )
    namespace = {"torch": torch, "deepcopy": copy.deepcopy}
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return partial(score_with_upstream, namespace["get_para_score"])


@torch.no_grad()
def score_with_upstream(function, gram, edge, edge_weight):
    from types import SimpleNamespace

    name = "backbone.layer" if edge_weight == 0.4 else "decoder.layer"
    if edge_weight not in (0.0, 0.4):
        raise ValueError("The study uses encoder/decoder edge weights 0.4/0.0")
    state = SimpleNamespace(eic={name: gram}, edge={name: edge})
    return function(state, name)


def encoder_layer(name):
    return name.startswith("encoder") or name.startswith("bridge")


@torch.no_grad()
def collect(model, batches, decay=0.999):
    accum = {}
    handles = []
    seen = []

    def hook(name):
        def update(module, inputs, output):
            gram, edge = spatial_statistics(output)
            if name not in accum:
                accum[name] = dict(
                    gram=torch.zeros_like(gram), edge=torch.zeros_like(edge), batches=0
                )
            accum[name]["gram"].mul_(decay).add_(gram, alpha=1 - decay)
            accum[name]["edge"].mul_(decay).add_(edge, alpha=1 - decay)
            accum[name]["batches"] += 1

        return update

    model.eval()
    try:
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                handles.append(module.register_forward_hook(hook(name)))
        for image, _, ids in batches:
            # Float32 avoids overflowing L1 sums on high-resolution feature maps.
            with torch.autocast("cuda", enabled=False):
                model(image.cuda())
            seen.extend(ids)
    finally:
        for handle in handles:
            handle.remove()
    assert accum and len(set(v["batches"] for v in accum.values())) == 1
    for value in accum.values():
        assert (
            torch.isfinite(value["gram"]).all() and torch.isfinite(value["edge"]).all()
        )
    return accum, seen


def prune(model, statistics, target, scorer, tolerance=0.012):
    """One-shot author-code scores; dependent BN utilities averaged per group.

    Author encoder/decoder thresholds remain separate. Search only physical
    MACs, never labels or validation scores. Each trial starts from the parent.
    """
    if not 0 < target < 1:
        raise ValueError("MAC reduction target must be strictly between 0 and 1")
    model.eval()
    example = torch.zeros(1, 21, 256, 256, device="cuda")
    baseline_macs, baseline_params = tp.utils.count_ops_and_params(model, example)
    named_scores = {
        name: scorer(value["gram"], value["edge"], 0.4 if encoder_layer(name) else 0.0)
        for name, value in statistics.items()
    }
    names = {module: name for name, module in model.named_modules()}
    modules = dict(model.named_modules())
    importance = LearnedGateGroupImportance(
        {modules[name]: value for name, value in named_scores.items()}
    )
    graph = tp.DependencyGraph().build_dependency(
        model, example_inputs=example, verbose=False
    )
    root_scores = {}
    for group in graph.get_all_groups(
        ignored_layers=[model.head], root_module_types=[nn.Conv2d]
    ):
        root = group[0].dep.layer
        score = importance(group)
        assert score is not None and len(score) == root.out_channels
        root_scores[names[root]] = score
    pools = {
        kind: torch.cat(
            [
                value
                for name, value in root_scores.items()
                if encoder_layer(name) == kind
            ]
        )
        .sort()
        .values
        for kind in [True, False]
    }
    low, high, best = 0.0, 0.98, None
    trace = []
    for step in range(15):
        ratio = (low + high) / 2
        candidate = copy.deepcopy(model).eval()
        modules = dict(candidate.named_modules())
        scores = {modules[name]: value for name, value in named_scores.items()}
        thresholds = {
            kind: values[min(int(len(values) * ratio), len(values) - 1)]
            for kind, values in pools.items()
        }
        ratios = {}
        minimums = {}
        retained = {}
        for name, value in root_scores.items():
            width = len(value)
            minimum = max(1, int(width * 0.02))
            remove = min(
                width - minimum, int((value <= thresholds[encoder_layer(name)]).sum())
            )
            # MetaPruner computes retained width by floor(C*(1-ratio)). An
            # interior quarter-channel avoids floating-point integer edges.
            ratios[modules[name]] = max(0.0, (remove - 0.25) / width)
            minimums[name] = minimum
            retained[name] = width - remove
        pruner = tp.pruner.MetaPruner(
            candidate,
            example,
            importance=LearnedGateGroupImportance(scores),
            global_pruning=False,
            pruning_ratio=0.0,
            pruning_ratio_dict=ratios,
            iterative_steps=1,
            max_pruning_ratio=0.999,
            ignored_layers=[candidate.head],
            round_to=1,
        )
        pruner.step()
        macs, params = tp.utils.count_ops_and_params(candidate, example)
        achieved = 1 - macs / baseline_macs
        assert (
            candidate.head.out_channels == 1
            and candidate.encoder1.conv1.in_channels == 21
        )
        assert all(
            m.out_channels >= 1 for m in candidate.modules() if isinstance(m, nn.Conv2d)
        )
        assert all(
            dict(candidate.named_modules())[name].out_channels >= minimum
            for name, minimum in minimums.items()
        )
        assert all(
            dict(candidate.named_modules())[name].out_channels == width
            for name, width in retained.items()
        )
        record = dict(
            step=step,
            channel_pruning_ratio=ratio,
            macs=int(macs),
            parameters=int(params),
            achieved_mac_reduction=float(achieved),
        )
        trace.append(record)
        error = abs(achieved - target)
        if best is None or error < best[0]:
            best = (error, candidate, record, pruner.DG.pruning_history())
        else:
            del candidate
        if achieved < target:
            low = ratio
        else:
            high = ratio
    error, selected, record, history = best
    ordered = sorted(trace, key=lambda row: row["channel_pruning_ratio"])
    assert all(
        a["achieved_mac_reduction"] <= b["achieved_mac_reduction"] + 1e-12
        for a, b in zip(ordered, ordered[1:])
    ), trace
    if error > tolerance:
        raise ValueError(
            f"SIRFP adaptation target infeasible within tolerance: target={target},error={error},trace={trace}"
        )
    return selected, dict(
        baseline_macs=int(baseline_macs),
        baseline_parameters=int(baseline_params),
        pruned_macs=record["macs"],
        pruned_parameters=record["parameters"],
        achieved_mac_reduction=record["achieved_mac_reduction"],
        absolute_target_error=error,
        physical_pruning_history=history,
        physical_trace=trace,
    )


def construct(teacher, target, cache, train, norm, job, design, output):
    started = time.perf_counter()
    by_id = {r["sample_id"]: r for r in train}
    ids = design["gate_ids"]
    assert len(ids) == 128 and set(ids) <= set(design["fit_ids"])
    assert not set(ids) & (set(design["validation_ids"]) | set(design["outer_ids"]))
    batches = loader(Dataset(cache, [by_id[sid] for sid in ids], norm))
    state_before = {
        k: v.detach().cpu().clone() for k, v in teacher.state_dict().items()
    }
    statistics, seen = collect(teacher, batches)
    assert seen == ids
    assert all(
        torch.equal(v.detach().cpu(), state_before[k])
        for k, v in teacher.state_dict().items()
    )
    torch.cuda.synchronize()
    calibration_seconds = time.perf_counter() - started
    evidence = output / "sirfp_calibration.pt"
    torch.save(
        {
            n: {
                k: v.cpu() if isinstance(v, torch.Tensor) else v
                for k, v in stats.items()
            }
            for n, stats in statistics.items()
        },
        evidence,
    )
    search_start = time.perf_counter()
    model, measurement = prune(
        teacher, statistics, target, load_scorer(job["sirfp_source"])
    )
    torch.cuda.synchronize()
    measurement.update(
        method_label="SIRFP ResUNet adaptation",
        calibration_ids=ids,
        calibration_sha256=sha256(evidence),
        ema_decay=0.999,
        edge_weights=dict(encoder=0.4, decoder=0.0),
        parent_unchanged=True,
        training_only=True,
        phase_seconds=dict(
            calibration=calibration_seconds,
            physical_search=time.perf_counter() - search_start,
        ),
        preparation_seconds=time.perf_counter() - started,
        preparation_peak_gpu_bytes=torch.cuda.max_memory_allocated(),
    )
    return model, measurement
