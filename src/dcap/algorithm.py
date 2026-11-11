"""Dependency-shared gates and task-aware recovery for DCAP."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import torch
import torch_pruning as tp
from scipy import ndimage
from torch import nn
from torch.nn import functional as F

from .importance import LearnedGateGroupImportance
from .paths import study
from .training_helpers import compute_loss, recalibrate_training_bn

LOSS_ARGS = SimpleNamespace(
    loss_name="bce_dice",
    overlap_weight=0.5,
    focal_weight=0.4,
    tversky_weight=0.6,
    focal_alpha=0.75,
    focal_gamma=2.0,
    tversky_alpha=0.3,
    tversky_beta=0.7,
    loss_smooth=1.0,
    gradient_clip=1.0,
)


def supervised(logits, target):
    return compute_loss(logits.float(), target, "bce_dice", 20.0, LOSS_ARGS)


def balanced(value, target, valid):
    terms = [
        value[m].mean()
        for m in (valid & (target > 0.5), valid & (target <= 0.5))
        if m.any()
    ]
    return torch.stack(terms).mean() if terms else value.sum() * 0.0


def overlap(p, y, valid):
    mask = valid.float()
    dims = (1, 2, 3)
    return (
        1 - (2 * (p * y * mask).sum(dims) + 1) / ((p + y).mul(mask).sum(dims) + 1)
    ).mean()


def preservation_loss(
    logits, teacher_logits, target, settings=None, small_object_pixels=41
):
    """Combine supervision with five dense-teacher preservation penalties."""
    cfg = study()["recovery_loss"] if settings is None else settings
    if set(cfg) != set(study()["recovery_loss"]):
        raise ValueError("All five independent recovery weights are required")
    p = logits.float().sigmoid()
    q = teacher_logits.detach().float().sigmoid()
    valid = (target != 255).unsqueeze(1)
    y = target.clamp(0, 1).float().unsqueeze(1)
    task = supervised(logits, target)
    kd = balanced((p - q).square(), y, valid)
    brier = balanced((p - y).square(), y, valid)
    teacher_brier = balanced((q - y).square(), y, valid)
    dice_excess = F.relu(overlap(p, y, valid) - overlap(q, y, valid))
    dilated = F.max_pool2d(y, 5, 1, 2)
    eroded = -F.max_pool2d(-y, 5, 1, 2)
    boundary = (dilated - eroded > 0.5) & valid
    boundary_kd = balanced(
        F.smooth_l1_loss(p, q, reduction="none", beta=0.1), y, boundary
    )
    masks = []
    for array in target.detach().cpu().numpy():
        labels, n = ndimage.label(array == 1, np.ones((3, 3), bool))
        areas = np.bincount(labels.ravel())
        chosen = np.flatnonzero((areas <= small_object_pixels) & (areas > 0))
        chosen = chosen[chosen != 0]
        foreground = np.isin(labels, chosen)
        roi = ndimage.binary_dilation(foreground, iterations=2) & (array != 255)
        masks.append(roi)
    roi = torch.as_tensor(np.stack(masks), device=p.device).unsqueeze(1)
    small = F.relu(
        balanced((p - y).square(), y, roi) - balanced((q - y).square(), y, roi)
    )
    terms = dict(
        supervised=task,
        kd=kd,
        brier_excess=F.relu(brier - teacher_brier),
        dice_excess=dice_excess,
        boundary_kd=boundary_kd,
        small_excess=small,
    )
    loss = task + sum((float(cfg[k]) * terms[k] for k in cfg))
    return (loss, terms)


class SharedGateBank(nn.Module):
    """Share channel gates across linked layers and apply them at BN outputs."""

    def __init__(self, model, example):
        super().__init__()
        self.temperature = 1.0
        self.handles = []
        self.layouts = {}
        self.names = {m: n for n, m in model.named_modules()}
        self.layer_map = dict(model.named_modules())
        self.conv_in = {}
        self.conv_out = {}
        self.bn_maps = {}
        self.roots = {}
        self.spatial = {}
        for n, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                assert (
                    m.groups == 1
                ), "Grouped convolution requires a different MAC estimator"
                self.conv_in[n] = [None] * m.in_channels
                self.conv_out[n] = [None] * m.out_channels
        graph = tp.DependencyGraph().build_dependency(
            model, example_inputs=example, verbose=False
        )
        groups = list(
            graph.get_all_groups(
                ignored_layers=[model.head], root_module_types=[nn.Conv2d]
            )
        )
        params = {}
        fn = tp.pruner.function
        for i, group in enumerate(groups):
            root = group[0].dep.layer
            key = f"g{i:03d}"
            width = root.out_channels
            params[key] = nn.Parameter(torch.full((width,), 3.5, device=example.device))
            self.roots[key] = self.names[root]
            for item in group:
                dep, indices = item
                name = self.names.get(dep.layer)
                if name is None:
                    continue
                if dep.pruning_fn == fn.prune_batchnorm_out_channels:
                    dest = self.bn_maps.setdefault(
                        name, [None] * dep.layer.num_features
                    )
                elif dep.pruning_fn == fn.prune_conv_out_channels:
                    dest = self.conv_out[name]
                elif dep.pruning_fn == fn.prune_conv_in_channels:
                    dest = self.conv_in[name]
                else:
                    continue
                for local, root_idx in zip(indices, item.root_idxs, strict=True):
                    token = (key, root_idx)
                    assert dest[local] in (None, token), (
                        name,
                        local,
                        dest[local],
                        token,
                    )
                    dest[local] = token
        self.logits = nn.ParameterDict(params)
        for name, m in model.named_modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                assert name in self.bn_maps and all(
                    (t is not None for t in self.bn_maps[name])
                ), name
        hs = []
        for name, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                hs.append(
                    m.register_forward_hook(
                        lambda module, inp, out, n=name: self.spatial.__setitem__(
                            n, int(out.shape[-2] * out.shape[-1])
                        )
                    )
                )
        with torch.no_grad():
            model(example)
        for h in hs:
            h.remove()
        self.dense_conv_macs = sum(
            (
                self.spatial[n]
                * m.kernel_size[0]
                * m.kernel_size[1]
                * m.in_channels
                * m.out_channels
                for n, m in model.named_modules()
                if isinstance(m, nn.Conv2d)
            )
        )

    def gate(self, key):
        if hasattr(self, "hard_masks"):
            return self.hard_masks[key]
        return self.logits[key].div(self.temperature).sigmoid()

    def values(self, tokens):
        signature = tuple(tokens)
        if signature not in self.layouts:
            layout = {}
            for position, token in enumerate(tokens):
                if token is not None:
                    key, index = token
                    entry = layout.setdefault(key, ([], []))
                    entry[0].append(position)
                    entry[1].append(index)
            device = next(iter(self.logits.values())).device
            self.layouts[signature] = {
                key: (torch.tensor(a, device=device), torch.tensor(b, device=device))
                for key, (a, b) in layout.items()
            }
        result = next(iter(self.logits.values())).new_ones(len(tokens))
        for key, (positions, indices) in self.layouts[signature].items():
            result = result.index_copy(0, positions, self.gate(key)[indices])
        return result

    def mac_fraction(self):
        result = next(iter(self.logits.values())).sum() * 0.0
        for name, tokens in self.conv_out.items():
            m = self.layer_map[name]
            result = (
                result
                + self.spatial[name]
                * m.kernel_size[0]
                * m.kernel_size[1]
                * self.values(self.conv_in[name]).sum()
                * self.values(tokens).sum()
            )
        return result / self.dense_conv_macs

    def attach(self):
        assert not self.handles
        for name, tokens in self.bn_maps.items():
            self.handles.append(
                self.layer_map[name].register_forward_hook(
                    lambda module, inp, out, t=tokens: (
                        out * self.values(t).view(1, -1, 1, 1)
                    )
                )
            )

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def scores(self):
        return {
            name: self.values(tokens).detach().cpu().tolist()
            for name, tokens in self.bn_maps.items()
        }

    def densities(self):
        return {
            self.roots[key]: float(self.gate(key).detach().mean())
            for key in self.logits
        }

    def audit(self):
        return dict(
            dependency_groups=len(self.logits),
            batchnorm_sites=len(self.bn_maps),
            dense_convolution_macs=self.dense_conv_macs,
            roots=self.roots,
        )


def search_shared(
    model, teacher, batches, target, epochs=18, objective="full", small_object_pixels=41
):
    """Learn shared gates while dense weights and BN statistics stay fixed."""
    model.eval()
    teacher.eval()
    example = torch.zeros(1, 21, 256, 256, device="cuda")
    for p in model.parameters():
        p.requires_grad_(True)
    bank = SharedGateBank(model, example)
    audit = bank.audit()
    for p in model.parameters():
        p.requires_grad_(False)
    bank.attach()
    optim = torch.optim.Adam(bank.parameters(), lr=0.035)
    history = []
    try:
        for epoch in range(1, epochs + 1):
            progress = (epoch - 1) / max(epochs - 1, 1)
            bank.temperature = 1.5 * (0.6 / 1.5) ** progress
            desired = 1 - target * min(1.0, epoch / epochs / 0.65)
            sums = {}
            n = 0
            for image, mask, _ in batches:
                image, mask = (image.cuda(), mask.cuda())
                optim.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    with torch.no_grad():
                        q = teacher(image)
                    logits = model(image)
                    if objective == "supervised":
                        loss = supervised(logits, mask)
                        terms = {"supervised": loss}
                    elif objective == "full":
                        loss, terms = preservation_loss(
                            logits, q, mask, small_object_pixels=small_object_pixels
                        )
                    else:
                        raise ValueError(objective)
                    frac = bank.mac_fraction()
                    budget = (frac - desired).square()
                    polarization = torch.cat([bank.gate(k) for k in bank.logits])
                    polarization = (polarization * (1 - polarization)).mean()
                    total = loss + (8 + 24 * progress) * budget + 0.01 * polarization
                assert torch.isfinite(total)
                total.backward()
                nn.utils.clip_grad_norm_(bank.parameters(), 5)
                optim.step()
                for k, v in dict(
                    loss=total, soft_mac_fraction=frac, budget=budget, **terms
                ).items():
                    sums[k] = sums.get(k, 0) + float(v.detach())
                n += 1
            row = dict(
                epoch=epoch,
                desired_mac_fraction=desired,
                temperature=bank.temperature,
                **{k: v / n for k, v in sums.items()},
            )
            history.append(row)
            print("SHARED_GATE " + json.dumps(row), flush=True)
        scores, densities = (bank.scores(), bank.densities())
    finally:
        bank.detach()
        for p in model.parameters():
            p.requires_grad_(True)
    return (scores, densities, dict(audit=audit, history=history))


def maximum_group_pruning_ratio(width, minimum_channels):
    if width < 1 or minimum_channels < 1:
        raise ValueError("Channel widths must be positive")
    return max(0.0, 1 - minimum_channels / width)


def physical_prune(
    model, scores, densities, target, dense_macs=None, minimum_channels=8
):
    """Remove channels and search for the requested measured MAC reduction.

    Gate scores rank channels; gate densities allocate removal across groups.
    Each candidate is measured after physical pruning.
    """
    example = torch.zeros(1, 21, 256, 256, device="cuda")
    base_macs, base_params = tp.utils.count_ops_and_params(model, example)
    reference = dense_macs or base_macs
    assert 1 - base_macs / reference < target
    modules = dict(model.named_modules())
    low, high = (
        0.0,
        max(
            (
                maximum_group_pruning_ratio(modules[n].out_channels, minimum_channels)
                / max(1 - density, 1e-06)
                for n, density in densities.items()
            )
        ),
    )
    best = None
    trace = []
    for step in range(13):
        scale = (low + high) / 2
        c = copy.deepcopy(model).eval()
        mods = dict(c.named_modules())
        by_module = {mods[n]: torch.tensor(v, device="cuda") for n, v in scores.items()}
        ratios = {
            mods[n]: min(
                max(scale * (1 - density), 0.0),
                maximum_group_pruning_ratio(mods[n].out_channels, minimum_channels),
            )
            for n, density in densities.items()
        }
        pruner = tp.pruner.MetaPruner(
            c,
            example,
            importance=LearnedGateGroupImportance(by_module),
            global_pruning=False,
            pruning_ratio=0.0,
            pruning_ratio_dict=ratios,
            iterative_steps=1,
            max_pruning_ratio=0.99,
            ignored_layers=[c.head],
            round_to=1,
        )
        pruner.step()
        macs, params = tp.utils.count_ops_and_params(c, example)
        achieved = 1 - macs / reference
        assert c.head.out_channels == 1 and c.encoder1.conv1.in_channels == 21
        assert all(
            (
                m.out_channels >= min(minimum_channels, modules[name].out_channels)
                for name, m in c.named_modules()
                if isinstance(m, nn.Conv2d) and m is not c.head
            )
        )
        trace.append(
            dict(
                step=step,
                scale=scale,
                mac_reduction=float(achieved),
                macs=int(macs),
                parameters=int(params),
            )
        )
        error = abs(achieved - target)
        if best is None or error < best[0]:
            best = (error, c, trace[-1], pruner.DG.pruning_history())
        else:
            del c
        if achieved < target:
            low = scale
        else:
            high = scale
    error, c, record, history = best
    assert error <= 0.012, (target, error, trace)
    measurement = dict(
        baseline_macs=int(reference),
        pruned_macs=record["macs"],
        pruned_parameters=record["parameters"],
        achieved_mac_reduction=record["mac_reduction"],
        absolute_target_error=error,
        minimum_channels=minimum_channels,
        training_only=True,
        physical_pruning_history=history,
    )
    return (c, measurement, trace)


def absorb_gate_affine(dense, pruned, scores, history):
    bank = SharedGateBank(dense, torch.zeros(1, 21, 256, 256, device="cuda"))
    removed = {name: set(indices) for name, out, indices in history if out}
    baked = copy.deepcopy(pruned)
    modules = dict(baked.named_modules())
    with torch.no_grad():
        for name, tokens in bank.bn_maps.items():
            values = [
                value
                for value, (key, index) in zip(scores[name], tokens, strict=True)
                if index not in removed.get(bank.roots[key], set())
            ]
            bn = modules[name]
            assert len(values) == bn.num_features, (name, len(values), bn.num_features)
            gate = torch.tensor(values, device=bn.weight.device)
            bn.weight.mul_(gate)
            bn.bias.mul_(gate)
    return baked


def select_gate_inheritance(
    dense,
    pruned,
    teacher,
    scores,
    history,
    bn_batches,
    scoring_batches,
    small_object_pixels=41,
):
    """Choose gate-amplitude inheritance using a separate fitting subset.

    Dropping gate amplitudes after using them to optimize a network can create
    a soft-to-physical mismatch. BN affine scaling exactly absorbs each retained
    post-BN gate before recalibration. Both alternatives use the same fitting-only BN
    recalibration and selection objective.
    """
    baked = absorb_gate_affine(dense, pruned, scores, history)
    ranked = []
    for label, candidate in (("raw_weights", pruned), ("retained_gate_affine", baked)):
        recalibrate_training_bn(candidate, bn_batches)
        candidate.eval()
        total = count = 0.0
        terms_sum = {}
        with torch.no_grad():
            for image, target, _ in scoring_batches:
                image, target = (image.cuda(), target.cuda())
                with torch.autocast("cuda", dtype=torch.float16):
                    loss, terms = preservation_loss(
                        candidate(image),
                        teacher(image),
                        target,
                        small_object_pixels=small_object_pixels,
                    )
                total += float(loss) * len(image)
                count += len(image)
                for key, value in terms.items():
                    terms_sum[key] = terms_sum.get(key, 0) + float(value) * len(image)
        ranked.append(
            (
                total / count,
                label,
                candidate,
                {k: v / count for k, v in terms_sum.items()},
            )
        )
    chosen = min(ranked, key=lambda r: (r[0], r[1]))
    record = dict(
        selected=chosen[1],
        scores={r[1]: dict(total=r[0], components=r[3]) for r in ranked},
        both_training_bn_refreshed=True,
        scoring_training_chips=count,
        gate_fit_and_selection_samples_disjoint=True,
    )
    return (chosen[2], record)
