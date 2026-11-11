"""Portable tensor-only checkpoints with explicit layer configurations."""

from pathlib import Path

import torch
from torch import nn

from .architectures import ResNet34DeepLabV3Plus, ResUNet, VariableWidthResUNet
from .paths import ROOT, model_root, read, sha256


def build_model(configuration, pretrained=False):
    architecture = configuration.get("architecture", "resunet")
    channels = configuration.get("input_channels", 21)
    if architecture == "resnet34_deeplabv3plus":
        return ResNet34DeepLabV3Plus(
            channels,
            configuration.get("input_mode", "reflectance_change"),
            pretrained=pretrained,
        )
    if architecture == "compact_resunet":
        return VariableWidthResUNet(
            channels,
            configuration["base_channels"],
            configuration["bridge_channels"],
            "batch",
        )
    if architecture != "resunet":
        raise ValueError(f"Unsupported architecture: {architecture}")
    return ResUNet(channels, 32, "batch")


def layer_specs(model):
    result = {}
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            result[name] = dict(
                kind="conv2d",
                in_channels=layer.in_channels,
                out_channels=layer.out_channels,
                kernel_size=list(layer.kernel_size),
                stride=list(layer.stride),
                padding=list(layer.padding),
                dilation=list(layer.dilation),
                groups=layer.groups,
                bias=layer.bias is not None,
                padding_mode=layer.padding_mode,
            )
        elif isinstance(layer, nn.BatchNorm2d):
            result[name] = dict(
                kind="batchnorm2d",
                num_features=layer.num_features,
                eps=layer.eps,
                momentum=layer.momentum,
                affine=layer.affine,
                track_running_stats=layer.track_running_stats,
            )
    return result


def save_checkpoint(model, path, configuration, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        format="dcap-state-dict-v1",
        configuration=configuration,
        layers=layer_specs(model),
        state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        **metadata,
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "dcap-state-dict-v1":
        raise ValueError("Expected a DCAP tensor-only release checkpoint")
    # Loading supplied weights must not consume the caller's training RNG.
    with torch.random.fork_rng(devices=[]):
        model = build_model(payload["configuration"], pretrained=False)
        for name, settings in payload["layers"].items():
            values = dict(settings)
            kind = values.pop("kind")
            layer = (
                nn.Conv2d(**values) if kind == "conv2d" else nn.BatchNorm2d(**values)
            )
            parent_name, _, key = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, key, layer)
        model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval(), payload


def catalog():
    return read(ROOT / "metadata/models.json")


def ensemble_paths(name, directory=None):
    entry = catalog()["ensembles"].get(name)
    if entry is None:
        raise ValueError(f"Unknown ensemble {name}; run 'python -m dcap models list'")
    root = model_root(directory)
    paths = [root / asset for asset in entry["assets"]]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; install the model assets first")
    return paths, entry


def load_ensemble(name, directory=None, device="cpu"):
    paths, entry = ensemble_paths(name, directory)
    records = {m["asset"]: m for m in catalog()["models"]}
    for p in paths:
        if sha256(p) != records[p.name]["sha256"]:
            raise ValueError("Checkpoint checksum mismatch: " + str(p))
    return [load_checkpoint(p, device)[0] for p in paths], [
        records[p.name] for p in paths
    ]


def probability_mean(members, inputs, precision="amp", numerics=None):
    """Shared inference path: sigmoid in output dtype, then FP32 probability mean."""
    previous = torch.are_deterministic_algorithms_enabled()
    values = []
    try:
        with torch.autocast(
            inputs.device.type, dtype=torch.float16, enabled=precision == "amp"
        ):
            for i, model in enumerate(members):
                if numerics is not None:
                    torch.use_deterministic_algorithms(numerics[i])
                values.append(model(inputs).sigmoid().float())
        return torch.stack(values).mean(0)
    finally:
        if numerics is not None:
            torch.use_deterministic_algorithms(previous)


def install_assets(source, directory=None, group="main"):
    import shutil

    root = model_root(directory)
    root.mkdir(parents=True, exist_ok=True)
    notice = read(ROOT / "metadata/notices.json")["models"]
    source_notice = Path(source) / notice["asset"]
    if sha256(source_notice) != notice["sha256"]:
        raise ValueError("Model notice checksum mismatch")
    target_notice = root / source_notice.name
    if source_notice.resolve() != target_notice.resolve():
        shutil.copy2(source_notice, target_notice)
    chosen = [
        m for m in catalog()["models"] if group == "all" or m["asset_group"] == group
    ]
    for record in chosen:
        path = Path(source) / record["asset"]
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Asset checksum mismatch: {path.name}")
        target = root / path.name
        if target.exists() and sha256(target) != record["sha256"]:
            raise FileExistsError(f"Refusing to overwrite different weights: {target}")
        if not target.exists():
            shutil.copy2(path, target)
    return dict(installed=len(chosen), directory=str(root))
