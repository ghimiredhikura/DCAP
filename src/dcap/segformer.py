# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
# SegFormer-derived components: NVIDIA Source Code License (licenses/).
"""SegFormer-B2 for the study's 21-channel binary segmentation task.

Derived from NVlabs/SegFormer; see licenses/SegFormer-LICENSE.txt.
RGB kernels initialize post-event R/G/B channels (8, 7, 6); the other
input kernels start at zero and remain trainable.
"""

import re
import torch
from torch import nn
from torch.nn import functional as F
from .mit_backbone import mit_b2


def original_to_hf(key):
    key = re.sub(
        r"^patch_embed([1-4])", lambda m: f"patch_embeddings.{int(m[1])-1}", key
    )
    key = re.sub(r"^block([1-4])", lambda m: f"block.{int(m[1])-1}", key)
    key = re.sub(r"^norm([1-4])", lambda m: f"layer_norm.{int(m[1])-1}", key)
    key = key.replace(".norm1.", ".layer_norm_1.").replace(".norm2.", ".layer_norm_2.")
    key = key.replace(".norm.", ".layer_norm.")
    key = key.replace(".attn.q.", ".attention.self.query.").replace(
        ".attn.proj.", ".attention.output.dense."
    )
    key = key.replace(".attn.", ".attention.self.")
    key = key.replace(".fc1.", ".dense1.").replace(".fc2.", ".dense2.")
    return "segformer.encoder." + key


def convert_nvidia_hf_checkpoint(source):
    """Invert the published HF conversion; only discard the ImageNet head."""
    state = torch.load(source, map_location="cpu", weights_only=True)
    model = mit_b2()
    result, used, mapping = {}, set(), {}
    for key, expected in model.state_dict().items():
        name = original_to_hf(key)
        if ".attention.self.kv." in name:
            keys = [name.replace(".kv.", ".key."), name.replace(".kv.", ".value.")]
            value = torch.cat([state[k] for k in keys], dim=0)
        else:
            keys = [name]
            value = state[name]
        assert value.shape == expected.shape, (key, value.shape, expected.shape)
        result[key] = value
        used.update(keys)
        mapping[key] = keys
    assert set(state) - used == {"classifier.weight", "classifier.bias"}, (
        set(state) - used
    )
    model.load_state_dict(result, strict=True)
    return result, mapping


class LinearEmbedding(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.proj = nn.Linear(channels, 768)

    def forward(self, x):
        return self.proj(x.flatten(2).transpose(1, 2))


class SegFormerB2(nn.Module):
    def __init__(self, in_channels=21, pretrained_checkpoint=None):
        super().__init__()
        if in_channels not in (3, 21):
            raise ValueError(
                "Supported input channel counts are 3 (RGB) and 21 (study inputs)"
            )
        self.encoder = mit_b2()
        if pretrained_checkpoint:
            self.encoder.load_state_dict(
                torch.load(
                    pretrained_checkpoint, map_location="cpu", weights_only=True
                ),
                strict=True,
            )
        if in_channels == 21:
            old = self.encoder.patch_embed1.proj
            adapted = nn.Conv2d(
                21,
                old.out_channels,
                old.kernel_size,
                old.stride,
                old.padding,
                bias=True,
            )
            with torch.no_grad():
                adapted.weight.zero_()
                adapted.weight[:, [8, 7, 6]] = old.weight
                adapted.bias.copy_(old.bias)
            self.encoder.patch_embed1.proj = adapted
        # Match the official B2 decoder: four768-wide projections, reversed
        # concatenation,1x1 Conv-BN-ReLU, Dropout2d(.1), binary linear output.
        self.linear_c4 = LinearEmbedding(512)
        self.linear_c3 = LinearEmbedding(320)
        self.linear_c2 = LinearEmbedding(128)
        self.linear_c1 = LinearEmbedding(64)
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(3072, 768, 1, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
        )
        nn.init.kaiming_normal_(
            self.linear_fuse[0].weight, mode="fan_out", nonlinearity="relu"
        )
        self.dropout = nn.Dropout2d(0.1)
        self.head = nn.Conv2d(768, 1, 1)

    def forward(self, inputs):
        features = self.encoder(inputs)
        size = features[0].shape[-2:]
        projected = []
        for stage in [4, 3, 2, 1]:
            feature = features[stage - 1]
            x = (
                getattr(self, f"linear_c{stage}")(feature)
                .permute(0, 2, 1)
                .reshape(feature.shape[0], 768, *feature.shape[-2:])
            )
            if stage != 1:
                x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
            projected.append(x)
        logits = self.head(self.dropout(self.linear_fuse(torch.cat(projected, dim=1))))
        return F.interpolate(
            logits, size=inputs.shape[-2:], mode="bilinear", align_corners=False
        )

    def optimizer_parameter_groups(self, encoder_learning_rate, decoder_learning_rate):
        groups = {}
        norm_params = {
            id(p)
            for m in self.modules()
            if isinstance(m, (nn.LayerNorm, nn.BatchNorm2d))
            for p in m.parameters()
        }
        for name, param in self.named_parameters():
            lr = (
                encoder_learning_rate
                if name.startswith("encoder.")
                else decoder_learning_rate
            )
            decay = 0.0 if id(param) in norm_params else 0.01
            groups.setdefault((lr, decay), []).append(param)
        return [
            dict(params=params, lr=lr, weight_decay=decay)
            for (lr, decay), params in groups.items()
        ]

    def attention_matmul_macs(self, height, width):
        total = 0
        height, width = (height + 3) // 4, (width + 3) // 4
        for stage in range(1, 5):
            if stage > 1:
                height, width = (height + 1) // 2, (width + 1) // 2
            for block in getattr(self.encoder, f"block{stage}"):
                attention = block.attn
                keys = (height // attention.sr_ratio) * (width // attention.sr_ratio)
                total += 2 * height * width * keys * attention.dim
        return total


def backbone_digest(state):
    """Hash tensor names, shapes, dtypes and values independently of serialization."""
    import hashlib

    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(f"{name}:{tuple(value.shape)}:{value.dtype}\n".encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def prepare_pretraining(output, source=None):
    """Obtain the pinned NVIDIA weights and invert their published conversion."""
    from pathlib import Path
    import urllib.request
    from .paths import ROOT, read, sha256

    cfg = read(ROOT / "configs/comparators.json")["segformer_b2"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source = Path(source) if source else output / "pytorch_model.bin"
    if not source.exists():
        temporary = source.with_suffix(".part")
        urllib.request.urlretrieve(cfg["pretraining"]["source"], temporary)
        if sha256(temporary) != cfg["pretraining"]["sha256"]:
            raise ValueError("MiT-B2 download checksum mismatch")
        temporary.rename(source)
    if sha256(source) != cfg["pretraining"]["sha256"]:
        raise ValueError("MiT-B2 source checksum mismatch")
    state, _ = convert_nvidia_hf_checkpoint(source)
    if backbone_digest(state) != cfg["backbone_tensor_sha256"]:
        raise ValueError(
            "Converted backbone tensors differ from the study initialization"
        )
    target = output / "mit_b2_original_backbone.pt"
    if target.exists():
        current = torch.load(target, map_location="cpu", weights_only=True)
        if backbone_digest(current) != cfg["backbone_tensor_sha256"]:
            raise FileExistsError("Existing converted backbone differs")
    else:
        torch.save(state, target)
    return dict(checkpoint=str(target), tensors=len(state), sha256=sha256(target))
