"""Loss selection, exponential moving averages, schedules and training metrics."""

import argparse
import contextlib
import math
import random

import numpy as np
import torch

from .losses import masked_balanced_focal_tversky_loss, masked_bce_dice_loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be in (0, 1)")
        self.decay = decay
        self.shadow = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, value in model.state_dict().items():
            if value.is_floating_point():
                self.shadow[name].mul_(self.decay).add_(
                    value.detach(), alpha=1.0 - self.decay
                )
            else:
                self.shadow[name].copy_(value.detach())

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: value.clone() for name, value in self.shadow.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        if state.keys() != self.shadow.keys():
            raise ValueError("EMA state keys do not match the model")
        self.shadow = {name: value.clone() for name, value in state.items()}

    @contextlib.contextmanager
    def average_parameters(self, model: torch.nn.Module):
        original = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }
        model.load_state_dict(self.shadow)
        try:
            yield
        finally:
            model.load_state_dict(original)


def learning_rate_factor(epoch_index: int, epochs: int, warmup_epochs: int) -> float:
    if warmup_epochs and epoch_index < warmup_epochs:
        return float(epoch_index + 1) / warmup_epochs
    cosine_epochs = max(epochs - warmup_epochs, 1)
    progress = min(max(epoch_index - warmup_epochs, 0), cosine_epochs - 1)
    denominator = max(cosine_epochs - 1, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress / denominator))


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_name: str,
    positive_weight: float,
    args: argparse.Namespace,
) -> torch.Tensor:
    if loss_name == "bce_dice":
        return masked_bce_dice_loss(
            logits, targets, positive_weight, dice_weight=args.overlap_weight
        )
    if loss_name == "balanced_focal_tversky":
        return masked_balanced_focal_tversky_loss(
            logits,
            targets,
            focal_weight=args.focal_weight,
            tversky_weight=args.tversky_weight,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            tversky_alpha=args.tversky_alpha,
            tversky_beta=args.tversky_beta,
            smooth=args.loss_smooth,
        )
    raise ValueError(f"Unsupported loss: {loss_name}")


class Metrics:
    def __init__(self):
        self.pos = torch.zeros(1001, dtype=torch.float64, device="cuda")
        self.neg = torch.zeros_like(self.pos)

    @torch.no_grad()
    def update(self, logits, target):
        valid = target != 255
        p = torch.sigmoid(logits[:, 0])[valid].double()
        lab = target[valid] == 1
        bins = (p * 1000).long().clamp(0, 1000)
        self.pos += torch.bincount(bins[lab], minlength=1001)
        self.neg += torch.bincount(bins[~lab], minlength=1001)

    def compute(self):
        pos, neg = (self.pos.cpu().numpy(), self.neg.cpu().numpy())
        tp, fp = (pos[::-1].cumsum()[::-1], neg[::-1].cumsum()[::-1])
        dice = 2 * tp / np.maximum(tp + fp + pos.sum(), 1)
        cut = int(np.argmax(dice[1:-1])) + 1
        return dict(
            ap=float((tp / np.maximum(tp + fp, 1) * pos).sum() / max(pos.sum(), 1)),
            optimized_dice=float(dice[cut]),
            threshold=cut / 1000,
        )


@torch.no_grad()
def recalibrate_training_bn(model, batches, verify_parameters=False):
    modes = {module: module.training for module in model.modules()}
    bns = {
        module: module.momentum
        for module in model.modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    }
    parameters = (
        {name: p.detach().clone() for name, p in model.named_parameters()}
        if verify_parameters
        else None
    )
    model.eval()
    for bn in bns:
        bn.reset_running_stats()
        bn.train()
    seen = 0
    try:
        for image, _, _ in batches:
            for bn in bns:
                bn.momentum = len(image) / (seen + len(image))
            with torch.autocast("cuda", dtype=torch.float16):
                model(image.cuda())
            seen += len(image)
    finally:
        for bn, momentum in bns.items():
            bn.momentum = momentum
        for module, mode in modes.items():
            module.training = mode
    if verify_parameters:
        assert all(
            (torch.equal(p, parameters[name]) for name, p in model.named_parameters())
        )
    return dict(
        training_images=seen,
        batchnorm_layers=len(bns),
        parameters_unchanged=(True if verify_parameters else None),
    )
