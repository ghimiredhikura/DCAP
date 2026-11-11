"""Losses that respect unobservable pixels."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def masked_bce_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    positive_weight: float,
    dice_weight: float = 0.5,
) -> torch.Tensor:
    valid = target != 255
    if not valid.any():
        raise ValueError("Batch contains no valid target pixels")
    logits_valid = logits.squeeze(1)[valid]
    target_valid = target[valid].to(torch.float32)
    bce = F.binary_cross_entropy_with_logits(
        logits_valid,
        target_valid,
        pos_weight=torch.as_tensor(
            positive_weight, device=logits.device, dtype=logits.dtype
        ),
    )
    probabilities = torch.sigmoid(logits_valid)
    intersection = (probabilities * target_valid).sum()
    dice_loss = 1 - (2 * intersection + 1.0) / (
        probabilities.sum() + target_valid.sum() + 1.0
    )
    return (1 - dice_weight) * bce + dice_weight * dice_loss


def _zero_loss(logits: torch.Tensor) -> torch.Tensor:
    """Return a differentiable zero for an all-ignore batch."""
    return logits.sum() * 0.0


def masked_balanced_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Normalize focal positive and negative terms independently.

    Separate normalization prevents the much larger provisional-background
    population from determining the scale of the positive contribution.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if gamma < 0.0:
        raise ValueError("gamma must be non-negative")
    scores = logits.squeeze(1)
    valid = target != 255
    if not valid.any():
        return _zero_loss(logits)
    labels = target.to(dtype=logits.dtype)
    positive = valid & (target == 1)
    negative = valid & (target == 0)
    positive_loss = _zero_loss(logits)
    negative_loss = _zero_loss(logits)
    if positive.any():
        positive_bce = F.binary_cross_entropy_with_logits(
            scores[positive], labels[positive], reduction="none"
        )
        positive_probability = torch.sigmoid(scores[positive])
        positive_loss = (
            alpha * (1.0 - positive_probability).pow(gamma) * positive_bce
        ).mean()
    if negative.any():
        negative_bce = F.binary_cross_entropy_with_logits(
            scores[negative], labels[negative], reduction="none"
        )
        negative_probability = torch.sigmoid(scores[negative])
        negative_loss = (
            (1.0 - alpha) * negative_probability.pow(gamma) * negative_bce
        ).mean()
    return positive_loss + negative_loss


def masked_per_sample_tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.30,
    beta: float = 0.70,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Average Tversky loss across samples that contain positive pixels."""
    if alpha < 0.0 or beta < 0.0:
        raise ValueError("Tversky weights must be non-negative")
    if smooth <= 0.0:
        raise ValueError("smooth must be positive")
    scores = logits.squeeze(1)
    probabilities = torch.sigmoid(scores)
    losses: list[torch.Tensor] = []
    for sample_index in range(target.shape[0]):
        valid = target[sample_index] != 255
        positive = valid & (target[sample_index] == 1)
        if not positive.any():
            continue
        labels = target[sample_index].to(dtype=logits.dtype)
        sample_probabilities = probabilities[sample_index][valid]
        sample_labels = labels[valid]
        true_positive = (sample_probabilities * sample_labels).sum()
        false_positive = (sample_probabilities * (1.0 - sample_labels)).sum()
        false_negative = ((1.0 - sample_probabilities) * sample_labels).sum()
        tversky = (true_positive + smooth) / (
            true_positive + alpha * false_positive + beta * false_negative + smooth
        )
        losses.append(1.0 - tversky)
    if not losses:
        return _zero_loss(logits)
    return torch.stack(losses).mean()


def masked_balanced_focal_tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    focal_weight: float = 0.40,
    tversky_weight: float = 0.60,
    focal_alpha: float = 0.75,
    focal_gamma: float = 2.0,
    tversky_alpha: float = 0.30,
    tversky_beta: float = 0.70,
    smooth: float = 1.0,
) -> torch.Tensor:
    if focal_weight < 0.0 or tversky_weight < 0.0:
        raise ValueError("Loss weights must be non-negative")
    weight_sum = focal_weight + tversky_weight
    if weight_sum <= 0.0:
        raise ValueError("At least one loss weight must be positive")
    focal = masked_balanced_focal_loss(
        logits, target, alpha=focal_alpha, gamma=focal_gamma
    )
    tversky = masked_per_sample_tversky_loss(
        logits,
        target,
        alpha=tversky_alpha,
        beta=tversky_beta,
        smooth=smooth,
    )
    return (focal_weight * focal + tversky_weight * tversky) / weight_sum
