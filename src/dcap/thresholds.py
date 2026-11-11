"""Validation-only threshold fitting."""

import numpy as np


def select_threshold_from_histograms(
    positives: np.ndarray, negatives: np.ndarray
) -> float:
    if positives.shape != negatives.shape or positives.ndim != 1:
        raise ValueError("Threshold histograms must be aligned one-dimensional arrays")
    if positives.size < 3 or positives.sum() == 0:
        raise ValueError(
            "Threshold fitting requires positive labels and at least three bins"
        )
    true_positive = np.cumsum(positives[::-1])[::-1]
    false_positive = np.cumsum(negatives[::-1])[::-1]
    false_negative = positives.sum() - true_positive
    dice = (
        2
        * true_positive
        / np.maximum(2 * true_positive + false_positive + false_negative, 1)
    )
    return (int(np.argmax(dice[1:-1])) + 1) / (positives.size - 1)


def fit_leave_one_block_out_thresholds(
    probability_maps: list[np.ndarray],
    target_maps: list[np.ndarray],
    sample_ids: list[str],
    block_mapping: dict[str, str],
    *,
    bin_count: int = 1001,
) -> tuple[float, dict[str, float]]:
    """Fit every block's threshold using all other development blocks."""
    histograms: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sample_id, probability, target in zip(
        sample_ids, probability_maps, target_maps, strict=True
    ):
        block_id = block_mapping[sample_id]
        positive, negative = histograms.setdefault(
            block_id,
            (np.zeros(bin_count, dtype=np.int64), np.zeros(bin_count, dtype=np.int64)),
        )
        valid = target != 255
        indices = np.minimum(
            (probability[valid] * (bin_count - 1)).astype(np.int64), bin_count - 1
        )
        positive += np.bincount(indices[target[valid] == 1], minlength=bin_count)
        negative += np.bincount(indices[target[valid] == 0], minlength=bin_count)
    if len(histograms) < 2:
        raise ValueError(
            "Leave-one-block-out threshold fitting requires at least two blocks"
        )
    total_positive = sum(
        (value[0] for value in histograms.values()),
        start=np.zeros(bin_count, dtype=np.int64),
    )
    total_negative = sum(
        (value[1] for value in histograms.values()),
        start=np.zeros(bin_count, dtype=np.int64),
    )
    global_threshold = select_threshold_from_histograms(total_positive, total_negative)
    thresholds = {
        block_id: select_threshold_from_histograms(
            total_positive - positive, total_negative - negative
        )
        for block_id, (positive, negative) in sorted(histograms.items())
    }
    return (global_threshold, thresholds)
