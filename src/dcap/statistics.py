"""Additive statistics and the five segmentation outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SufficientStatistics:
    """Pixel and object counts that can be summed across spatial blocks."""

    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    true_negative: int = 0
    small_detected: int = 0
    small_total: int = 0
    boundary_matched_prediction: int = 0
    boundary_prediction: int = 0
    boundary_matched_target: int = 0
    boundary_target: int = 0
    positive_brier_sum: float = 0.0
    positive_count: int = 0
    negative_brier_sum: float = 0.0
    negative_count: int = 0
    positive_score_histogram: tuple[int, ...] = ()
    negative_score_histogram: tuple[int, ...] = ()

    def __add__(self, other: "SufficientStatistics") -> "SufficientStatistics":
        if not isinstance(other, SufficientStatistics):
            return NotImplemented
        values = {}
        for name in self.__dataclass_fields__:
            left = getattr(self, name)
            right = getattr(other, name)
            if name.endswith("_score_histogram"):
                if not left:
                    values[name] = right
                elif not right:
                    values[name] = left
                elif len(left) != len(right):
                    raise ValueError("Score histograms must have equal lengths")
                else:
                    values[name] = tuple(
                        a + b for a, b in zip(left, right, strict=True)
                    )
            else:
                values[name] = left + right
        return SufficientStatistics(**values)


def sum_statistics(records: Iterable[SufficientStatistics]) -> SufficientStatistics:
    total = SufficientStatistics()
    for record in records:
        total = total + record
    return total


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def compute_outcomes(statistics: SufficientStatistics) -> dict[str, float]:
    """Compute the primary predictive/reliability outcomes."""

    tp = statistics.true_positive
    fp = statistics.false_positive
    fn = statistics.false_negative
    dice = _safe_ratio(2 * tp, 2 * tp + fp + fn)
    small_recall = _safe_ratio(statistics.small_detected, statistics.small_total)
    boundary_precision = _safe_ratio(
        statistics.boundary_matched_prediction, statistics.boundary_prediction
    )
    boundary_recall = _safe_ratio(
        statistics.boundary_matched_target, statistics.boundary_target
    )
    # Missing reference is undefined; failure to predict an existing boundary is zero.
    if statistics.boundary_target == 0:
        boundary_f1 = float("nan")
    elif (
        statistics.boundary_prediction == 0 or boundary_precision + boundary_recall == 0
    ):
        boundary_f1 = 0.0
    else:
        boundary_f1 = (
            2
            * boundary_precision
            * boundary_recall
            / (boundary_precision + boundary_recall)
        )
    positive_brier = _safe_ratio(
        statistics.positive_brier_sum, statistics.positive_count
    )
    negative_brier = _safe_ratio(
        statistics.negative_brier_sum, statistics.negative_count
    )
    balanced_brier = (
        0.5 * (positive_brier + negative_brier)
        if not (np.isnan(positive_brier) or np.isnan(negative_brier))
        else float("nan")
    )
    if statistics.positive_score_histogram:
        positive_histogram = np.asarray(
            statistics.positive_score_histogram, dtype=np.float64
        )
        negative_histogram = np.asarray(
            statistics.negative_score_histogram, dtype=np.float64
        )
        if positive_histogram.shape != negative_histogram.shape:
            raise ValueError("Positive and negative score histograms must align")
        true_positive = np.cumsum(positive_histogram[::-1])
        false_positive = np.cumsum(negative_histogram[::-1])
        precision = true_positive / np.maximum(true_positive + false_positive, 1)
        average_precision = _safe_ratio(
            float(np.sum(precision * positive_histogram[::-1])),
            float(positive_histogram.sum()),
        )
    else:
        average_precision = float("nan")
    return {
        "dice": dice,
        "average_precision": average_precision,
        "small_object_recall": small_recall,
        "boundary_f1": boundary_f1,
        "balanced_brier_score": balanced_brier,
    }
