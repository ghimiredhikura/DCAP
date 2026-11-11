"""Convert a probability map and reference mask into additive statistics."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .statistics import SufficientStatistics


def _boundary(mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3)), border_value=0)
    return np.logical_and(mask, np.logical_not(eroded))


def _matched_boundary_counts(
    prediction: np.ndarray, target: np.ndarray, tolerance_pixels: int
) -> tuple[int, int, int, int]:
    predicted_boundary = _boundary(prediction)
    target_boundary = _boundary(target)
    if tolerance_pixels < 0:
        raise ValueError("tolerance_pixels must be non-negative")
    structure = ndimage.generate_binary_structure(2, 2)
    predicted_neighbourhood = (
        ndimage.binary_dilation(
            predicted_boundary, structure=structure, iterations=tolerance_pixels
        )
        if tolerance_pixels
        else predicted_boundary
    )
    target_neighbourhood = (
        ndimage.binary_dilation(
            target_boundary, structure=structure, iterations=tolerance_pixels
        )
        if tolerance_pixels
        else target_boundary
    )
    return (
        int(np.logical_and(predicted_boundary, target_neighbourhood).sum()),
        int(predicted_boundary.sum()),
        int(np.logical_and(target_boundary, predicted_neighbourhood).sum()),
        int(target_boundary.sum()),
    )


def _small_object_detection_counts(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    maximum_area_pixels: int,
    minimum_coverage: float,
) -> tuple[int, int]:
    if maximum_area_pixels < 1:
        raise ValueError("maximum_area_pixels must be positive")
    if not 0 < minimum_coverage <= 1:
        raise ValueError("minimum_coverage must be in (0, 1]")
    components, component_count = ndimage.label(
        target, structure=ndimage.generate_binary_structure(2, 2)
    )
    detected = 0
    total = 0
    for component_id in range(1, component_count + 1):
        component = components == component_id
        area = int(component.sum())
        if area > maximum_area_pixels:
            continue
        total += 1
        coverage = float(np.logical_and(prediction, component).sum()) / area
        detected += int(coverage >= minimum_coverage)
    return detected, total


def compute_sample_statistics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    threshold: float,
    small_object_maximum_area_pixels: int,
    small_object_minimum_coverage: float = 0.50,
    boundary_tolerance_pixels: int = 2,
    score_histogram_bins: int = 1001,
) -> SufficientStatistics:
    """Convert one probability map and labelled mask to additive statistics.

    Targets use 0/1 labels and 255 for invalid pixels. Invalid pixels are
    excluded from every metric, including connected components and boundaries.
    """

    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets)
    if probabilities.shape != targets.shape or probabilities.ndim != 2:
        raise ValueError("probabilities and targets must be aligned 2-D arrays")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    if score_histogram_bins < 2:
        raise ValueError("score_histogram_bins must be at least two")
    valid = targets != 255
    if not np.any(valid):
        return SufficientStatistics(
            positive_score_histogram=(0,) * score_histogram_bins,
            negative_score_histogram=(0,) * score_histogram_bins,
        )
    if not np.isfinite(probabilities[valid]).all():
        raise ValueError("Valid probabilities must be finite")
    if np.any((probabilities[valid] < 0) | (probabilities[valid] > 1)):
        raise ValueError("Probabilities must be in [0, 1]")

    target = np.logical_and(targets == 1, valid)
    negative = np.logical_and(targets == 0, valid)
    prediction = np.logical_and(probabilities >= threshold, valid)
    positive_errors = (probabilities[target] - 1.0) ** 2
    negative_errors = probabilities[negative] ** 2
    indices = np.minimum(
        (probabilities[valid] * (score_histogram_bins - 1)).astype(np.int64),
        score_histogram_bins - 1,
    )
    valid_targets = targets[valid]
    positive_histogram = np.bincount(
        indices[valid_targets == 1], minlength=score_histogram_bins
    )
    negative_histogram = np.bincount(
        indices[valid_targets == 0], minlength=score_histogram_bins
    )
    small_detected, small_total = _small_object_detection_counts(
        prediction,
        target,
        maximum_area_pixels=small_object_maximum_area_pixels,
        minimum_coverage=small_object_minimum_coverage,
    )
    boundary_counts = _matched_boundary_counts(
        prediction, target, boundary_tolerance_pixels
    )
    return SufficientStatistics(
        true_positive=int(np.logical_and(prediction, target).sum()),
        false_positive=int(np.logical_and(prediction, negative).sum()),
        false_negative=int(np.logical_and(~prediction, target).sum()),
        true_negative=int(np.logical_and(~prediction, negative).sum()),
        small_detected=small_detected,
        small_total=small_total,
        boundary_matched_prediction=boundary_counts[0],
        boundary_prediction=boundary_counts[1],
        boundary_matched_target=boundary_counts[2],
        boundary_target=boundary_counts[3],
        positive_brier_sum=float(positive_errors.sum()),
        positive_count=int(target.sum()),
        negative_brier_sum=float(negative_errors.sum()),
        negative_count=int(negative.sum()),
        positive_score_histogram=tuple(int(value) for value in positive_histogram),
        negative_score_histogram=tuple(int(value) for value in negative_histogram),
    )
