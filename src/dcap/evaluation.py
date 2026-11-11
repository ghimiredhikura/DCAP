"""Five manuscript endpoints, validation thresholds and paired spatial inference."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .paths import ROOT, read, sha256, write
from .pixel_statistics import compute_sample_statistics
from .statistics import SufficientStatistics, compute_outcomes, sum_statistics
from .thresholds import fit_leave_one_block_out_thresholds


def predict_ensemble(
    name,
    split,
    directory=None,
    models=None,
    device="cuda",
    precision="amp",
    output=None,
    checkpoints=None,
    validation_result=None,
):
    from .data import Dataset, load_cache, loader
    from .models import load_checkpoint

    if split not in ("validation", "test"):
        raise ValueError("Evaluation split must be validation or test")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; choose --device cpu --precision fp32")
    if device == "cpu" and precision != "fp32":
        raise ValueError("CPU evaluation uses --precision fp32")
    from .models import load_ensemble, probability_mean

    if checkpoints:
        if len(checkpoints) != 3 or len(set(map(str, checkpoints))) != 3:
            raise ValueError("Provide three distinct ensemble checkpoints")
        loaded = [load_checkpoint(path, device) for path in checkpoints]
        members = [m for m, payload in loaded]
        records = [
            dict(
                configuration=payload["configuration"],
                sha256=sha256(path),
                deterministic_algorithms=payload.get("deterministic_algorithms", True),
            )
            for path, (m, payload) in zip(checkpoints, loaded)
        ]
        if any(r["configuration"] != records[0]["configuration"] for r in records):
            raise ValueError("Ensemble input/model configurations differ")
        if split == "test":
            if not validation_result:
                raise ValueError(
                    "Fresh test inference requires --validation-result from the same checkpoints"
                )
            fitted = read(validation_result)
            if (
                fitted["split"] != "validation"
                or fitted["checkpoint_hashes"] != [r["sha256"] for r in records]
                or fitted["precision"] != precision
            ):
                raise ValueError(
                    "Validation thresholds do not belong to these checkpoints/precision"
                )
    else:
        if validation_result:
            raise ValueError("--validation-result is for --checkpoints evaluation")
        members, records = load_ensemble(name, models, device)
    checkpoint_hashes = [r["sha256"] for r in records]
    cache, frame, norm = load_cache(directory, (split,))
    rows = frame[frame.split == split].to_dict("records")
    dataset = Dataset(cache, rows, norm, records[0]["configuration"]["input_mode"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    probabilities = []
    targets = []
    sample_ids = []
    with torch.inference_mode():
        for images, masks, ids in loader(dataset):
            pred = probability_mean(
                members,
                images.to(device),
                precision,
                [m.get("deterministic_algorithms", False) for m in records],
            )
            probabilities.append(pred[:, 0].cpu().numpy())
            targets.append(masks.numpy().astype(np.uint8))
            sample_ids.extend(ids)
    p = np.concatenate(probabilities)
    t = np.concatenate(targets)
    if checkpoints:
        cutoff = fitted["global_validation_threshold"] if split == "test" else None
        frozen_cuts = None
    else:
        thresholds = read(ROOT / "metadata/thresholds.json")[name]
        cutoff = thresholds["global_validation_threshold"]
        frozen_cuts = (
            thresholds["cross_fitted_thresholds"] if split == "validation" else None
        )
    assessment, records = assess(p, t, sample_ids, cutoff, frozen_cuts=frozen_cuts)
    assessment.update(
        ensemble=name,
        checkpoint_hashes=checkpoint_hashes,
        released_thresholds=not bool(checkpoints),
        split=split,
        device=device,
        precision=precision,
        test_threshold_fitting=False,
        checkpoint_selection_on_test=False,
    )
    output = Path(output or ROOT / f"generated/evaluation/{name}_{split}.json")
    raw = np.array(
        [
            [s[k] for k in SCALARS]
            + list(s["positive_score_histogram"])
            + list(s["negative_score_histogram"])
            for s in records["statistics"]
        ],
        dtype=np.float64,
    )
    return save_assessment(output, assessment, sample_ids, raw)


# Column order shared by saved statistics and bootstrap calculations.
METRICS = (
    "average_precision",
    "dice",
    "small_object_recall",
    "boundary_f1",
    "balanced_brier_score",
)
SCALARS = [
    n for n in SufficientStatistics.__dataclass_fields__ if not n.endswith("_histogram")
]
INDEX = {n: i for i, n in enumerate(SCALARS)}


def clean(v):
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, (float, np.floating)):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, np.integer):
        return int(v)
    return v


def assess(p, t, ids, cutoff=None, small_object_pixels=41, frozen_cuts=None):
    metadata = (
        pd.read_csv(ROOT / "metadata/spatial.csv")
        .set_index("sample_id")
        .to_dict("index")
    )
    mapping = {sid: metadata[sid]["block_20km"] for sid in ids}
    if frozen_cuts is not None:
        thresholds = frozen_cuts
    elif cutoff is None:
        cutoff, thresholds = fit_leave_one_block_out_thresholds(p, t, ids, mapping)
    else:
        thresholds = {b: cutoff for b in set(mapping.values())}
    records = [
        compute_sample_statistics(
            pp,
            tt,
            threshold=thresholds[mapping[sid]],
            small_object_maximum_area_pixels=small_object_pixels,
        )
        for pp, tt, sid in zip(p, t, ids, strict=True)
    ]
    result = dict(
        chips=len(ids),
        global_validation_threshold=float(cutoff),
        thresholds=thresholds,
        outcomes=clean(compute_outcomes(sum_statistics(records))),
    )
    return result, dict(sample_ids=ids, statistics=[asdict(r) for r in records])


def save_assessment(path, assessment, sample_ids, statistics):
    """Write the statistics first and bind them to the JSON record by checksum."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    archive = path.with_suffix(".npz")
    temporary = archive.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, sample_ids=sample_ids, sample_statistics=statistics)
    temporary.replace(archive)
    result = clean({**assessment, "statistics_sha256": sha256(archive)})
    write(path, result)
    return result


def load_assessment(path):
    """Read a complete endpoint/statistics pair, rejecting stale or altered files."""
    path = Path(path)
    result = read(path)
    archive = path.with_suffix(".npz")
    if not archive.is_file() or result.get("statistics_sha256") != sha256(archive):
        raise ValueError(
            f"Incomplete or changed evaluation statistics: {path}; rerun evaluation"
        )
    with np.load(archive, allow_pickle=False) as data:
        ids = data["sample_ids"].copy()
        statistics = data["sample_statistics"].copy()
    if statistics.shape != (len(ids), len(SCALARS) + 2002) or len(set(ids)) != len(ids):
        raise ValueError(f"Invalid sample statistics: {archive}")
    return result, ids, statistics


def vector_outcomes(a):

    def col(n):
        return a[:, INDEX[n]]

    ratio = lambda x, y: np.divide(
        x, y, out=np.full_like(x, np.nan, dtype=float), where=y > 0
    )
    precision = ratio(col("boundary_matched_prediction"), col("boundary_prediction"))
    recall = ratio(col("boundary_matched_target"), col("boundary_target"))
    pos = a[:, len(SCALARS) : len(SCALARS) + 1001]
    neg = a[:, len(SCALARS) + 1001 :]
    tp = pos[:, ::-1].cumsum(1)
    fp = neg[:, ::-1].cumsum(1)
    return np.stack(
        (
            ratio((tp / np.maximum(tp + fp, 1) * pos[:, ::-1]).sum(1), pos.sum(1)),
            ratio(
                2 * col("true_positive"),
                2 * col("true_positive")
                + col("false_positive")
                + col("false_negative"),
            ),
            ratio(col("small_detected"), col("small_total")),
            np.where(
                col("boundary_target") == 0,
                np.nan,
                np.where(
                    (col("boundary_prediction") == 0) | (precision + recall == 0),
                    0.0,
                    ratio(2 * precision * recall, precision + recall),
                ),
            ),
            0.5
            * (
                ratio(col("positive_brier_sum"), col("positive_count"))
                + ratio(col("negative_brier_sum"), col("negative_count"))
            ),
        ),
        axis=1,
    )
