"""Fixed-threshold paired spatial inference shared by replay and new predictions."""

from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import METRICS, clean, vector_outcomes
from .paths import ROOT, study, write


def aggregate(ids, values, width=20, ox=0.0, oy=0.0):
    frame = pd.read_csv(ROOT / "metadata/spatial.csv").set_index("sample_id").loc[ids]
    x = np.array([round(float(v), 3) for v in frame.metric_x])
    y = np.array([round(float(v), 3) for v in frame.metric_y])
    ix = np.floor((x - ox * width * 1000) / (width * 1000)).astype(int)
    iy = np.floor((y - oy * width * 1000) / (width * 1000)).astype(int)
    labels = np.array([f"epsg32645_{width}km_x{a}_y{b}" for a, b in zip(ix, iy)])
    blocks, indices = np.unique(labels, return_inverse=True)
    matrix = np.zeros((len(blocks), values.shape[1]), dtype=np.float64)
    np.add.at(matrix, indices, values)
    return blocks, matrix


def replay(output=None, grid=False, evidence=None, candidate="dcap70"):
    cfg = study()
    margins = {k: v[1] for k, v in cfg["joint_constraints"].items()}
    path = Path(evidence or ROOT / "results/spatial_evidence.npz")
    result = {}
    grid_rows = []
    with np.load(path, allow_pickle=False) as data:
        for split in ["validation", "test"]:
            ids = data[f"dense_{split}__sample_ids"]
            np.testing.assert_array_equal(ids, data[f"{candidate}_{split}__sample_ids"])
            primary = {}
            for model in ["dense", candidate]:
                raw = data[f"{model}_{split}__sample_statistics"]
                blocks, matrix = aggregate(ids, raw)
                point = vector_outcomes(matrix.sum(0, keepdims=True))[0]
                primary[model] = (point, bootstrap(matrix))
                result[f"{model}_{split}"] = dict(
                    outcomes=dict(zip(METRICS, point)), blocks=len(blocks)
                )
            a, da = primary["dense"]
            b, db = primary[candidate]
            result[f"{candidate}_{split}"]["preservation"] = paired_summary(
                a, b, da, db, margins
            )
            if grid:
                for width in [10, 20, 30]:
                    for ox, oy in [(0, 0), (0.5, 0), (0, 0.5), (0.5, 0.5)]:
                        vals = []
                        for model in ["dense", candidate]:
                            blocks, matrix = aggregate(
                                ids,
                                data[f"{model}_{split}__sample_statistics"],
                                width,
                                ox,
                                oy,
                            )
                            vals.append(bootstrap(matrix))
                        for scale in [0.5, 1, 1.5]:
                            p = paired_summary(
                                a, b, *vals, {k: v * scale for k, v in margins.items()}
                            )
                            grid_rows.append(
                                dict(
                                    split=(
                                        "retrospective_test"
                                        if split == "test"
                                        else split
                                    ),
                                    width_km=width,
                                    origin_x_fraction=ox,
                                    origin_y_fraction=oy,
                                    blocks=len(blocks),
                                    margin_scale=scale,
                                    joint_support=p["joint_support"],
                                    joint_invalid_draws=p["joint_invalid_draws"],
                                )
                            )
    result = clean(result)
    output = Path(output or ROOT / "generated/replay")
    output.mkdir(parents=True, exist_ok=True)
    write(output / "representative_replay.json", result)
    if grid:
        pd.DataFrame(grid_rows).to_csv(output / "figure8.csv", index=False)
    return result


def bootstrap(matrix):
    cfg = study()
    replicates = cfg["bootstrap_replicates"]
    rng = np.random.default_rng(cfg["bootstrap_seed"])
    n = len(matrix)
    if n == 0:
        raise ValueError("Bootstrap requires at least one spatial block")
    result = []
    for start in range(0, replicates, 250):
        counts = rng.multinomial(
            n, np.full(n, 1 / n), size=min(250, replicates - start)
        )
        result.append(vector_outcomes(counts @ matrix))
    return np.concatenate(result)


def paired_summary(a, b, da, db, margins):
    point = b - a
    delta = db - da
    valid_joint = np.ones(len(delta), dtype=bool)
    pass_joint = np.ones(len(delta), dtype=bool)
    results = {}
    for i, metric in enumerate(METRICS):
        values = delta[:, i]
        valid = np.isfinite(values)
        ci = (
            np.quantile(values[valid], [0.025, 0.975], method="linear").tolist()
            if valid.any()
            else [None, None]
        )
        if np.isfinite(point[i]) and valid.any():
            tail = (np.abs(values[valid] - point[i]) >= abs(point[i])).sum() + (
                ~valid
            ).sum()
            p = (1 + tail) / (1 + len(values))
        else:
            p = 1.0
        results[metric] = dict(
            delta=point[i], ci95=ci, invalid_draws=int((~valid).sum()), p_centered=p
        )
        if metric in margins:
            passed = (
                values <= margins[metric]
                if metric == "balanced_brier_score"
                else values >= margins[metric]
            )
            pass_joint &= valid & passed
            valid_joint &= valid
            results[metric]["preservation_support"] = float((valid & passed).mean())
    return dict(
        metrics=results,
        joint_support=float(pass_joint.mean()),
        joint_defined_draws=int(valid_joint.sum()),
        joint_invalid_draws=int((~valid_joint).sum()),
        joint_support_conditional_on_defined=(
            float(pass_joint.sum() / valid_joint.sum()) if valid_joint.any() else None
        ),
    )
