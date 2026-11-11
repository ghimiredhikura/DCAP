"""Main compression grid: shared training, evaluation and analysis functions."""

from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import aggregate, bootstrap, paired_summary, replay
from .evaluation import METRICS, clean, load_assessment
from .paths import ROOT, read, sha256, study, write

METHODS = ("dcap", "taylor", "magnitude", "compact")


def jobs():
    from .training import job_for

    cfg = study()
    return [job_for("dense", s) for s in cfg["seeds"]] + [
        job_for(m, s, t) for m in METHODS for t in cfg["targets"] for s in cfg["seeds"]
    ]


def label(job):
    return (
        "dense"
        if job["method"] == "dense"
        else f"{job['method']}{round(job['target']*100)}"
    )


def campaign(phase="plan", output=None, directory=None):
    root = Path(output or ROOT / "generated/campaign").resolve()
    cfg = study()
    planned = jobs()
    if phase == "plan":
        return dict(
            jobs=len(planned),
            training=planned,
            output=str(root),
            phases=["train", "evaluate", "collect"],
        )
    if phase not in ("train", "evaluate", "collect"):
        raise ValueError("Unknown campaign phase: " + phase)
    root.mkdir(parents=True, exist_ok=True)
    if phase == "train":
        from .paths import data_root, sha256
        from .training import train_one

        design = read(ROOT / "configs/design.json")
        for job in planned:
            path = root / "training" / label(job) / f"seed{job['seed']}"
            if job["method"] in ("dcap", "taylor", "magnitude"):
                teacher = (
                    root / "training/dense" / f"seed{job['seed']}" / "checkpoint.pt"
                )
                job.update(
                    teacher_checkpoint=str(teacher), teacher_sha256=sha256(teacher)
                )
            done = path / "complete.json"
            if done.exists():
                config = read(path / "configuration.json")
                if config["job"] != job or config["design"] != design:
                    raise ValueError("Completed job configuration differs")
                completed_checkpoint(path)
                continue
            result = train_one(
                job,
                data_root(directory),
                output=path,
                design=design,
                resume=(path / "configuration.json").exists(),
            )
            if result["interrupted"]:
                return result
            result.update(
                checkpoint_sha256=sha256(path / "checkpoint.pt"),
                structure_sha256=sha256(path / "structure.json"),
            )
            write(done, result)
        return dict(completed=len(planned))
    # Evaluation and collection require the complete planned training outputs.
    for job in planned:
        completed_checkpoint(root / "training" / label(job) / f"seed{job['seed']}")
    if phase == "evaluate":
        from .evaluation import predict_ensemble

        for name in dict.fromkeys(label(j) for j in planned):
            checkpoints = [
                root / "training" / name / f"seed{s}/checkpoint.pt"
                for s in cfg["seeds"]
            ]
            validation = root / "evaluation" / f"{name}_validation.json"
            for split in ["validation", "test"]:
                target = root / "evaluation" / f"{name}_{split}.json"
                if target.exists():
                    from .paths import sha256

                    r, _, _ = load_assessment(target)
                    if r["checkpoint_hashes"] != [sha256(p) for p in checkpoints]:
                        raise ValueError("Evaluation weights changed")
                    if r["split"] != split or r["precision"] != "amp":
                        raise ValueError("Cached evaluation split/precision differs")
                    continue
                predict_ensemble(
                    name,
                    split,
                    directory,
                    device="cuda",
                    output=target,
                    checkpoints=checkpoints,
                    validation_result=validation if split == "test" else None,
                )
        return dict(evaluated_ensembles=21)
    return collect(root)


def collect(root):
    root = Path(root)
    out = root / "results"
    out.mkdir(parents=True, exist_ok=True)
    cfg = study()
    margins = {k: v[1] for k, v in cfg["joint_constraints"].items()}
    cache = {}

    def result(name, split):
        key = (name, split)
        if key not in cache:
            path = root / "evaluation" / f"{name}_{split}.json"
            record, ids, values = load_assessment(path)
            paths = [
                root / "training" / name / f"seed{seed}/checkpoint.pt"
                for seed in cfg["seeds"]
            ]
            if record["split"] != split or record["checkpoint_hashes"] != [
                sha256(p) for p in paths
            ]:
                raise ValueError(
                    "Evaluation does not match the completed training checkpoints"
                )
            blocks, matrix = aggregate(ids, values)
            point = np.array([record["outcomes"][m] for m in METRICS])
            cache[key] = (record, ids, values, blocks, point, bootstrap(matrix))
        return cache[key]

    def compare(name, split):
        a, ids, _, blocks, point, draws = result("dense", split)
        b, ids_b, _, blocks_b, candidate, draws_b = result(name, split)
        np.testing.assert_array_equal(ids, ids_b)
        np.testing.assert_array_equal(blocks, blocks_b)
        if a["precision"] != b["precision"]:
            raise ValueError("Mixed evaluation precision")
        return paired_summary(point, candidate, draws, draws_b, margins)

    def macs(name):
        return np.mean(
            [
                read(root / "training" / name / f"seed{s}/structure.json")[
                    "measured_macs"
                ]
                for s in cfg["seeds"]
            ]
        )

    dense_macs = macs("dense")
    rows = []
    comparisons = {}
    operating_points = []
    representative = {
        f"dense_{split}": {
            **result("dense", split)[0]["outcomes"],
            "mac_reduction_percent": 0.0,
            "joint_support": None,
        }
        for split in ["validation", "test"]
    }
    for method in METHODS:
        for target in cfg["targets"]:
            pct = round(target * 100)
            name = f"{method}{pct}"
            reduction = 100 * (1 - macs(name) / dense_macs)
            comparisons[name] = {
                split: compare(name, split) for split in ["validation", "test"]
            }
            if method != "compact":
                operating_points.append(
                    dict(
                        ensemble=name,
                        method=method,
                        budget=pct,
                        mac_reduction=reduction,
                        validation_ap=result(name, "validation")[0]["outcomes"][
                            "average_precision"
                        ],
                        joint_support=comparisons[name]["validation"]["joint_support"],
                    )
                )
            rows.append(
                dict(
                    method=method,
                    target_percent=pct,
                    mac_reduction_percent=reduction,
                    **result(name, "test")[0]["outcomes"],
                    joint_support=comparisons[name]["test"]["joint_support"],
                )
            )
    selections = select_operating_points(
        operating_points,
        result("dense", "validation")[0]["outcomes"]["average_precision"],
    )
    intervals = []
    for method in METHODS[:3]:
        name = f"{method}70"
        for split in ["validation", "test"]:
            p = compare(name, split)
            for metric in METRICS[1:]:
                r = p["metrics"][metric]
                intervals.append(
                    dict(
                        method=method,
                        split=split,
                        metric=metric,
                        delta=r["delta"],
                        ci_low=r["ci95"][0],
                        ci_high=r["ci95"][1],
                    )
                )
    selected = "dcap70"
    evidence = {}
    for split in ["validation", "test"]:
        for source, key in [("dense", "dense"), (selected, "dcap70")]:
            record, ids, values, *_ = result(source, split)
            evidence[f"{key}_{split}__sample_ids"] = ids
            evidence[f"{key}_{split}__sample_statistics"] = values
            representative[f"{key}_{split}"] = {
                **record["outcomes"],
                "mac_reduction_percent": 100 * (1 - macs(source) / dense_macs),
                "joint_support": (
                    compare(source, split)["joint_support"] if key != "dense" else None
                ),
            }
    np.savez_compressed(out / "spatial_evidence.npz", **evidence)
    replay(out, True, out / "spatial_evidence.npz", candidate="dcap70")
    pd.DataFrame(rows).to_csv(out / "figure6.csv", index=False)
    pd.DataFrame(intervals).to_csv(out / "figure7.csv", index=False)
    write(out / "representative.json", clean(representative))
    write(out / "selections.json", selections)
    pd.DataFrame(operating_points).to_csv(out / "selection.csv", index=False)
    write(out / "comparisons.json", clean(comparisons))
    return dict(results=str(out), selected=selections, selection_split="validation")


def select_operating_points(rows, dense_ap):
    """Apply the validation AP guard and joint rule; controls are not candidates."""
    cfg = study()
    eligible = [
        r
        for r in rows
        if r["method"] in METHODS[:3]
        and r["validation_ap"] >= dense_ap - cfg["validation_ap_tolerance"]
        and r["joint_support"] >= cfg["required_joint_support"]
    ]
    result = {}
    for method in METHODS[:3]:
        candidates = [r for r in eligible if r["method"] == method]
        result[method] = (
            max(candidates, key=lambda r: r["mac_reduction"])["ensemble"]
            if candidates
            else "dense"
        )
    result["overall_included_pruning_methods"] = (
        max(eligible, key=lambda r: r["mac_reduction"])["ensemble"]
        if eligible
        else "dense"
    )
    return result


def completed_checkpoint(directory):
    """Verify the selected weights and structure of one completed campaign fit."""
    directory = Path(directory)
    marker = directory / "complete.json"
    if not marker.is_file():
        raise ValueError(
            f"Incomplete training run: {directory}; finish campaign train first"
        )
    record = read(marker)
    if record.get("interrupted") or record.get("smoke_only"):
        raise ValueError(f"Training run is not a completed scientific fit: {directory}")
    for filename, key in [
        ("checkpoint.pt", "checkpoint_sha256"),
        ("structure.json", "structure_sha256"),
    ]:
        path = directory / filename
        if not path.is_file() or record.get(key) != sha256(path):
            raise ValueError(f"Missing or changed completed output: {path}")
    return directory / "checkpoint.pt"
