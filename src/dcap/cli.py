"""Train, evaluate and reproduce the DCAP segmentation study."""

import argparse
import json
import os
from pathlib import Path

from .paths import ROOT, read, sha256, study


def positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def build_parser():
    parser = argparse.ArgumentParser(prog="dcap", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "verify",
        help="Check source-file hashes and split separation",
        description="Check source-file hashes and split separation",
    )
    command = commands.add_parser(
        "campaign",
        help="Plan, train, evaluate or collect the main compression grid",
        description="Plan, train, evaluate or collect the main compression grid",
    )
    command.add_argument(
        "phase",
        choices=["plan", "train", "evaluate", "collect"],
        default="plan",
        nargs="?",
    )
    command.add_argument(
        "--output",
        help="Output file or directory; defaults are listed in the reproduction guide",
    )
    command.add_argument(
        "--data-root", help="Extracted rasters (default: DCAP_DATA or checkout/data)"
    )
    command = commands.add_parser(
        "models",
        help="List or install the six released checkpoints",
        description="List or install the six released checkpoints",
    )
    command.add_argument("action", choices=["list", "install"])
    command.add_argument(
        "--source", help="Folder containing the input assets or result tables"
    )
    command.add_argument(
        "--models-root",
        help="Checkpoint directory (default: DCAP_MODELS or checkout/models)",
    )
    command = commands.add_parser(
        "data",
        help="Install, verify or rebuild the study rasters",
        description="Install, verify or rebuild the study rasters",
    )
    command.add_argument("action", choices=["install", "verify", "prepare"])
    command.add_argument(
        "--source", help="Folder containing the input assets or result tables"
    )
    command.add_argument(
        "--data-root", help="Extracted rasters (default: DCAP_DATA or checkout/data)"
    )
    command.add_argument(
        "--limit", type=positive_integer, help="Prepare only the first N chips"
    )
    command.add_argument("--mask-only", action="store_true")
    command.add_argument("--imagery-root")
    command = commands.add_parser(
        "plot",
        help="Draw Figures 6–8 and 10 from result tables",
        description="Draw Figures 6–8 and 10 from result tables",
    )
    command.add_argument(
        "--figures", type=int, nargs="+", default=[6, 7, 8], choices=[6, 7, 8, 10]
    )
    command.add_argument(
        "--source", help="Folder containing the input assets or result tables"
    )
    command.add_argument(
        "--output",
        help="Output file or directory; defaults are listed in the reproduction guide",
    )
    command = commands.add_parser(
        "replay",
        help="Recalculate the supplied dense/DCAP spatial evidence",
        description="Recalculate the supplied dense/DCAP spatial evidence",
    )
    command.add_argument(
        "--grid",
        action="store_true",
        help="Also recalculate all grid/margin sensitivity cells",
    )
    command.add_argument(
        "--output",
        help="Output file or directory; defaults are listed in the reproduction guide",
    )
    command.add_argument("--evidence")
    command.add_argument(
        "--candidate", default="dcap70", help="Candidate key in the evidence NPZ"
    )
    command = commands.add_parser(
        "evaluate",
        help="Evaluate a three-model ensemble",
        description="Evaluate a three-model ensemble",
    )
    command.add_argument("--ensemble", default="dcap70")
    command.add_argument("--split", required=True, choices=["validation", "test"])
    command.add_argument(
        "--data-root", help="Extracted rasters (default: DCAP_DATA or checkout/data)"
    )
    command.add_argument(
        "--models-root",
        help="Checkpoint directory (default: DCAP_MODELS or checkout/models)",
    )
    command.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    command.add_argument("--precision", choices=["fp32", "amp"], default="amp")
    command.add_argument(
        "--output",
        help="Output file or directory; defaults are listed in the reproduction guide",
    )
    command.add_argument(
        "--checkpoints",
        nargs=3,
        metavar="PATH",
        help="Custom checkpoint paths in seed order 42, 123, 2026",
    )
    command.add_argument(
        "--validation-result",
        help="Validation JSON from these custom checkpoints; required for their test evaluation",
    )
    command = commands.add_parser(
        "train",
        help="Train one model with a fixed fitting/validation design",
        description="Train one model with a fixed fitting/validation design",
    )
    command.add_argument(
        "--method",
        choices=["dense", "dcap", "taylor", "magnitude", "compact", "dense_recovery"],
        default="dcap",
    )
    command.add_argument("--seed", type=int, default=42, choices=[42, 123, 2026])
    command.add_argument(
        "--target", type=float, default=0.7, choices=[0.25, 0.4, 0.55, 0.7, 0.85]
    )
    command.add_argument("--recovery", choices=["full", "supervised"], default="full")
    command.add_argument("--omit-loss", choices=list(study()["recovery_loss"]))
    command.add_argument(
        "--gate-objective", choices=["full", "supervised"], default="full"
    )
    command.add_argument(
        "--gate-epochs",
        type=positive_integer,
        help="Override the default 18 gate-learning epochs",
    )
    command.add_argument(
        "--cell",
        choices=[c["cell"] for c in read(ROOT / "configs/factorial.json")["cells"]],
        help="Factorial configuration; requires --method dense",
    )
    command.add_argument("--teacher", help="Dense parent checkpoint for this seed")
    command.add_argument(
        "--initial", help="Existing initial pruned checkpoint for a recovery-only run"
    )
    command.add_argument("--design", default=str(ROOT / "configs/design.json"))
    command.add_argument(
        "--data-root", help="Extracted rasters (default: DCAP_DATA or checkout/data)"
    )
    command.add_argument(
        "--models-root",
        help="Checkpoint directory (default: DCAP_MODELS or checkout/models)",
    )
    command.add_argument(
        "--output",
        help="Output file or directory; defaults are listed in the reproduction guide",
    )
    command.add_argument(
        "--plan", action="store_true", help="Print the job settings without training"
    )
    command.add_argument(
        "--smoke",
        action="store_true",
        help="Short functional check, not a study result",
    )
    command.add_argument(
        "--resume",
        action="store_true",
        help="Continue this output directory with the same job settings",
    )
    command.add_argument(
        "--epochs",
        type=positive_integer,
        help="Override the training cap; study defaults are used when omitted",
    )
    command = commands.add_parser(
        "benchmark",
        help="Run the 12 Ubuntu GPU timing configurations",
        description="Run the 12 Ubuntu GPU timing configurations",
    )
    command.add_argument("--output", required=True)
    command.add_argument(
        "--models-root",
        help="Checkpoint directory (default: DCAP_MODELS or checkout/models)",
    )
    command.add_argument(
        "--smoke",
        action="store_true",
        help="Short functional check, not a study result",
    )
    command.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    command.add_argument("--ensemble", default="dcap70", help=argparse.SUPPRESS)
    command.add_argument(
        "--mode",
        default="fp32_graph",
        choices=["fp32_eager", "fp32_graph", "amp"],
        help=argparse.SUPPRESS,
    )
    command.add_argument(
        "--batch", type=int, default=1, choices=[1, 4], help=argparse.SUPPRESS
    )
    command = commands.add_parser(
        "download",
        help="Download separate assets from a published GitHub release",
        description="Download separate assets from a published GitHub release",
    )
    command.add_argument("--repository", required=True)
    command.add_argument("--tag", default="v1.1.0")
    command.add_argument("--kind", choices=["data", "models"], required=True)
    command.add_argument("--output", required=True)
    return parser


def main():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "verify":
        manifest = read(ROOT / "metadata/files.json")
        for name, digest in manifest.items():
            if not (ROOT / name).is_file() or sha256(ROOT / name) != digest:
                raise ValueError("File checksum mismatch: " + name)
        design = read(ROOT / "configs/design.json")
        sets = [set(design[k]) for k in ["fit_ids", "validation_ids", "outer_ids"]]
        if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
            raise ValueError("Overlapping study partitions")
        result = dict(
            passed=True, files=len(manifest), partition_sizes=[len(s) for s in sets]
        )
    elif args.command == "campaign":
        import torch

        torch.set_num_threads(4)
        from .experiment import campaign

        result = campaign(args.phase, args.output, args.data_root)
    elif args.command == "models":
        from .models import catalog, install_assets

        if args.action == "install" and not args.source:
            parser.error("models install requires --source")
        result = (
            catalog()["ensembles"]
            if args.action == "list"
            else install_assets(args.source, args.models_root)
        )
    elif args.command == "data":
        from . import data

        if args.action == "install":
            if not args.source:
                parser.error("data install requires --source")
            result = data.install(args.source, args.data_root)
        elif args.action == "verify":
            result = data.verify(args.data_root)
        else:
            from .preprocess import prepare

            result = prepare(
                args.data_root, args.limit, args.mask_only, args.imagery_root
            )
    elif args.command == "plot":
        from .figures import plot

        result = plot(args.figures, args.source, args.output)
    elif args.command == "replay":
        from .analysis import replay

        result = replay(args.output, args.grid, args.evidence, args.candidate)
    elif args.command == "evaluate":
        import torch

        torch.set_num_threads(4)
        from .evaluation import predict_ensemble

        result = predict_ensemble(
            args.ensemble,
            args.split,
            args.data_root,
            args.models_root,
            args.device,
            args.precision,
            args.output,
            args.checkpoints,
            args.validation_result,
        )
    elif args.command == "train":
        import torch

        torch.set_num_threads(4)
        from .paths import data_root
        from .training import job_for, train_one

        if args.cell and args.method != "dense":
            parser.error("--cell is args dense factorial training configuration")
        job = job_for(args.method, args.seed, args.target, args.cell)
        job["gate_objective"] = args.gate_objective
        if args.gate_epochs is not None:
            job["gate_epochs"] = args.gate_epochs
        if args.recovery == "supervised":
            job["recovery_weights"] = {k: 0.0 for k in job["recovery_weights"]}
        if args.omit_loss:
            job["recovery_weights"][args.omit_loss] = 0.0
        if args.teacher:
            job.update(
                teacher_checkpoint=str(Path(args.teacher).resolve()),
                teacher_sha256=sha256(args.teacher),
            )
        if args.initial:
            job.update(
                initial_checkpoint=str(Path(args.initial).resolve()),
                initial_sha256=sha256(args.initial),
            )
            if not args.teacher and any(job["recovery_weights"].values()):
                parser.error("--initial with full recovery also requires --teacher")
        design = read(args.design)
        result = (
            dict(job=job, design=args.design)
            if args.plan
            else train_one(
                job,
                directory=data_root(args.data_root),
                models=args.models_root,
                output=args.output,
                epochs=args.epochs,
                smoke=args.smoke,
                resume=args.resume,
                design=design,
            )
        )
    elif args.command == "benchmark":
        from .benchmark import run, worker

        result = (
            worker(
                args.ensemble,
                args.mode,
                args.batch,
                args.output,
                args.models_root,
                args.smoke,
            )
            if args.worker
            else run(args.output, args.models_root, args.smoke)
        )
    else:
        from .assets import download

        result = download(args.repository, args.tag, args.kind, args.output)
    print(json.dumps(result, indent=2, allow_nan=False))
