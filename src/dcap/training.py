"""Dense training and pruned-model recovery with dual early stopping."""

import contextlib
import copy
import json
import os
import random
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import psutil
import torch
from torch import nn

from .architectures import optimizer_parameter_groups
from .data import Dataset, input_indices, loader
from .models import build_model, ensemble_paths, load_checkpoint, save_checkpoint
from .paths import ROOT, read, sha256, study, write
from .training_helpers import (
    Metrics,
    ModelEMA,
    compute_loss,
    learning_rate_factor,
    recalibrate_training_bn,
    seed_everything,
)


def loss_args(name="bce_dice"):
    return SimpleNamespace(
        loss_name=name,
        overlap_weight=0.5,
        focal_weight=0.4,
        tversky_weight=0.6,
        focal_alpha=0.75,
        focal_gamma=2.0,
        tversky_alpha=0.3,
        tversky_beta=0.7,
        loss_smooth=1.0,
        gradient_clip=1.0,
    )


def objective(logits, target, job):
    return compute_loss(
        logits.float(),
        target,
        job.get("loss", "bce_dice"),
        20.0,
        loss_args(job.get("loss", "bce_dice")),
    )


def epoch_pass(
    model, batches, job, optimizer=None, scaler=None, ema=None, teacher=None
):
    training = optimizer is not None
    model.train(training)
    metrics = Metrics()
    loss_sum = total_sum = 0.0
    count = 0
    components = {}
    for image, target, _ in batches:
        image, target = image.cuda(non_blocking=True), target.cuda(non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with (
            torch.set_grad_enabled(training),
            torch.autocast("cuda", dtype=torch.float16),
        ):
            logits = model(image)
            task = objective(logits, target, job)
            loss = task
            if teacher is not None and training:
                from .algorithm import preservation_loss

                with torch.no_grad():
                    reference = teacher(image)
                loss, terms = preservation_loss(
                    logits,
                    reference,
                    target,
                    settings=job["recovery_weights"],
                    small_object_pixels=job["small_object_pixels"],
                )
                for key, value in terms.items():
                    components[key] = components.get(key, 0.0) + float(
                        value.detach()
                    ) * len(image)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training objective")
        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if job.get("optimization", "enhanced") == "enhanced":
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)
        metrics.update(logits.detach(), target)
        loss_sum += float(task.detach()) * len(image)
        total_sum += float(loss.detach()) * len(image)
        count += len(image)
    return dict(
        loss=loss_sum / count,
        total_objective=total_sum / count,
        **metrics.compute(),
        components={k: v / count for k, v in components.items()},
    )


def parent_for_seed(seed, directory=None):
    paths, entry = ensemble_paths("dense", directory)
    return load_checkpoint(paths[[42, 123, 2026].index(seed)], "cuda")[0]


def construct_pruned(
    method, target, seed, cache, train, norm, models=None, job=None, design=None
):
    from dataclasses import asdict

    from .algorithm import physical_prune, search_shared, select_gate_inheritance
    from .structural import prune_to_measured_mac_target

    preparation_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    phase_times = {}
    teacher = (
        load_checkpoint(job["teacher_checkpoint"], "cuda")[0]
        if job and job.get("teacher_checkpoint")
        else parent_for_seed(seed, models)
    )
    teacher.eval()
    model = copy.deepcopy(teacher)
    if design is None:
        raise ValueError("Pruning requires fitting-only calibration IDs in the design")
    ids = design
    by_id = {r["sample_id"]: r for r in train}
    if method == "dcap":
        fit_ids = ids["gate_ids"]
        if job.get("calibration_size", 128) != 128:
            fit_ids = (
                np.random.default_rng(2026)
                .permutation(sorted(by_id))[: job["calibration_size"]]
                .tolist()
            )
        batches = loader(Dataset(cache, [by_id[s] for s in fit_ids], norm))
        phase_started = time.perf_counter()
        scores, densities, audit = search_shared(
            model,
            teacher,
            batches,
            target,
            job.get("gate_epochs", study()["gate_epochs"]),
            objective=job.get("gate_objective", "full"),
            small_object_pixels=job["small_object_pixels"],
        )
        torch.cuda.synchronize()
        phase_times["gate_seconds"] = time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        model, measurement, trace = physical_prune(
            model,
            scores,
            densities,
            target,
            minimum_channels=job.get("minimum_channels", 8),
        )
        torch.cuda.synchronize()
        phase_times["physical_search_seconds"] = time.perf_counter() - phase_started
        remaining = sorted(set(by_id) - set(fit_ids))
        rng = np.random.default_rng(seed + 7013)
        selected = rng.choice(remaining, 128, replace=False).tolist()
        score_batches = loader(Dataset(cache, [by_id[s] for s in selected], norm))
        bn_batches = loader(Dataset(cache, train, norm))
        phase_started = time.perf_counter()
        model, inheritance = select_gate_inheritance(
            teacher,
            model,
            teacher,
            scores,
            measurement["physical_pruning_history"],
            bn_batches,
            score_batches,
            small_object_pixels=job["small_object_pixels"],
        )
        torch.cuda.synchronize()
        phase_times["bn_and_inheritance_seconds"] = time.perf_counter() - phase_started
        measurement.update(
            gate_inheritance=inheritance["selected"],
            gate_search=audit,
            inheritance=inheritance,
            gate_ids=fit_ids,
            inheritance_ids=selected,
            physical_trace=trace,
        )
    else:
        callback = None
        if method == "taylor":
            batches = loader(
                Dataset(cache, [by_id[s] for s in ids["taylor_ids"]], norm)
            )

            def callback(step, current):
                current.eval()
                current.zero_grad(set_to_none=True)
                for images, masks, _ in batches:
                    images, masks = images.cuda(), masks.cuda()
                    with torch.autocast("cuda", dtype=torch.float16):
                        loss = objective(current(images), masks, {})
                    (loss / len(batches)).backward()

        model, measurement, trace = prune_to_measured_mac_target(
            model,
            torch.zeros(1, 21, 256, 256, device="cuda"),
            method="group_taylor" if method == "taylor" else "group_magnitude_l2",
            target_mac_reduction=target,
            iterative_steps=100,
            maximum_channel_pruning_ratio=0.90,
            achieved_mac_tolerance=0.01,
            round_to=1,
            random_seed=seed,
            before_step=callback,
        )
        measurement = asdict(measurement)
        measurement["physical_trace"] = trace
        if not measurement["within_tolerance"]:
            raise ValueError(f"Measured pruning target infeasible: {measurement}")
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    measurement.update(
        preparation_seconds=time.perf_counter() - preparation_started,
        preparation_peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        phase_times=phase_times,
    )
    return (
        model,
        teacher if any(job["recovery_weights"].values()) else None,
        measurement,
    )


def train_one(
    job,
    directory=None,
    models=None,
    output=None,
    epochs=None,
    smoke=False,
    resume=False,
    design=None,
    stop_after_epoch=None,
):
    """Train using training/validation only; test access is a separate command."""
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA; use --plan to inspect settings")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    seed_everything(job["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if design is None:
        raise ValueError("Training requires a frozen fitting/validation design")
    if epochs is not None and epochs < 1:
        raise ValueError("Epoch count must be positive")
    if (
        smoke
        and job["method"] in ("dcap", "taylor")
        and not job.get("initial_checkpoint")
    ):
        raise ValueError(
            "DCAP/Taylor calibration requires the full fitting set; use --plan or an existing --initial checkpoint"
        )
    from concurrent.futures import ThreadPoolExecutor

    from .data import read_chip, split_rows

    frame = split_rows().set_index("sample_id", drop=False)
    train = frame.loc[design["fit_ids"]].to_dict("records")
    validation = frame.loc[design["validation_ids"]].to_dict("records")
    assert not set(design["fit_ids"]) & set(design["validation_ids"])
    assert not (set(design["fit_ids"]) | set(design["validation_ids"])) & set(
        design["outer_ids"]
    )
    norm = design["normalization"]
    if smoke:
        train, validation = train[:8], validation[:4]
    with ThreadPoolExecutor(max_workers=4) as pool:
        cache = dict(
            pool.map(lambda row: read_chip(Path(directory), row), train + validation)
        )
    full_train = train
    name = job["name"]
    out = Path(output or ROOT / "generated/training" / name)
    if out.exists() and any(out.iterdir()) and not resume:
        raise FileExistsError(f"Output exists; use --resume: {out}")
    out.mkdir(parents=True, exist_ok=True)
    # Record the exact dense parent in resumable run provenance.
    job = dict(job)
    if (
        job["method"] in ("dcap", "taylor", "magnitude", "dense_recovery")
        and not job.get("teacher_checkpoint")
        and not job.get("initial_checkpoint")
    ):
        from .models import catalog

        paths, _ = ensemble_paths("dense", models)
        parent = paths[study()["seeds"].index(job["seed"])]
        expected = next(
            r["sha256"] for r in catalog()["models"] if r["asset"] == parent.name
        )
        actual = sha256(parent)
        if actual != expected:
            raise ValueError("Dense parent checksum mismatch")
        job.update(teacher_checkpoint=str(parent.resolve()), teacher_sha256=actual)
    if job.get("teacher_checkpoint"):
        actual = sha256(job["teacher_checkpoint"])
        if job.get("teacher_sha256", actual) != actual:
            raise ValueError("Teacher checkpoint changed")
        job["teacher_sha256"] = actual
    configuration_record = dict(
        job=job, smoke_only=smoke, test_access=False, design=design, epochs=epochs
    )
    if resume and read(out / "configuration.json") != configuration_record:
        raise ValueError("Immutable job configuration differs")
    if not resume:
        write(out / "configuration.json", configuration_record)
    configuration = dict(
        architecture=job.get("architecture", "resunet"),
        input_mode=job.get("input_mode", "reflectance_change"),
        input_channels=len(input_indices(job.get("input_mode", "reflectance_change"))),
    )
    method = job["method"]
    teacher = None
    measurement = {}
    if job.get("initial_checkpoint"):
        if sha256(job["initial_checkpoint"]) != job["initial_sha256"]:
            raise ValueError("Initial checkpoint changed")
        model, initial_payload = load_checkpoint(job["initial_checkpoint"], "cuda")
        if initial_payload["configuration"] != configuration:
            raise ValueError(
                "Initial checkpoint does not match the requested architecture/input configuration"
            )
        if any(job["recovery_weights"].values()):
            teacher = (
                load_checkpoint(job["teacher_checkpoint"], "cuda")[0]
                if job.get("teacher_checkpoint")
                else parent_for_seed(job["seed"], models)
            )
        measurement = dict(
            retained_initial=job["initial_checkpoint"],
            initial_sha256=job["initial_sha256"],
        )
    elif method in ("dcap", "taylor", "magnitude"):
        if (
            resume
            and (out / "initial.pt").exists()
            and (out / "structure.json").exists()
        ):
            model, _ = load_checkpoint(out / "initial.pt", "cuda")
            measurement = read(out / "structure.json")
            if any(job["recovery_weights"].values()):
                teacher = (
                    load_checkpoint(job["teacher_checkpoint"], "cuda")[0]
                    if job.get("teacher_checkpoint")
                    else parent_for_seed(job["seed"], models)
                )
        else:
            model, teacher, measurement = construct_pruned(
                method,
                job["target"],
                job["seed"],
                cache,
                full_train,
                norm,
                models,
                job,
                design,
            )
            save_checkpoint(model, out / "initial.pt", configuration)
    elif method == "dense_recovery":
        teacher = (
            load_checkpoint(job["teacher_checkpoint"], "cuda")[0]
            if job.get("teacher_checkpoint")
            else parent_for_seed(job["seed"], models)
        )
        model = copy.deepcopy(teacher)
    else:
        if method == "compact":
            base, bridge = study()["compact_widths"][str(round(job["target"] * 100))]
            configuration.update(
                architecture="compact_resunet",
                base_channels=base,
                bridge_channels=bridge,
            )
        model = build_model(
            configuration,
            pretrained=configuration["architecture"] == "resnet34_deeplabv3plus",
        ).cuda()
    if teacher is not None:
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    # Start recovery from the same RNG state regardless of pruning preparation.
    seed_everything(job["seed"])
    import torch_pruning as tp

    with torch.no_grad():
        macs, params = tp.utils.count_ops_and_params(
            model.eval(),
            torch.zeros(1, configuration["input_channels"], 256, 256, device="cuda"),
        )
    measurement.update(measured_macs=int(macs), measured_parameters=int(params))
    write(out / "structure.json", measurement)
    dense = method in ("dense", "compact")
    settings = study()["dense" if dense else "recovery"]
    maximum = epochs or settings["max_epochs"]
    minimum = settings["min_epochs"]
    patience = settings["dual_patience"]
    if smoke:
        maximum, minimum, patience = 2, 2, 1
    enhanced = job.get("optimization", "enhanced") == "enhanced"
    groups = optimizer_parameter_groups(
        model, settings["encoder_lr"], settings["decoder_lr"]
    )
    optimizer = torch.optim.AdamW(groups, weight_decay=settings["weight_decay"])
    lrs = [g["lr"] for g in groups]
    scaler = torch.amp.GradScaler("cuda")
    ema = ModelEMA(model, settings["ema_decay"]) if enhanced else None
    train_ds = Dataset(
        cache,
        train,
        norm,
        configuration["input_mode"],
        job.get("optimization", "enhanced"),
        True,
        job["seed"],
    )
    train_batches = loader(train_ds, True, job["seed"])
    val_batches = loader(Dataset(cache, validation, norm, configuration["input_mode"]))
    bn_batches = (
        loader(Dataset(cache, train, norm, configuration["input_mode"]))
        if configuration["architecture"] != "resnet34_deeplabv3plus"
        else None
    )
    state = dict(
        history=[],
        best=-float("inf"),
        ap_anchor=-float("inf"),
        loss_anchor=float("inf"),
        ap_epoch=0,
        loss_epoch=0,
    )
    resume_path = out / "resume.pt"
    if resume and resume_path.exists():
        # Resume files are trusted, locally generated optimizer/RNG state, not release assets.
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        if saved["job"] != job:
            raise ValueError("Resume job differs")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        if ema:
            ema.load_state_dict({k: v.cuda() for k, v in saved["ema"].items()})
        state = saved["state"]
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
    started = time.perf_counter()
    previous_elapsed = (
        state["history"][-1]["elapsed_seconds"] if state["history"] else 0.0
    )
    torch.cuda.reset_peak_memory_stats()
    interrupted = False
    for epoch in range(len(state["history"]) + 1, maximum + 1):
        if (
            state["history"]
            and state["history"][-1]["epoch"] >= minimum
            and min(
                state["history"][-1]["stale_ap"], state["history"][-1]["stale_loss"]
            )
            >= patience
        ):
            break
        if (
            (out / "PAUSE").exists()
            or (out.parent.parent / "PAUSE").exists()
            or shutil.disk_usage(out).free < 8 * 2**30
        ):
            print("Paused; resume state retained.")
            interrupted = True
            break
        train_ds.epoch = epoch
        train_batches.generator.manual_seed(job["seed"] * 1000003 + epoch * 1000033)
        warmup = settings["warmup_epochs"] if enhanced else 0
        factor = learning_rate_factor(epoch - 1, maximum, min(warmup, maximum - 1))
        for group, lr in zip(optimizer.param_groups, lrs):
            group["lr"] = lr * factor
        training = epoch_pass(
            model, train_batches, job, optimizer, scaler, ema, teacher
        )
        with (
            ema.average_parameters(model)
            if ema is not None
            else contextlib.nullcontext()
        ):
            if bn_batches is not None:
                recalibrate_training_bn(model, bn_batches)
            valid = epoch_pass(model, val_batches, job)
            score = valid["ap"] / (1 + valid["loss"])
            if score > state["best"]:
                state["best"] = score
                save_checkpoint(
                    model,
                    out / "checkpoint.pt",
                    configuration,
                    epoch=epoch,
                    deterministic_algorithms=True,
                    validation=valid,
                    score=float(score),
                )
        if valid["ap"] >= state["ap_anchor"] + settings["ap_delta"]:
            state["ap_anchor"], state["ap_epoch"] = valid["ap"], epoch
        if valid["loss"] <= state["loss_anchor"] - settings["loss_delta"]:
            state["loss_anchor"], state["loss_epoch"] = valid["loss"], epoch
        row = dict(
            epoch=epoch,
            train_ap=training["ap"],
            train_loss=training["loss"],
            train_objective=training["total_objective"],
            validation_ap=valid["ap"],
            validation_loss=valid["loss"],
            score=score,
            stale_ap=epoch - state["ap_epoch"],
            stale_loss=epoch - state["loss_epoch"],
            elapsed_seconds=previous_elapsed + time.perf_counter() - started,
            objective_components=training["components"],
            peak_gpu_bytes=torch.cuda.max_memory_allocated(),
            rss_bytes=psutil.Process().memory_info().rss,
        )
        state["history"].append(row)
        write(out / "history.json", state["history"])
        temporary = resume_path.with_suffix(".tmp")
        torch.save(
            dict(
                job=job,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                scaler=scaler.state_dict(),
                ema=ema.state_dict() if ema else None,
                state=state,
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all(),
                python_rng=random.getstate(),
                numpy_rng=np.random.get_state(),
            ),
            temporary,
        )
        temporary.replace(resume_path)
        print(json.dumps(row), flush=True)
        if stop_after_epoch == epoch:
            interrupted = True
            break
    return dict(
        output=str(out),
        epochs_completed=len(state["history"]),
        test_access=False,
        smoke_only=smoke,
        interrupted=interrupted,
        selected_checkpoint=str(out / "checkpoint.pt"),
        selected_epoch=(
            torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=True)[
                "epoch"
            ]
            if (out / "checkpoint.pt").exists()
            else None
        ),
    )


def job_for(method, seed=42, target=0.7, cell=None):
    job = dict(
        method=method,
        seed=seed,
        input_mode="reflectance_change",
        loss="bce_dice",
        optimization="enhanced",
        architecture="resunet",
        recovery_weights=(
            dict(study()["recovery_loss"])
            if method in ("dcap", "taylor", "magnitude", "dense_recovery")
            else {k: 0.0 for k in study()["recovery_loss"]}
        ),
        small_object_pixels=study()["evaluation"]["small_object_pixels"],
        deterministic_algorithms=True,
    )
    if cell:
        record = next(
            c
            for c in read(ROOT / "configs/factorial.json")["cells"]
            if c["cell"] == cell
        )
        job.update(record)
    if method in ("dcap", "taylor", "magnitude", "compact"):
        job["target"] = target
    job["name"] = (
        (cell or method)
        + (f"_mac{round(target * 100)}" if "target" in job else "")
        + f"_seed{seed}"
    )
    return job
