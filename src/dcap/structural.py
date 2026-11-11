"""Measured-MAC structural pruning shared by the baseline criteria."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable

import torch
import torch_pruning as tp

from .importance import build_importance


@dataclass(frozen=True)
class PruningMeasurement:
    baseline_macs: int
    baseline_parameters: int
    pruned_macs: int
    pruned_parameters: int
    achieved_mac_reduction: float
    selected_step: int
    within_tolerance: bool


def output_layers_to_preserve(model: torch.nn.Module) -> list[torch.nn.Module]:
    head = getattr(model, "head", None)
    if not isinstance(head, torch.nn.Module):
        raise ValueError("Pruning requires a model with an explicit head module")
    return [head]


def prune_to_measured_mac_target(
    model: torch.nn.Module,
    example_inputs: torch.Tensor,
    *,
    method: str,
    target_mac_reduction: float,
    iterative_steps: int = 100,
    maximum_channel_pruning_ratio: float = 0.90,
    achieved_mac_tolerance: float = 0.01,
    round_to: int | None = 1,
    random_seed: int = 2026,
    before_step: Callable[[int, torch.nn.Module], None] | None = None,
) -> tuple[torch.nn.Module, PruningMeasurement, list[dict]]:
    if not 0 < target_mac_reduction < 1:
        raise ValueError("target_mac_reduction must be in (0, 1)")
    if iterative_steps < 2:
        raise ValueError("iterative_steps must be at least two")
    if not 0 < maximum_channel_pruning_ratio < 1:
        raise ValueError("maximum_channel_pruning_ratio must be in (0, 1)")
    if achieved_mac_tolerance < 0:
        raise ValueError("achieved_mac_tolerance must be non-negative")
    torch.manual_seed(random_seed)
    if example_inputs.is_cuda:
        torch.cuda.manual_seed_all(random_seed)
    model.eval()
    baseline_macs, baseline_parameters = tp.utils.count_ops_and_params(
        model, example_inputs
    )
    pruner = tp.pruner.MetaPruner(
        model,
        example_inputs,
        importance=build_importance(method),
        global_pruning=True,
        pruning_ratio=maximum_channel_pruning_ratio,
        iterative_steps=iterative_steps,
        ignored_layers=output_layers_to_preserve(model),
        round_to=round_to,
    )
    trace = [
        {
            "step": 0,
            "scheduled_channel_pruning_ratio": 0.0,
            "macs": int(baseline_macs),
            "parameters": int(baseline_parameters),
            "mac_reduction": 0.0,
            "parameter_reduction": 0.0,
            "selected": False,
        }
    ]
    selected_model = None
    selected_record = None
    device = example_inputs.device
    for step in range(1, iterative_steps + 1):
        previous_model = copy.deepcopy(model).cpu()
        previous_record = trace[-1]
        if before_step is not None:
            before_step(step, model)
        pruner.step()
        model.zero_grad(set_to_none=True)
        macs, parameters = tp.utils.count_ops_and_params(model, example_inputs)
        record = {
            "step": step,
            "scheduled_channel_pruning_ratio": float(
                pruner.per_step_pruning_ratio[step]
            ),
            "macs": int(macs),
            "parameters": int(parameters),
            "mac_reduction": float(1 - macs / baseline_macs),
            "parameter_reduction": float(1 - parameters / baseline_parameters),
            "selected": False,
        }
        trace.append(record)
        if record["mac_reduction"] >= target_mac_reduction:
            if abs(previous_record["mac_reduction"] - target_mac_reduction) < abs(
                record["mac_reduction"] - target_mac_reduction
            ):
                selected_model = previous_model.to(device)
                selected_record = previous_record
            else:
                selected_model = model
                selected_record = record
            break
        del previous_model
    if selected_model is None or selected_record is None:
        raise RuntimeError("Pruning schedule did not reach the requested MAC target")
    selected_record["selected"] = True
    achieved = float(selected_record["mac_reduction"])
    measurement = PruningMeasurement(
        baseline_macs=int(baseline_macs),
        baseline_parameters=int(baseline_parameters),
        pruned_macs=int(selected_record["macs"]),
        pruned_parameters=int(selected_record["parameters"]),
        achieved_mac_reduction=achieved,
        selected_step=int(selected_record["step"]),
        within_tolerance=abs(achieved - target_mac_reduction) <= achieved_mac_tolerance,
    )
    return selected_model, measurement, trace
