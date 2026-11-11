"""Torch-Pruning dependency-group criteria."""

import torch
import torch_pruning as tp
from torch import nn


class LearnedGateGroupImportance(tp.importance.GroupMagnitudeImportance):
    """Aggregate learned normalization-gate utility over a DepGraph group."""

    def __init__(self, module_scores: dict[nn.Module, torch.Tensor]) -> None:
        super().__init__(group_reduction="mean", normalizer=None)
        self.module_scores = module_scores

    @torch.no_grad()
    def __call__(self, group, **kwargs):
        del kwargs
        local_scores = []
        root_indices = []
        functions = tp.pruner.function
        accepted = {
            functions.prune_batchnorm_out_channels,
            functions.prune_conv_out_channels,
            functions.prune_linear_out_channels,
        }
        for index, (dependency, indices) in enumerate(group):
            layer = dependency.layer
            if dependency.pruning_fn not in accepted or layer not in self.module_scores:
                continue
            score = self.module_scores[layer]
            local_scores.append(score[indices])
            root_indices.append(group[index].root_idxs)
        if not local_scores:
            return None
        return self._reduce(local_scores, root_indices)


def build_importance(method):
    if method == "group_magnitude_l2":
        return tp.importance.GroupMagnitudeImportance(
            p=2, group_reduction="mean", normalizer="mean"
        )
    if method == "group_taylor":
        return tp.importance.GroupTaylorImportance(
            group_reduction="mean", normalizer="mean"
        )
    raise ValueError(method)
