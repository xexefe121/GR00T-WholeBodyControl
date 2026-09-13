"""One fixed teacher-energy weighting; retain the complete original54-cell loss.

This module has no nominal/physical loss, sampler, model or optimizer API.
Inputs and original arithmetic are delegated unchanged to full_state_loss.
"""
from typing import NamedTuple
import torch
from balance_contract import GROUP_WEIGHTS, CELL_GROUPS
from full_state_objective import full_state_loss


class BalancedResponseLoss(NamedTuple):
    weighted_objective: torch.Tensor
    original_objective: torch.Tensor
    original_cells: torch.Tensor
    weighted_cells: torch.Tensor


def balanced_full_state_loss(nominal_prediction, endpoint_prediction,
                            nominal_indices, teacher_change_normalized):
    original, cells = full_state_loss(nominal_prediction, endpoint_prediction,
                                     nominal_indices, teacher_change_normalized)
    # Fixed groups vary fastest inside each of the nine dataset/phase cells.
    # Use float64 on the same device; preserve all live center/endpoint gradients.
    weights = cells.new_tensor([GROUP_WEIGHTS[group] for group in CELL_GROUPS])
    weighted_cells = cells * weights
    return BalancedResponseLoss(weighted_cells.mean(), original, cells, weighted_cells)
