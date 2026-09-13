"""Physical finite responses; tensors only, no model, optimizer or runtime calls."""
import numpy as np
import torch

REQUESTED_COUNTS = (99, 819, 100) * 3

def physical_cells(dataset, successor_control):
    return [np.flatnonzero((dataset == d) & mask) for d in range(3) for mask in (
        (successor_control >= 251) & (successor_control < 350),
        (successor_control >= 350) & (successor_control < 1169),
        (successor_control >= 1169) & (successor_control < 1269))]

def physical_response_loss(normalized_centers, normalized_branches, successors,
                           center_base, branch_base, center_target, branch_target,
                           span, cells, requested_counts=REQUESTED_COUNTS):
    """Both live predictions; fixed requested denominators; no clip/detach/distance."""
    center_proposal = center_base[successors] + normalized_centers[successors] * span
    branch_proposal = branch_base + normalized_branches * span
    teacher_change = branch_target - center_target[successors]
    difference = ((branch_proposal - center_proposal) - teacher_change) / span
    square = difference ** 2
    assert len(cells) == len(requested_counts) == 9
    assert all(n > 0 for n in requested_counts)
    # Empty valid cells yield differentiable zero without evaluating NaN placeholders.
    per_cell = torch.stack([square[ids].sum() / (n * span.numel())
                            for ids, n in zip(cells, requested_counts)])
    return per_cell.mean(), per_cell

def expected_budgets(valid):
    assert isinstance(valid, int) and 0 <= valid <= 3054
    return dict(valid_rows=valid, updates=5000,
        training_head_rows=5000 * (3057 + 1152 + valid),
        legacy_head_onnx_calls=1126, physical_head_onnx_calls=2 * ((valid + 255) // 256),
        head_onnx_calls=1126 + 2 * ((valid + 255) // 256),
        diagnostic_torch_rows=2 * (143679 + 7 + valid),
        analytical_head_evaluations=14, BFM_calls=0, native_steps=0)
