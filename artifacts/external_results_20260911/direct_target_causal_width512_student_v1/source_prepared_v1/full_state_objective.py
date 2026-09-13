"""54-cell finite-feedback matching over declared physical neighborhoods."""
import torch

def full_state_loss(nominal_prediction, endpoint_prediction, nominal_indices, teacher_change_normalized):
    if endpoint_prediction.shape != (1728, 23) or nominal_indices.shape != (864,):
        raise ValueError('One fixed864-pair full-state batch required.')
    if teacher_change_normalized.shape != (864, 2, 23) or teacher_change_normalized.dtype != torch.float64:
        raise ValueError('Fixed float64 target-span-normalized changes required.')
    if nominal_prediction.dtype != torch.float32 or endpoint_prediction.dtype != torch.float32:
        raise ValueError('Float32 model predictions required.')
    center = nominal_prediction[nominal_indices].to(torch.float64)
    endpoint = endpoint_prediction.to(torch.float64).reshape(864, 2, 23)
    error = endpoint - center[:, None] - teacher_change_normalized
    square = error ** 2
    per_cell = torch.stack([square[cell * 16:(cell + 1) * 16].mean() for cell in range(54)])
    return per_cell.mean(), per_cell
