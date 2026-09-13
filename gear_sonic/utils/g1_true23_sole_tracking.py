"""Native physical sole-point tracking, a SIM training cost, not contact proof."""

SOLE_SPHERE_CENTERS_M = (
    (-0.05, 0.025, -0.03),
    (-0.05, -0.025, -0.03),
    (0.12, 0.03, -0.03),
    (0.12, -0.03, -0.03),
)
SOLE_SPHERE_RADIUS_M = 0.005
SOLE_TRACKING_NORMALIZATION_M = 0.02


def sole_points_world(position, quaternion):
    """Lowest world-Z points of the four physical spheres on each ankle body."""
    import torch

    if (
        not isinstance(position, torch.Tensor)
        or not isinstance(quaternion, torch.Tensor)
        or position.dtype not in (torch.float32, torch.float64)
        or position.ndim != 3
        or position.shape[1:] != (2, 3)
        or quaternion.shape != (*position.shape[:-1], 4)
        or quaternion.dtype != position.dtype
        or quaternion.device != position.device
    ):
        raise ValueError(
            "sole tracking needs matching floating [env,2,3] positions and [env,2,4] WXYZ quaternions"
        )
    if not torch.isfinite(position).all() or not torch.isfinite(quaternion).all():
        raise ValueError("sole tracking requires finite measured or received states")
    if not torch.all(torch.abs(torch.linalg.vector_norm(quaternion, dim=-1) - 1) <= 1e-4):
        raise ValueError("sole tracking requires unit WXYZ body quaternions")
    points = torch.as_tensor(SOLE_SPHERE_CENTERS_M, dtype=position.dtype, device=position.device)
    points = points.expand(*position.shape[:-1], 4, 3)
    vector = quaternion[..., None, 1:].expand_as(points)
    twice_cross = 2 * torch.linalg.cross(vector, points)
    rotated = points + quaternion[..., None, :1] * twice_cross + torch.linalg.cross(vector, twice_cross)
    down = torch.tensor([0, 0, SOLE_SPHERE_RADIUS_M], dtype=position.dtype, device=position.device)
    return position[..., None, :] + rotated - down


def sole_world_position_l2(desired_position, desired_quaternion, measured_position, measured_quaternion):
    """One coupled stance-placement and swing-clearance cost; no action teacher."""
    desired = sole_points_world(desired_position, desired_quaternion)
    measured = sole_points_world(measured_position, measured_quaternion)
    if desired.shape != measured.shape or desired.dtype != measured.dtype or desired.device != measured.device:
        raise ValueError("measured and desired sole points must have identical batch, dtype and device")
    return ((measured - desired) / SOLE_TRACKING_NORMALIZATION_M).square().sum(-1).mean(dim=(-1, -2))
