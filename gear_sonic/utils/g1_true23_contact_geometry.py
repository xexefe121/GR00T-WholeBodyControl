"""Independent flat-floor contact geometry; no dynamics or robot control."""

from __future__ import annotations

import numpy as np


def audit_reset_contacts(model, qpos):
    """Inspect actual compiled geometry in separate CPU data, without stepping."""
    import mujoco

    qpos = np.asarray(qpos)
    if qpos.ndim != 2 or qpos.shape[1] != model.nq or not np.isfinite(qpos).all():
        raise ValueError("contact audit requires finite [environments, nq] positions")
    probe = mujoco.MjData(model)
    plane_ids = set(np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE).tolist())
    rows = []
    for index, position in enumerate(qpos):
        probe.qpos[:] = position
        mujoco.mj_fwdPosition(model, probe)
        contacts = []
        for contact in probe.contact[: probe.ncon]:
            first, second = map(int, contact.geom)
            contacts.append(
                {
                    "geom1": model.geom(first).name,
                    "geom2": model.geom(second).name,
                    "floor_contact": first in plane_ids or second in plane_ids,
                    "distance_m": float(contact.dist),
                }
            )
        floor = [row for row in contacts if row["floor_contact"]]
        rows.append(
            {
                "env_index": index,
                "minimum_floor_contact_distance_m": min((row["distance_m"] for row in floor), default=None),
                "floor_penetration_contact_count": sum(row["distance_m"] < 0 for row in floor),
                "worst_contacts": sorted(contacts, key=lambda row: row["distance_m"])[:6],
            }
        )
    return {
        "method": "compiled_model_cpu_position_forward_on_independent_data_no_physics_step",
        "negative_distance_means_geometric_overlap": True,
        "dynamics_or_impulse_causation_proven": False,
        "rows": rows,
    }


def lift_reset_floor_overlap(model, qpos, *, clearance_m=1e-5, maximum_lift_m=0.2):
    """Diagnostic reset intervention, not contact retargeting or robot control.

    Lift only penetrated states of a single floating articulation over a flat
    world plane. Preserve every joint, orientation and horizontal coordinate;
    velocities are not inputs and must be preserved by the caller. Never lower
    airborne states. Reject unsupported geometry or excessive corrections.
    """
    import mujoco

    if not np.isfinite([clearance_m, maximum_lift_m]).all() or not 0 < clearance_m < maximum_lift_m:
        raise ValueError("floor-lift bounds must be finite, positive and ordered")
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    planes = np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
    if len(free) != 1 or len(planes) != 1:
        raise ValueError("floor lift requires one free articulation and one static world plane")
    plane_body = int(model.geom_bodyid[planes[0]])
    while plane_body != 0:
        if model.body_jntnum[plane_body] != 0 or model.body_mocapid[plane_body] >= 0:
            raise ValueError("floor lift requires a plane welded to the world")
        plane_body = int(model.body_parentid[plane_body])
    probe = mujoco.MjData(model)
    mujoco.mj_fwdPosition(model, probe)
    if not np.allclose(probe.geom_xmat[planes[0]].reshape(3, 3)[:, 2], [0, 0, 1], atol=1e-8, rtol=0):
        raise ValueError("floor lift requires a horizontal upward-facing plane")
    root_body = int(model.jnt_bodyid[free[0]])
    for geom_body in model.geom_bodyid[np.arange(model.ngeom) != planes[0]]:
        body = int(geom_body)
        while body not in (0, root_body):
            body = int(model.body_parentid[body])
        if body != root_body:
            raise ValueError("floor lift requires all non-plane geometry on the free articulation")
    before = audit_reset_contacts(model, qpos)
    lifted = np.array(qpos, copy=True)
    if lifted.dtype.kind != "f":
        raise ValueError("floor-lift positions must have floating dtype")
    root_z = int(model.jnt_qposadr[free[0]]) + 2
    lifts = np.array(
        [
            max(0.0, clearance_m - row["minimum_floor_contact_distance_m"])
            if row["minimum_floor_contact_distance_m"] is not None and row["minimum_floor_contact_distance_m"] < 0
            else 0.0
            for row in before["rows"]
        ]
    )
    if np.any(lifts > maximum_lift_m):
        raise ValueError("reset floor penetration exceeds diagnostic lift bound")
    lifted[:, root_z] += lifts
    after = audit_reset_contacts(model, lifted)
    if any(
        row["minimum_floor_contact_distance_m"] is not None and row["minimum_floor_contact_distance_m"] < 0
        for row in after["rows"]
    ):
        raise ValueError("floor-lift correction left geometric penetration")
    return lifted, {
        "clearance_m": clearance_m,
        "maximum_lift_m": maximum_lift_m,
        "root_z_qpos_index": root_z,
        "actual_lifts_m": (lifted[:, root_z] - np.asarray(qpos)[:, root_z]).tolist(),
        "before": before,
        "after": after,
        "reference_motion_changed": False,
        "dynamic_contact_consistency_proven": False,
    }
