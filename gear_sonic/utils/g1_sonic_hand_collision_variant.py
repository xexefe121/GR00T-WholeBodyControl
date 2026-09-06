"""Isolated original29 hand-collision experiment; never a hardware model edit.

Recompile an mjSpec copy rather than patching compiled collision masks without
rebuilding MuJoCo's acceleration structures. Preserve the recorded baseline
and all physical geometry, inertias, joints, actuators and solver settings.
"""

from __future__ import annotations

import mujoco
import numpy as np

from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256

HAND_MESHES = ("left_rubber_hand", "right_rubber_hand")
COLLISION_DERIVED_FIELDS = frozenset(
    (
        "body_bvhadr",
        "body_bvhnum",
        "bvh_aabb",
        "bvh_child",
        "bvh_depth",
        "bvh_nodeid",
        "mesh_bvhadr",
        "mesh_graph",
        "mesh_graphadr",
        "mesh_polyadr",
        "mesh_polymap",
        "mesh_polymapadr",
        "mesh_polymapnum",
        "mesh_polynormal",
        "mesh_polynum",
        "mesh_polyvert",
        "mesh_polyvertadr",
        "mesh_polyvertnum",
        "nbuffer",
        "nbvh",
        "nbvhstatic",
        "nmeshgraph",
        "nmeshpoly",
        "nmeshpolymap",
        "nmeshpolyvert",
    )
)


def numeric_differences(left, right):
    changed = []
    for name in dir(left):
        if name.startswith("_"):
            continue
        before, after = getattr(left, name), getattr(right, name)
        if isinstance(before, np.ndarray):
            if before.shape != after.shape or not np.array_equal(before, after, equal_nan=True):
                changed.append(name)
        elif isinstance(before, (int, float, str, bytes)) and before != after:
            changed.append(name)
    return changed


def hand_geom_ids(model):
    result = [
        geom
        for geom in range(model.ngeom)
        if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH
        and model.mesh(int(model.geom_dataid[geom])).name in HAND_MESHES
    ]
    if len(result) != 2 or [model.mesh(int(model.geom_dataid[g])).name for g in result] != list(HAND_MESHES):
        raise ValueError("requires exactly the two original29 rubber-hand meshes")
    return result


def compile_hand_collision_variant(scene_path, *, expected_baseline_sha256):
    """Return untouched baseline plus a separately recompiled two-mask variant."""
    spec = mujoco.MjSpec.from_file(str(scene_path))
    baseline = spec.compile()
    baseline.opt.timestep = 0.002
    original_hash = compiled_model_sha256(baseline)
    if original_hash != expected_baseline_sha256 or (baseline.nq, baseline.nv, baseline.nu) != (36, 35, 29):
        raise ValueError("source scene is not the exact recorded original29 baseline")
    hands = hand_geom_ids(baseline)
    if baseline.npair or any(baseline.geom_contype[g] or baseline.geom_conaffinity[g] for g in hands):
        raise ValueError("the experiment requires visual-only source hands without explicit contact pairs")
    copied = spec.copy()
    selected = [geom for geom in copied.geoms if geom.meshname in HAND_MESHES]
    if len(selected) != 2 or any(geom.density != 0 for geom in selected):
        raise ValueError("hand collision activation must not add mass or duplicate hand geometry")
    for geom in selected:
        geom.contype = geom.conaffinity = 1
    variant = copied.compile()
    variant.opt.timestep = 0.002
    if hand_geom_ids(variant) != hands:
        raise ValueError("recompilation changed hand geometry identity or ordering")
    differences = numeric_differences(baseline, variant)
    if set(differences) - COLLISION_DERIVED_FIELDS - {"geom_contype", "geom_conaffinity"}:
        raise ValueError(f"collision experiment changed unrelated model data: {differences}")
    if numeric_differences(baseline.opt, variant.opt) or numeric_differences(baseline.stat, variant.stat):
        raise ValueError("collision experiment changed solver settings or model statistics")
    for field in ("geom_contype", "geom_conaffinity"):
        before, after = getattr(baseline, field), getattr(variant, field)
        if not np.array_equal(np.flatnonzero(before != after), hands) or not np.array_equal(after[hands], [1, 1]):
            raise ValueError("collision masks changed outside the two original hand meshes")
    if compiled_model_sha256(baseline) != original_hash:
        raise ValueError("source baseline was mutated")
    return (
        baseline,
        variant,
        {
            "kind": "g1_sonic_original29_hand_collision_variant_v1",
            "baseline_compiled_sha256": original_hash,
            "variant_compiled_sha256": compiled_model_sha256(variant),
            "activated_hand_geom_ids": hands,
            "numeric_model_fields_changed": differences,
            "noncollision_numeric_arrays_unchanged": True,
            "joint_count": 29,
            "original_source_unchanged": True,
            "native23_model_modified": False,
            "stock_deployment_equivalence_proven": False,
            "teacher_accepted": False,
            "hardware_authorized": False,
            "deployment_ready": False,
        },
    )


def inspect_hand_contacts(model, poses):
    poses = np.asarray(poses, dtype=float)
    if poses.ndim != 2 or poses.shape[1] != 36 or not len(poses) or not np.isfinite(poses).all():
        raise ValueError("hand contact inspection requires finite complete original29 poses")
    hands = hand_geom_ids(model)
    floors = set(np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE))
    if len(floors) != 1:
        raise ValueError("hand inspection needs one fixed horizontal world floor")
    floor = next(iter(floors))
    data, lowest, distances, nonfloor_distances = mujoco.MjData(model), [], [], []
    for pose in poses:
        data.qpos[:] = pose
        mujoco.mj_fwdPosition(model, data)
        if model.geom_bodyid[floor] != 0 or not np.allclose(
            data.geom_xmat[floor].reshape(3, 3)[:, 2], [0, 0, 1], atol=1e-10, rtol=0
        ):
            raise ValueError("hand inspection supports only the fixed horizontal floor")
        vertices_z = []
        for geom in hands:
            mesh = int(model.geom_dataid[geom])
            start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
            vertices = (
                model.mesh_vert[start : start + count] @ data.geom_xmat[geom].reshape(3, 3).T
                + data.geom_xpos[geom]
            )
            vertices_z.append(float(vertices[:, 2].min() - data.geom_xpos[floor, 2]))
        lowest.append(min(vertices_z))
        contacts = [
            float(contact.dist)
            for contact in data.contact[: data.ncon]
            if floor in contact.geom and any(geom in contact.geom for geom in hands)
        ]
        distances.append(min(contacts, default=None))
        other = [
            float(contact.dist)
            for contact in data.contact[: data.ncon]
            if floor not in contact.geom and any(geom in contact.geom for geom in hands)
        ]
        nonfloor_distances.append(min(other, default=None))
    return {
        "frames": len(poses),
        "minimum_hand_vertex_clearance_m": min(lowest),
        "frames_with_hand_mesh_below_floor": sum(value < 0 for value in lowest),
        "frames_with_active_hand_floor_contacts": sum(value is not None for value in distances),
        "minimum_active_hand_contact_distance_m": min((x for x in distances if x is not None), default=None),
        "lowest_vertex_clearance_by_frame_m": lowest,
        "active_hand_contact_distance_by_frame_m": distances,
        "frames_with_active_hand_nonfloor_contacts": sum(value is not None for value in nonfloor_distances),
        "minimum_active_hand_nonfloor_contact_distance_m": min(
            (x for x in nonfloor_distances if x is not None), default=None
        ),
        "activation_changes_hand_floor_and_self_collision_masks": True,
        "physical_contact_feasibility_proven": False,
    }
