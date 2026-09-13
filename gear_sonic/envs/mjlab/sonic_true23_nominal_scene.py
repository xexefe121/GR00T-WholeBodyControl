"""Opt-in training on the pinned CPU-referee robot, contacts and solver.

This replaces the training asset, not the evaluation model or physical robot.
The action manager still clips finite torques to the native effort limits and
the compiled joints retain those same force limits. No runtime state rewrite.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES
from gear_sonic.utils.g1_true23_generalist_corpus import sha256_file
from gear_sonic.utils.g1_true23_native_model_actuation import NativeModelActuationProfile
from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model

NOMINAL_SCENE_PROFILE = "pinned_cpu_referee_scene_v1"
GEOM_FIELDS = (
    "type",
    "size",
    "pos",
    "quat",
    "contype",
    "conaffinity",
    "condim",
    "priority",
    "solmix",
    "solref",
    "solimp",
    "friction",
    "margin",
    "gap",
)
OPTION_FIELDS = (
    "timestep",
    "integrator",
    "cone",
    "solver",
    "iterations",
    "ls_iterations",
    "tolerance",
    "ls_tolerance",
    "ccd_iterations",
    "ccd_tolerance",
    "impratio",
    "gravity",
    "wind",
    "density",
    "viscosity",
    "disableflags",
    "enableflags",
    "jacobian",
)


def _assign(obj, name, value):
    current = getattr(obj, name)
    if isinstance(current, np.ndarray):
        current[:] = value
    else:
        if type(current).__module__ == "mujoco._enums":
            value = type(current)(int(value))
        elif isinstance(value, np.generic):
            value = value.item()
        setattr(obj, name, value)


def nominal_scene_contract(model_path, sim_config):
    model_path, sim_config = (Path(p).resolve(strict=True) for p in (model_path, sim_config))
    # Validate the existing model/config pins before accepting mesh references.
    prepare_true23_model(model_path, sim_config)
    spec = mujoco.MjSpec.from_file(str(model_path))
    inputs = {str(p): sha256_file(p) for p in (model_path, sim_config)}
    for mesh in spec.meshes:
        path = (model_path.parent / spec.meshdir / mesh.file).resolve(strict=True)
        if not path.is_relative_to(model_path.parent) or not path.is_file():
            raise ValueError("nominal mesh must be a regular file inside the pinned asset directory")
        inputs[str(path)] = sha256_file(path)
    payload = {
        "name": NOMINAL_SCENE_PROFILE,
        "inputs": inputs,
        "training_robot_geometry": "pinned_native23_cpu_referee_mjcf_and_meshes",
        "contacts_root_passive_parameters_and_solver": "unchanged_cpu_referee_values",
        "reference_and_evaluation_unchanged": True,
        "physical_effort_limits_unchanged": True,
        "hardware_authorized": False,
        "deployment_ready": False,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {**payload, "contract_sha256": digest}


def configure_nominal_scene(cfg, model_path, sim_config):
    """Return a new config; keep nominal CPU assets and caller config untouched."""
    contract = nominal_scene_contract(model_path, sim_config)
    model_path, sim_config = (Path(p).resolve(strict=True) for p in (model_path, sim_config))
    _, reference, _ = prepare_true23_model(model_path, sim_config)
    profile = NativeModelActuationProfile.from_sim_config(sim_config)
    if getattr(cfg.actions["joint_pos"], "profile", None) != profile:
        raise ValueError("nominal scene requires the matching explicit native motor action profile")
    if cfg.scene.spec_fn is not None or any(
        k in cfg.events for k in ("foot_friction", "base_com", "encoder_bias")
    ):
        raise ValueError("nominal scene refuses competing geometry/physics randomization")
    result = copy.deepcopy(cfg)
    robot = result.scene.entities["robot"]

    def robot_spec():
        for path, digest in contract["inputs"].items():
            if sha256_file(Path(path)) != digest:
                raise ValueError("nominal scene input changed before construction")
        spec = mujoco.MjSpec.from_file(str(model_path))
        spec.assets = {
            mesh.file: (model_path.parent / spec.meshdir / mesh.file).read_bytes() for mesh in spec.meshes
        }
        # Scene owns exactly one floor. Removal is from this in-memory spec only.
        spec.delete(spec.geom("floor"))
        for actuator in list(spec.actuators):
            spec.delete(actuator)
        for sensor in list(spec.sensors):
            spec.delete(sensor)
        for name, kind in (
            ("imu_ang_vel", mujoco.mjtSensor.mjSENS_GYRO),
            ("imu_lin_vel", mujoco.mjtSensor.mjSENS_VELOCIMETER),
            ("imu_lin_acc", mujoco.mjtSensor.mjSENS_ACCELEROMETER),
        ):
            spec.add_sensor(name=name, type=kind, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu_in_pelvis")
        spec.add_sensor(
            name="root_angmom",
            type=mujoco.mjtSensor.mjSENS_SUBTREEANGMOM,
            objtype=mujoco.mjtObj.mjOBJ_BODY,
            objname="pelvis",
        )
        for i, name in enumerate(HARDWARE_23_JOINT_NAMES):
            joint = spec.joint(name)
            joint.damping = profile.damping[i]
            joint.actfrclimited = True
            joint.actfrcrange[:] = (-profile.effort[i], profile.effort[i])
        return spec

    def scene_spec(spec):
        for name in OPTION_FIELDS:
            _assign(spec.option, name, getattr(reference.opt, name))
        floor = spec.worldbody.add_geom(name="floor")
        floor_id = reference.geom("floor").id
        for name in GEOM_FIELDS:
            _assign(floor, name, getattr(reference, "geom_" + name)[floor_id])
        if len(spec.actuators) != 23:
            raise ValueError("nominal scene must retain exactly 23 torque actuators")
        for i, actuator in enumerate(spec.actuators):
            if actuator.target != "robot/" + HARDWARE_23_JOINT_NAMES[i]:
                raise ValueError("nominal scene actuator order differs from hardware23")
            # Match the referee's motor representation, including redundant
            # ctrl/actuator clamps. The native action clip and joint-force caps
            # remain authoritative and unchanged in both execution paths.
            for name in ("gear", "gainprm", "biasprm", "forcerange", "ctrlrange"):
                _assign(actuator, name, getattr(reference, "actuator_" + name)[i])
            for name in ("forcelimited", "ctrllimited", "dyntype", "gaintype", "biastype"):
                _assign(actuator, name, getattr(reference, "actuator_" + name)[i])

    robot.spec_fn = robot_spec
    robot.collisions = ()  # Preserve source MJCF contact masks, shapes and coefficients.
    result.scene.terrain = None
    result.scene.spec_fn = scene_spec
    options = result.sim.mujoco
    for name in (
        "timestep",
        "impratio",
        "iterations",
        "tolerance",
        "ls_iterations",
        "ls_tolerance",
        "ccd_iterations",
    ):
        setattr(options, name, getattr(reference.opt, name))
    options.gravity = tuple(reference.opt.gravity)
    options.integrator, options.cone, options.solver, options.jacobian = "euler", "pyramidal", "newton", "auto"
    options.multiccd = False
    return result


def verify_nominal_scene(model, model_path, sim_config):
    """Fail closed on constructed physical drift; ignore names/visual-only data."""
    from gear_sonic.scripts.audit_g1_true23_training_replay_parity import (
        collision_parameters,
        compare_model_parameters,
        validate_native_layout,
    )

    _, reference, _ = prepare_true23_model(Path(model_path), Path(sim_config))
    validate_native_layout(model, "robot/")
    checks = compare_model_parameters(model, reference)
    failures = [name for name, value in checks.items() if value.get("within_tolerance") is False]
    for name in ("dof_M0", "dof_invweight0", "body_invweight0", "actuator_acc0"):
        if not np.allclose(getattr(model, name), getattr(reference, name), atol=1e-9, rtol=0):
            failures.append("derived." + name)
    for name in OPTION_FIELDS:
        if not np.array_equal(getattr(model.opt, name), getattr(reference.opt, name)):
            failures.append("option." + name)

    # Unnamed geometries use stable model order; only the scene's floor moves
    # relative to robot geoms during attachment. No shape/parameter omission.
    def collisions(value):
        rows = []
        for row in collision_parameters(value):
            rows.append({**row, "name": "", "body": row["body"].removeprefix("robot/")})
        return sorted(rows, key=lambda r: json.dumps(r, sort_keys=True))

    if collisions(model) != collisions(reference):
        failures.append("collision_geometry_and_parameters")
    for name in ("body_pos", "body_quat"):
        ids = [model.body("robot/" + reference.body(i).name).id for i in range(1, reference.nbody)]
        if not np.allclose(getattr(model, name)[ids], getattr(reference, name)[1:], atol=1e-12, rtol=0):
            failures.append(name)
    if failures:
        raise ValueError("constructed nominal scene differs from CPU referee: " + ", ".join(failures))
    return {
        "kind": "constructed_native23_nominal_scene_parameter_parity_v1",
        "profile": nominal_scene_contract(model_path, sim_config),
        "parameter_parity_passed": True,
        "derived_solver_constants_checked": True,
        "collision_geom_count": len(collisions(model)),
        "numeric_engine_parity_proven": False,
        "dance_tracking_qualified": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }
