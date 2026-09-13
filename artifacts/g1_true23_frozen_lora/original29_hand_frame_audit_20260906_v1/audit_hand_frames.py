"""Save the neutral-frame derivation and actual hand-mesh vertex witness.

Nearest-vertex distances are not continuous triangle-surface distances or
proof that the two hand meshes are identical. No source files are edited.
"""

import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.spatial import cKDTree

from gear_sonic.scripts.record_g1_sonic_original29_baseline import dump
from gear_sonic.utils.g1_sonic_cpp_parameters import file_sha256
from gear_sonic.utils.g1_true23_hand_frame_tasks import neutral_wrist_hand_tasks
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent


def main():
    if (HERE / "report.json").exists():
        raise FileExistsError("preserve the prior audit")
    inputs = {}

    def bind(path, expected=None):
        path = Path(path).resolve(strict=True)
        digest = file_sha256(path)
        if (expected is not None and digest != expected) or digest != inputs.get(str(path), digest):
            raise ValueError("audit input changed")
        inputs[str(path)] = digest
        return path

    baseline = PARENT / "original29_recorded_baseline_20260906_v1"
    fitted = PARENT / "original29_recorded_native23_fit_20260906_v1"
    baseline_report = json.loads(bind(baseline / "report.json").read_text())
    fitted_report = json.loads(bind(fitted / "report.json").read_text())
    source = mujoco.MjModel.from_binary_path(
        str(bind(baseline / "original29.mjb", baseline_report["compiled_model_sha256"]))
    )
    target = mujoco.MjModel.from_binary_path(
        str(bind(fitted / "native23.mjb", fitted_report["compiled_models"]["target"]))
    )
    hashes = {"source": compiled_model_sha256(source), "target": compiled_model_sha256(target)}
    _, frames = neutral_wrist_hand_tasks(source, target)
    rows = []
    for side in ("left", "right"):
        vertices, meshes = [], []
        for model, mesh_name in ((source, side + "_rubber_hand"), (target, side + "_wrist_roll_rubber_hand")):
            data = mujoco.MjData(model)
            data.qpos[:] = 0
            data.qpos[2:4] = [0.8, 1]
            mujoco.mj_fwdPosition(model, data)
            geom = next(
                geom
                for geom in range(model.ngeom)
                if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH
                and model.mesh(int(model.geom_dataid[geom])).name == mesh_name
            )
            mesh = int(model.geom_dataid[geom])
            first, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
            world = (
                model.mesh_vert[first : first + count] @ data.geom_xmat[geom].reshape(3, 3).T
                + data.geom_xpos[geom]
            )
            roll = int(model.joint(side + "_wrist_roll_joint").bodyid[0])
            local = (world - data.xpos[roll]) @ data.xmat[roll].reshape(3, 3)
            vertices.append(local)
            meshes.append(
                {
                    "mesh": mesh_name,
                    "vertices": count,
                    "wrist_roll_local_min_m": local.min(0).tolist(),
                    "wrist_roll_local_max_m": local.max(0).tolist(),
                }
            )
        distances = cKDTree(vertices[1]).query(vertices[0])[0]
        rows.append(
            {
                "side": side,
                "meshes": meshes,
                "source_vertices_to_nearest_native_vertex_max_m": float(distances.max()),
                "source_vertices_to_nearest_native_vertex_median_m": float(np.median(distances)),
                "fraction_source_vertices_with_native_vertex_within_1mm": float(np.mean(distances < 0.001)),
                "identical_mesh_geometry_proven": False,
                "distances_are_vertices_not_continuous_triangle_surfaces": True,
            }
        )
    bind(Path(__file__))
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith("gear_sonic.") and path and Path(path).suffix == ".py":
            bind(path)
    for path in list(inputs):
        bind(path)
    if hashes != {"source": compiled_model_sha256(source), "target": compiled_model_sha256(target)}:
        raise ValueError("audit changed a model")
    dump(
        HERE / "report.json",
        {
            "kind": "g1_true23_neutral_hand_frame_and_mesh_audit_v1",
            "inputs": inputs,
            "compiled_models": hashes,
            "frame_derivation": frames,
            "mesh_vertex_witnesses": rows,
            "mujoco_version": mujoco.__version__,
            "task_proxy_is_not_a_physical_contact_landmark": True,
            "default_teleop_or_policy_input_convention_modified": False,
            "teacher_accepted": False,
            "hardware_authorized": False,
            "deployment_ready": False,
        },
    )
    print(HERE / "report.json", flush=True)


if __name__ == "__main__":
    main()
