"""Original-reference transition guidance for offline local search directions.

This does not alter the acceptance objective, source frames, physical model or
forces. It adds transition COM, under-lift and optional foot-placement terms
only to local quadratics.
Actual full-trajectory replay and the original metrics still decide acceptance.
"""

import mujoco
import numpy as np

from gear_sonic.utils.g1_true23_reference_floor import motion_qpos


class EntryGuidance:
    def __init__(self, model, motion, timeline, *, com_weight=10000.0, lift_weight=25000.0, foot_xy_weight=0.0):
        weights = (com_weight, lift_weight, foot_xy_weight)
        if not np.isfinite(weights).all() or min(weights) < 0:
            raise ValueError("entry guide weights must be finite and nonnegative")
        self.model, self.data = model, mujoco.MjData(model)
        self.com_weight, self.lift_weight = float(com_weight), float(lift_weight)
        self.foot_xy_weight = float(foot_xy_weight)
        self.pelvis = model.body("pelvis").id
        self.feet = [model.body(name).id for name in ("left_ankle_roll_link", "right_ankle_roll_link")]
        poses = motion_qpos(model, motion)[10:]
        count = timeline["total_requested_controls"]
        if len(poses) != count + 1:
            raise ValueError("entry guide must retain the exact original full lifecycle")
        self.active, self.swing = np.zeros(count + 1, bool), np.zeros((count + 1, 2), bool)
        self.com = np.zeros((count + 1, 2))
        self.foot_z = np.zeros((count + 1, 2))
        self.foot_xy = np.zeros((count + 1, 2, 2))
        self.phases = []
        for phase in timeline["phases"]:
            if phase["name"] not in ("acquisition_ramp", "return_ramp"):
                continue
            start, stop = phase["control_start"], phase["control_stop"]
            self.phases.append(dict(name=phase["name"], control_start=start, control_stop=stop))
            for index in range(start, stop + 1):
                self.data.qpos[:] = poses[index]
                mujoco.mj_kinematics(model, self.data)
                mujoco.mj_comPos(model, self.data)
                self.com[index] = self.data.subtree_com[self.pelvis, :2]
                self.foot_z[index] = self.data.xpos[self.feet, 2]
                self.foot_xy[index] = self.data.xpos[self.feet, :2]
            self.active[start + 1 : stop + 1] = True
            baseline = self.foot_z[start : stop + 1].min(axis=0)
            self.swing[start + 1 : stop + 1] = self.foot_z[start + 1 : stop + 1] > baseline + 0.005

    def state_terms(self, pose, index):
        gradient, hessian = np.zeros(81), np.zeros((81, 81))
        if not self.active[index]:
            return 0.0, gradient, hessian
        data, model = self.data, self.model
        data.qpos[:], data.qvel[:] = pose, 0
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        jacobian = np.zeros((3, 29))
        mujoco.mj_jacSubtreeCom(model, data, jacobian, self.pelvis)
        errors = list(np.sqrt(self.com_weight) * (data.subtree_com[self.pelvis, :2] - self.com[index]))
        rows = list(np.sqrt(self.com_weight) * jacobian[:2])
        for side, foot in enumerate(self.feet):
            error = data.xpos[foot, 2] - self.foot_z[index, side]
            if self.swing[index, side] and error < 0:
                mujoco.mj_jac(model, data, jacobian, None, data.xpos[foot], foot)
                errors.append(np.sqrt(self.lift_weight) * error)
                rows.append(np.sqrt(self.lift_weight) * jacobian[2].copy())
            if self.foot_xy_weight:
                # Match the unchanged world-space transition reference, including
                # landing placement. No measured endpoint or source offset repair.
                mujoco.mj_jac(model, data, jacobian, None, data.xpos[foot], foot)
                weight = np.sqrt(self.foot_xy_weight)
                errors.extend(weight * (data.xpos[foot, :2] - self.foot_xy[index, side]))
                rows.extend(weight * jacobian[:2].copy())
        errors, rows = np.asarray(errors), np.asarray(rows)
        gradient[:29] = rows.T @ errors
        hessian[:29, :29] = rows.T @ rows
        return float(0.5 * errors @ errors), gradient, hessian

    def augment(self, local, trajectory):
        result = {**local, "gradient": local["gradient"].copy(), "hessian": local["hessian"].copy()}
        indices = np.flatnonzero(self.active[:-1])
        extra_g, extra_h, cost = [], [], 0.0
        for index in indices:
            value, gradient, hessian = self.state_terms(trajectory["qpos"][index], index)
            result["gradient"][index] += gradient
            result["hessian"][index] += hessian
            extra_g.append(gradient)
            extra_h.append(hessian)
            cost += value
        evidence = dict(indices=indices, gradient=np.asarray(extra_g), hessian=np.asarray(extra_h))
        contract = dict(
            kind="g1_true23_original_reference_local_transition_guidance_v1",
            com_xy_weight=self.com_weight,
            underlift_weight=self.lift_weight,
            original_reference_air_selection_height_m=0.005,
            phases=self.phases,
            guided_state_nodes=len(indices),
            requested_air_nodes_per_foot=np.sum(self.swing, axis=0).tolist(),
            local_guide_cost_at_cached_nominal=cost,
            original_acceptance_objective_and_thresholds_unchanged=True,
            cached_physical_dynamics_unchanged=True,
            no_extra_physical_forces=True,
            not_policy_training_or_hardware_qualification=True,
        )
        if self.foot_xy_weight:
            contract.update(
                kind="g1_true23_original_reference_local_transition_placement_guidance_v2",
                foot_horizontal_weight=self.foot_xy_weight,
                foot_horizontal_target="unchanged_original_transition_native_ankle_world_xy",
                foot_horizontal_guided_nodes_per_foot=len(indices),
                measured_foot_offsets_not_subtracted_from_reference=True,
                source_motion_and_standing_nodes_not_given_placement_terms=True,
            )
        return result, evidence, contract
