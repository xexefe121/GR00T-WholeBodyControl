"""Actual-engine bounded-PD trajectory shooting, SIM only and not a policy.

No free contact-force variables, root forces, hardware transport or modified
physics. Optimizer trials are independent simulations; a trial never resets
state after its initial condition. Derivative probes use separate scratch data.

MuJoCo transition/state conventions:
https://mujoco.readthedocs.io/en/3.5.0/APIreference/APIfunctions.html#mjd-transitionfd
https://mujoco.readthedocs.io/en/3.5.0/computation/index.html#reproducibility
"""

import mujoco
import numpy as np

from gear_sonic.utils.g1_23dof_safe_target_transform import safe_target_transform_numpy
from gear_sonic.utils.g1_true23_native_model_actuation import native_model_pd_numpy
from gear_sonic.utils.g1_true23_reference_floor import compiled_model_sha256
from gear_sonic.utils.g1_true23_source_action_codec import source_scaled_precompensation


class PdShootingPlant:
    def __init__(self, model, profile):
        if (model.nq, model.nv, model.nu, model.na, model.ntendon, model.neq) != (30, 29, 23, 0, 0, 0):
            raise ValueError("PD shooting requires unchanged native23 free-root articulation")
        if model.opt.timestep != 0.002 or model.opt.integrator != mujoco.mjtIntegrator.mjINT_EULER:
            raise ValueError("PD shooting requires original 500 Hz Euler physics")
        if profile.decimation != 10 or profile.timestep_s != 0.002:
            raise ValueError("PD shooting requires original 50 Hz held targets")
        self.model, self.profile = model, profile
        self.model_sha = compiled_model_sha256(model)
        self.kp, self.kd, self.effort = [np.asarray(getattr(profile, name)) for name in ("kp", "kd", "effort")]
        # Exact emitted float32 endpoints, not the QP diagnostic's extra 2e-6
        # inward numerical guard. Existing policy traces legitimately touch
        # these ORIGINAL codec endpoints. Neither codec nor safety limits change.
        extreme = np.nextafter(np.float32(10), np.float32(0))
        self.lower, self.upper = [
            safe_target_transform_numpy(source_scaled_precompensation(np.full(23, sign * extreme, np.float32))[0])[
                1
            ].astype(np.float64)
            for sign in (-1, 1)
        ]
        self.state_signature = mujoco.mjtState.mjSTATE_INTEGRATION
        self.state_size = mujoco.mj_stateSize(model, self.state_signature)

    def assert_unchanged(self):
        if compiled_model_sha256(self.model) != self.model_sha:
            raise RuntimeError("PD shooting changed physical model")

    def initial_data(self, qpos, qvel):
        qpos, qvel = np.asarray(qpos), np.asarray(qvel)
        if qpos.shape != (30,) or qvel.shape != (29,) or not np.isfinite(np.r_[qpos, qvel]).all():
            raise ValueError("PD shooting initial state must be finite native23")
        data = mujoco.MjData(self.model)
        data.qpos[:], data.qvel[:] = qpos, qvel
        mujoco.mj_forward(self.model, data)  # Same once-only initial reset as the referee.
        return data

    def state(self, data):
        state = np.empty(self.state_size)
        mujoco.mj_getState(self.model, data, state, self.state_signature)
        return state

    def scratch(self, state):
        state = np.asarray(state)
        if state.shape != (self.state_size,) or not np.isfinite(state).all():
            raise ValueError("PD shooting requires complete finite integration state including warmstart")
        data = mujoco.MjData(self.model)
        mujoco.mj_setState(self.model, data, state, self.state_signature)
        return data

    def validate_target(self, target):
        target = np.asarray(target, dtype=np.float64)
        if (
            target.shape != (23,)
            or not np.isfinite(target).all()
            or np.any(target < self.lower)
            or np.any(target > self.upper)
        ):
            raise ValueError("PD shooting target exceeds unchanged codec envelope")
        return target

    def _tick(self, data, target):
        requested, applied, invalid, _ = native_model_pd_numpy(target, data.qpos[7:], data.qvel[6:], self.profile)
        if invalid:
            raise ValueError("PD shooting generated invalid torque")
        if np.any(data.qfrc_applied) or np.any(data.xfrc_applied):
            raise ValueError("nominal PD shooting cannot inject external forces")
        data.ctrl[:] = applied
        mujoco.mj_step(self.model, data)
        if not np.isfinite(np.r_[data.qpos, data.qvel]).all() or any(w.number for w in data.warning):
            raise RuntimeError("PD shooting numerical integration failed")
        np.testing.assert_array_equal(applied, data.qfrc_actuator[6:])
        return requested, applied

    def integrate_control(self, data, target, records=None):
        target = self.validate_target(target)
        # Match the referee's read-only observation refresh at control boundaries.
        mujoco.mj_kinematics(self.model, data)
        mujoco.mj_comPos(self.model, data)
        mujoco.mj_comVel(self.model, data)
        for _ in range(10):
            if records is not None:
                records["physics_pre_qpos"].append(data.qpos.copy())
                records["physics_pre_qvel"].append(data.qvel.copy())
                start = float(data.time)
            requested, applied = self._tick(data, target)
            if records is not None:
                for key, value in (
                    ("physics_post_qpos", data.qpos.copy()),
                    ("physics_post_qvel", data.qvel.copy()),
                    ("physics_time", (start, float(data.time))),
                    ("requested_torque23", requested.copy()),
                    ("applied_torque23", applied.copy()),
                    ("engine_actuator_force23", data.qfrc_actuator[6:].copy()),
                    ("physics_contact_count", int(data.ncon)),
                    ("torque_saturated23", np.abs(requested) > self.effort),
                ):
                    records[key].append(value)

    def rollout(self, initial_state, targets, *, physics_records=False):
        targets = np.asarray(targets)
        if targets.ndim != 2 or targets.shape[1] != 23 or not len(targets):
            raise ValueError("PD shooting requires a nonempty full target sequence")
        data = self.scratch(initial_state)
        records = (
            {
                key: []
                for key in (
                    "physics_pre_qpos",
                    "physics_pre_qvel",
                    "physics_post_qpos",
                    "physics_post_qvel",
                    "physics_time",
                    "requested_torque23",
                    "applied_torque23",
                    "engine_actuator_force23",
                    "physics_contact_count",
                    "torque_saturated23",
                )
            }
            if physics_records
            else None
        )
        states, positions, velocities = [self.state(data)], [data.qpos.copy()], [data.qvel.copy()]
        for target in targets:
            self.integrate_control(data, target, records)
            states.append(self.state(data))
            positions.append(data.qpos.copy())
            velocities.append(data.qvel.copy())
        self.assert_unchanged()
        return {
            "integration_state": np.asarray(states),
            "qpos": np.asarray(positions),
            "qvel": np.asarray(velocities),
            **({key: np.asarray(value) for key, value in records.items()} if records is not None else {}),
        }

    def linearize_control(self, initial_state, target, *, epsilon=1e-6):
        """Chain ten actual engine transitions, including PD and saturation.

        Dynamics derivatives are numerical approximations, not an execution
        certificate. Original solver settings stay unchanged. Scratch data is
        restored after each derivative call, including warmstart acceleration.
        """
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("PD shooting derivative epsilon must be positive")
        data, target = self.scratch(initial_state), self.validate_target(target)
        total_a, total_b = np.eye(58), np.zeros((58, 23))
        torque_a = np.zeros((23, 58))
        torque_a[:, 6:29], torque_a[:, 35:58] = -np.diag(self.kp), -np.diag(self.kd)
        for _ in range(10):
            before = self.state(data)
            requested, applied, invalid, _ = native_model_pd_numpy(
                target, data.qpos[7:], data.qvel[6:], self.profile
            )
            if invalid:
                raise ValueError("PD shooting derivative state invalid")
            data.ctrl[:] = applied
            a, b = np.empty((58, 58)), np.empty((58, 23))
            mujoco.mjd_transitionFD(self.model, data, epsilon, 1, a, b, None, None)
            active = (np.abs(requested) < self.effort).astype(float)
            local_a = a + b @ (active[:, None] * torque_a)
            local_b = b * (active * self.kp)[None, :]
            total_a, total_b = local_a @ total_a, local_a @ total_b + local_b
            mujoco.mj_setState(self.model, data, before, self.state_signature)
            self._tick(data, target)
        if not np.isfinite(total_a).all() or not np.isfinite(total_b).all():
            raise RuntimeError("PD shooting derivative nonfinite")
        self.assert_unchanged()
        return total_a, total_b


def state_difference(model, reference_qpos, reference_qvel, qpos, qvel):
    # Quaternion self-differentiation can leave a tiny nonzero rotation. Preserve
    # exact identity so a zero optimization step cannot move an unchanged rollout.
    # Use exact equality, not a tolerance that would erase real small deviations.
    position = np.zeros(model.nv)
    if not np.array_equal(reference_qpos, qpos):
        mujoco.mj_differentiatePos(model, position, 1.0, reference_qpos, qpos)
    return np.r_[position, np.asarray(qvel) - reference_qvel]
