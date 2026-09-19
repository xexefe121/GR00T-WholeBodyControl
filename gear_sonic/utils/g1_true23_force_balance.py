"""Read solved SIM forces without another solve, integration or state edit.

Force components describe one existing contact solution, not counterfactual
forces after changing a controller. See MuJoCo3.5 computation equations:
https://mujoco.readthedocs.io/en/3.5.0/computation/#general-framework
"""

import mujoco
import numpy as np

COMPONENTS = (
    "actuator",
    "passive",
    "negative_bias",
    "external_generalized",
    "left_foot_contact",
    "right_foot_contact",
    "other_contact",
    "joint_limit",
    "other_constraint",
)


def solve_components(mass, forces, acceleration):
    mass, forces, acceleration = (np.asarray(x, dtype=np.float64) for x in (mass, forces, acceleration))
    n = len(acceleration)
    if mass.shape != (n, n) or forces.ndim != 2 or forces.shape[1] != n or acceleration.shape != (n,):
        raise ValueError("force decomposition dimensions differ")
    if not all(np.isfinite(x).all() for x in (mass, forces, acceleration)):
        raise ValueError("force decomposition requires finite values")
    if not np.allclose(mass, mass.T, atol=1e-12, rtol=0):
        raise ValueError("inertia must be symmetric")
    try:
        np.linalg.cholesky(mass)
    except np.linalg.LinAlgError as error:
        raise ValueError("inertia must be positive definite") from error
    inverse = np.linalg.solve(mass, np.eye(n))
    parts = np.linalg.solve(mass, forces.T).T
    return dict(
        acceleration_components=parts,
        inverse_mass_diagonal=np.diag(inverse).copy(),
        force_closure_error=float(np.max(np.abs(mass @ acceleration - forces.sum(0)))),
        acceleration_closure_error=float(np.max(np.abs(parts.sum(0) - acceleration))),
    )


class SolvedForceObserver:
    """Read the most recently solved step; caller supplies its pre/post states."""

    def __init__(self, model):
        self.model = model
        self.floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.feet = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in ("left_ankle_roll_link", "right_ankle_roll_link")
        ]
        if min([self.floor, *self.feet]) < 0:
            raise ValueError("force observation requires named floor and both ankle-roll bodies")
        self.state_spec = mujoco.mjtState.mjSTATE_INTEGRATION

    def integration_state(self, data):
        result = np.empty(mujoco.mj_stateSize(self.model, self.state_spec))
        mujoco.mj_getState(self.model, data, result, self.state_spec)
        return result

    def capture(self, data):
        model = self.model
        state = self.integration_state(data)
        if np.count_nonzero(data.xfrc_applied):
            raise ValueError("spatial external-force mapping is outside this passive observer")
        mass = np.empty((model.nv, model.nv))
        mujoco.mj_fullM(model, mass, data.qM)
        force = np.zeros((len(COMPONENTS), model.nv))
        force[:4] = (data.qfrc_actuator, data.qfrc_passive, -data.qfrc_bias, data.qfrc_applied)
        types = np.asarray(data.efc_type).copy()
        ids = np.asarray(data.efc_id).copy()
        solved = np.asarray(data.efc_force).copy()
        assert types.shape == ids.shape == solved.shape == (data.nefc,)
        contact_types = np.isin(
            types,
            [
                int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS),
                int(mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL),
                int(mujoco.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC),
            ],
        )
        groups = np.full(data.nefc, 8, dtype=np.int64)
        groups[contact_types] = 6
        groups[types == int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT)] = 7
        contacts, foot_loads = [], np.zeros(2)
        for index in range(data.ncon):
            contact = data.contact[index]
            geoms = [int(contact.geom1), int(contact.geom2)]
            bodies = [int(model.geom_bodyid[g]) for g in geoms]
            wrench = np.empty(6)
            mujoco.mj_contactForce(model, data, index, wrench)
            foot = None
            if self.floor in geoms:
                robot_body = bodies[1 - geoms.index(self.floor)]
                if robot_body in self.feet:
                    foot = self.feet.index(robot_body)
                    groups[contact_types & (ids == index)] = 4 + foot
                    foot_loads[foot] += wrench[0]
            contacts.append(
                dict(
                    contact_id=index,
                    foot_index=foot,
                    geom_ids=geoms,
                    geom_names=[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in geoms],
                    body_names=[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in bodies],
                    distance_m=float(contact.dist),
                    position_w=contact.pos.copy().tolist(),
                    contact_frame_rows_world=contact.frame.copy().reshape(3, 3).tolist(),
                    solved_contact_frame_wrench=wrench.tolist(),
                )
            )
        for group in range(4, len(COMPONENTS)):
            masked = np.where(groups == group, solved, 0.0)
            mujoco.mj_mulJacTVec(model, data, force[group], masked)
        result = solve_components(mass, force, data.qacc.copy())
        result.update(
            mass=mass,
            force_components=force,
            forward_acceleration=data.qacc.copy(),
            qfrc_smooth=data.qfrc_smooth.copy(),
            qfrc_constraint=data.qfrc_constraint.copy(),
            self_actuator_acceleration=result["inverse_mass_diagonal"] * data.qfrc_actuator,
            foot_normal_load_n=foot_loads,
            contacts=contacts,
            constraint_partition_error=float(np.max(np.abs(force[4:].sum(0) - data.qfrc_constraint))),
            smooth_partition_error=float(np.max(np.abs(force[:4].sum(0) - data.qfrc_smooth))),
        )
        np.testing.assert_array_equal(self.integration_state(data), state)
        return result
