"""Independent saved one-control branch algebra, with no model evaluations.

The caller supplies bound center arrays and copied original GoalFeatures and
state_and_terms functions. No collection helper or producer validator is used.
"""
import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np


KEYS = ("actions", "base_ang_vel", "dof_pos", "dof_vel", "projected_gravity")
CORE_SHA256 = "1efc9cb21e446d8e41e3d3acebe05d12b951e8a0002b264ddf0a6b723e32f45a"


def load_original_difference(path):
    path = Path(path)
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != CORE_SHA256:
        raise ValueError("Original committed-map difference source changed")
    tree = ast.parse(content, filename=str(path))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ("quat_mul", "quat_log")]
    planner = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "Planner")
    functions.append(next(node for node in planner.body
                          if isinstance(node, ast.FunctionDef)
                          and node.name == "difference"))
    scope = {"np": np}
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions,
                 type_ignores=[])), str(path), "exec"), scope)
    robot = SimpleNamespace(nq=30, nv=29, lin_q=np.r_[0:3, 7:30],
                            lin_v=np.r_[0:3, 6:29], free=[(0, 0)])
    return lambda planned, q, v: scope["difference"](
        robot, planned[None], np.r_[q, v][None])[0]


class SavedBranchAlgebra:
    def __init__(self, contract, centers, goals, state_and_terms, difference):
        self.center = centers
        self.goals = goals
        self.state_and_terms = state_and_terms
        self.difference = difference
        self.default, self.kp, self.effort, self.limits = [
            np.asarray(contract[key]) for key in
            ("default_q", "kp", "training_effort", "joint_limits")]
        self.comparisons = 0

    def exact(self, actual, expected, name):
        actual, expected = np.asarray(actual), np.asarray(expected)
        if (actual.shape != expected.shape or actual.dtype != expected.dtype
                or actual.tobytes() != expected.tobytes()):
            raise AssertionError("Saved branch algebra: " + name)
        self.comparisons += 1

    def terms(self, q, v, previous):
        return self.state_and_terms(q[7:], v[6:], q[3:7], v[3:6],
                                    previous, self.default)

    def outgoing(self, center_index, delta):
        center = self.center
        raw_target = center["base_target"][center_index] + delta
        target = np.clip(raw_target, self.limits[:, 0], self.limits[:, 1])
        raw_prior = (center["base_action"][center_index]
                     + delta * self.kp / (.25 * self.effort)).astype(np.float32)
        applied_prior = ((target - self.default) * self.kp /
                         (.25 * self.effort)).astype(np.float32)
        return raw_target, target, raw_prior, applied_prior

    def advanced_history(self, center_index):
        center = self.center
        state, terms = self.terms(center["qpos"][center_index],
                                  center["qvel"][center_index],
                                  center["previous_action"][center_index])
        self.exact(state, center["state"][center_index], "starting state52")
        result = {}
        for key in KEYS:
            value = center["history_" + key][center_index].copy()
            value[1:] = value[:-1].copy()
            value[0] = terms[key]
            result[key] = value
            self.exact(value, center["history_" + key][center_index + 1],
                       "nominal next history " + key)
        flat = np.concatenate([result[key].reshape(-1) for key in KEYS])
        self.exact(flat, center["history"][center_index + 1],
                   "nominal next flattened history")
        return result, flat

    def endpoint(self, center_index, q, v, previous, base_action):
        successor = center_index + 1
        center = self.center
        if (center["dataset"][successor] != center["dataset"][center_index]
                or center["control"][successor] != center["control"][center_index] + 1):
            raise ValueError("Branch successor crossed a dataset boundary")
        state, _ = self.terms(q, v, previous)
        base = self.default + base_action * .25 * self.effort / self.kp
        previous_target = np.clip(self.default + previous * .25 * self.effort /
                                  self.kp, self.limits[:, 0], self.limits[:, 1])
        frame = int(center["control"][center_index]) + 12
        features = np.r_[self.goals(q, v, previous_target, frame),
                         base - self.default, previous].astype(np.float32)
        tangent = self.difference(center["planned_state"][successor], q, v)
        feedback_raw = center["gain"][successor] @ tangent
        feedback = np.clip(feedback_raw, -.1, .1)
        preclip = center["planned_target"][successor] + feedback
        target = np.clip(preclip, self.limits[:, 0], self.limits[:, 1])
        return dict(state=state, base_target=base, features=features,
                    teacher_target=target, residual_rad=target - base,
                    teacher_tangent=tangent,
                    teacher_feedback_raw=feedback_raw,
                    teacher_feedback_correction=feedback,
                    teacher_preclip=preclip,
                    teacher_feedback_clipped=feedback_raw != feedback,
                    teacher_native_clipped=preclip != target,
                    teacher_plan_control=center["plan_control"][successor],
                    teacher_plan_local=center["plan_local"][successor],
                    teacher_replan_boundary=np.bool_(center["plan_local"][successor] == 0),
                    teacher_zero_gain=np.bool_(np.all(center["gain"][successor] == 0)))
