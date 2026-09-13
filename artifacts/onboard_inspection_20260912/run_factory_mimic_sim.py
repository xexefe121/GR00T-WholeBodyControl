"""Simulation-only native23 factory mimic reproduction; no DDS or robot network.

Config and FsmMimicTest object code establish 76 observations (angular velocity,
gravity, q-default, dq, previous scaled action, phase), five past/current frames,
zero ankle-velocity observations, and q targets = default + clipped scaled action.
Source: public G1 Edu+ 1.4.5 / ai_sport 8.4.2.222, not installed-PC1 provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np
import yaml

FIRMWARE = Path(r"E:\codex-artifacts\sonic23_teleop_resume_20260911\onboard_factory_firmware_v1")
sys.path.insert(0, str(FIRMWARE / "python_deps"))
import MNN

REPO = Path(__file__).resolve().parents[2]
BUNDLE = REPO / "artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1"
ASSETS = FIRMWARE / "ai_sport_files/ai_sport_8.4.2.222/module/ai_sport/file/unitree/module/ai_sport"
CONFIG = FIRMWARE / "decoded_configs/policies/mimic_test/fsm_mimic_test.yaml"
SDK_IDS = list(range(13)) + list(range(15, 20)) + list(range(22, 27))


class FactoryMimic23:
    def __init__(self,onnx_path=None):
        c = self.cfg = yaml.safe_load(CONFIG.read_text())
        assert c["num_total_joints"] == 23 and c["joint_indices"] == SDK_IDS
        self.model_path = ASSETS / c["policy_file"].removeprefix("../")
        self.net = MNN.Interpreter(str(self.model_path))
        self.session = self.net.createSession({"numThread": 1, "backend": "CPU"})
        self.tensor = self.net.getSessionInput(self.session, "obs")
        assert self.tensor.getShape() == (1, 5, 76)
        self.default = np.asarray(c["default_joint_q"], np.float32)
        self.scale = np.asarray(c["action_scale"], np.float32) * c["action_scale_coeff"]
        self.kp = np.asarray(c["joint_kp"])
        self.kd = np.asarray(c["joint_kd"])
        self.history = np.zeros((5, 76), np.float32)
        self.last_scaled = np.zeros(23, np.float32)
        self.initialized = False
        self.onnx_path=onnx_path
        self.onnx_session=None
        if onnx_path is not None:
            import onnxruntime as ort
            opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
            self.onnx_session=ort.InferenceSession(str(onnx_path),sess_options=opts,providers=['CPUExecutionProvider'])

    def step(self, q, dq, angular_velocity, projected_gravity, phase):
        c = self.cfg
        dq = np.asarray(dq, np.float32).copy()
        # FsmMimicTest::formulateCurrentObservation zeros these four entries.
        dq[[4, 5, 10, 11]] = 0
        obs = np.concatenate((
            angular_velocity * c["observation_scale_ang_vel"],
            projected_gravity * c["observation_scale_proj_grav"],
            (q - self.default) * c["observation_scale_dof_pos"],
            dq * c["observation_scale_dof_vel"],
            self.last_scaled * c["observation_scale_actions"],
            [phase * c["observation_scale_phase"]],
        )).astype(np.float32)
        obs = np.clip(obs, -c["observation_clip"], c["observation_clip"])
        if not self.initialized:
            self.history[:] = obs
            self.initialized = True
        else:
            self.history[:-1] = self.history[1:]
            self.history[-1] = obs
        if self.onnx_session is not None:
            result=self.onnx_session.run(['act'],{'obs':self.history[None]})[0][0]
        else:
            value = MNN.Tensor((1, 5, 76), MNN.Halide_Type_Float, self.history.reshape(-1), MNN.Tensor_DimensionType_Caffe)
            self.tensor.copyFrom(value)
            self.net.runSession(self.session)
            result = np.asarray(self.net.getSessionOutput(self.session, "act").getData(), np.float32)
        if result.shape != (23,) or not np.isfinite(result).all():
            raise ValueError("invalid factory policy output")
        self.last_scaled = np.clip(result, -c["action_clip"], c["action_clip"]) * self.scale
        return self.default + self.last_scaled


def load_model():
    if mujoco.__version__ != "3.2.3":
        raise ValueError("Use native MuJoCo 3.2.3 referee")
    model = mujoco.MjModel.from_xml_path(str(BUNDLE / "native_prepared.xml"))
    with np.load(BUNDLE / "prepared_model_arrays.npz", allow_pickle=False) as z:
        for key in z.files:
            getattr(model, key)[:] = z[key]
    mujoco.mj_setConst(model, mujoco.MjData(model))
    contract = json.loads((BUNDLE / "contract.json").read_text())
    assert model.nq == 30 and model.nv == 29 and model.nu == 23
    joints = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(1, 24)]
    assert joints == contract["joint_names"]
    return model, contract


def render(model, trace, target):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    d = mujoco.MjData(model)
    model.vis.global_.offwidth = 640
    model.vis.global_.offheight = 480
    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.distance = 3.0
    camera.azimuth = 135
    camera.elevation = -12
    try:
        with imageio.get_writer(str(target), fps=25, codec="libx264", quality=7) as writer:
            for index, q in enumerate(trace[::2]):  # trace at 50 Hz
                d.qpos[:] = q
                mujoco.mj_forward(model, d)
                camera.lookat[:] = [q[0], q[1], .8]
                renderer.update_scene(d, camera=camera)
                frame = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(frame)
                seconds = index / 25
                stage = "INITIAL STAND" if seconds < 3 else ("FACTORY DANCE" if seconds < 12.35 else "STANDING HOLD")
                draw.rectangle((0, 0, 640, 48), fill=(16, 22, 28))
                draw.text((12, 8), f"Native23 factory controller | {stage} | {seconds:05.2f} s | 1x", fill="white")
                draw.text((12, 28), "Prepared factory motion. Simulation only. Not live Pico teleoperation.", fill=(195, 205, 218))
                writer.append_data(np.asarray(frame))
    finally:
        renderer.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=FIRMWARE / "native23_mimic_v1")
    ap.add_argument("--stand-only", action="store_true")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--initial-vx", type=float, default=0)
    ap.add_argument('--onnx',type=Path,help='Exact trainable reconstruction export')
    ap.add_argument('--initial-benchmark',action='store_true',help='Use fixed existing benchmark initial pose')
    ap.add_argument("--joint-margin", type=float, default=0.0,
                    help="Uniform inward target margin in radians; no state projection")
    ap.add_argument("--limit-brake", action="store_true",
                    help="Generic PD restoring torque inside a 0.10-rad limit band")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model, c = load_model()
    policy = FactoryMimic23(args.onnx)
    data = mujoco.MjData(model)
    data.qpos[:] = c["initial_qpos"]
    if args.initial_benchmark:
        with np.load(FIRMWARE.parent/'causal_dynamics_v1/bank/walk003.npz',allow_pickle=False) as initial:
            data.qpos[:]=initial['states'][10,:30]
            data.qvel[:]=initial['states'][10,30:]
    else:data.qpos[7:] = policy.default
    mujoco.mj_forward(model, data)
    foot_geoms = [i for i in range(model.ngeom) if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE and model.geom_contype[i]]
    assert len(foot_geoms) == 8
    lowest = min(data.geom_xpos[i, 2] - model.geom_size[i, 0] for i in foot_geoms)
    if not args.initial_benchmark:data.qpos[2] += .001 - lowest
    data.qvel[0] = args.initial_vx
    mujoco.mj_forward(model, data)
    initial_qpos = data.qpos.copy()
    limits = np.asarray(c["joint_limits"])
    if args.joint_margin < 0 or np.any(2 * args.joint_margin >= np.diff(limits, axis=1)):
        raise ValueError("invalid inward target margin")
    efforts = np.asarray(c["native_effort"])
    speeds = np.asarray(c["native_velocity"])
    pre = 3.0
    motion = 0.0 if args.stand_only else policy.cfg["mimic_trajectory_duration"]
    duration = pre + motion + 30.0
    controls = int(np.ceil(duration / .02))
    target = policy.default.copy()
    records, inference_ms = [], []
    max_speed = max_effort = max_bound = max_target_clip = max_tilt = 0.0
    min_height = float(data.qpos[2])
    first_rejection = None
    termination = "completed"
    begin = time.perf_counter()
    for step in range(controls * 10):
        if step % 10 == 0:
            phase = 0.0 if args.stand_only else np.clip((data.time - pre + .02) / motion, 0, 1)
            # MuJoCo free-joint angular velocity is local; gravity R^T*[0,0,-1].
            quat = data.qpos[3:7]
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, quat)
            gravity = rotation.reshape(3, 3).T @ np.array([0., 0., -1.])
            start = time.perf_counter_ns()
            raw_target = policy.step(data.qpos[7:], data.qvel[6:], data.qvel[3:6], gravity, phase)
            inference_ms.append((time.perf_counter_ns() - start) / 1e6)
            target = np.clip(raw_target, limits[:, 0] + args.joint_margin, limits[:, 1] - args.joint_margin)
            max_target_clip = max(max_target_clip, float(np.max(np.abs(target - raw_target))))
            records.append((data.time, phase, data.qpos.copy(), data.qvel.copy(), target.copy()))
        torque = policy.kp * (target - data.qpos[7:]) - policy.kd * data.qvel[6:]
        if args.limit_brake:
            band = np.minimum(.10, np.diff(limits, axis=1).ravel() * .2)
            penetration = data.qpos[7:] - np.clip(data.qpos[7:], limits[:, 0] + band, limits[:, 1] - band)
            outward_velocity = np.where(penetration * data.qvel[6:] > 0, data.qvel[6:], 0)
            torque -= 100 * penetration + 2 * outward_velocity
        data.ctrl[:] = np.clip(torque, -efforts, efforts)
        mujoco.mj_step(model, data)
        tilt = float(np.arccos(np.clip(1 - 2 * np.sum(data.qpos[4:6] ** 2), -1, 1)))
        speed = float(np.max(np.abs(data.qvel[6:]) / speeds))
        effort = float(np.max(np.abs(data.qfrc_actuator[6:]) / efforts))
        bound = float(np.maximum(0, np.maximum(limits[:, 0] - data.qpos[7:], data.qpos[7:] - limits[:, 1])).max())
        min_height = min(min_height, float(data.qpos[2]))
        max_speed, max_effort, max_bound, max_tilt = max(max_speed, speed), max(max_effort, effort), max(max_bound, bound), max(max_tilt, tilt)
        reasons = []
        if bound > 1e-6: reasons.append("native_joint_bound")
        if speed > 1: reasons.append("native_joint_speed")
        if effort > 1 + 1e-9: reasons.append("native_effort")
        if data.qpos[2] < .25 or tilt > 1.2: reasons.append("fall")
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all(): reasons.append("nonfinite")
        if np.any(data.warning.number): reasons.append("engine_warning")
        if reasons and first_rejection is None:
            first_rejection = {"time": float(data.time), "reasons": reasons}
        # Continue recording bounded diagnostics past a joint-limit rejection;
        # do not hide it, clamp state, or certify such a rollout as a pass.
        if any(x in reasons for x in ("fall", "nonfinite", "engine_warning")) or speed > 3:
            termination = ",".join(reasons)
            records.append((data.time, phase, data.qpos.copy(), data.qvel.copy(), target.copy()))
            break
    wall = time.perf_counter() - begin
    trace = {"time": np.array([r[0] for r in records]), "phase": np.array([r[1] for r in records]), "qpos": np.array([r[2] for r in records]), "qvel": np.array([r[3] for r in records]), "target": np.array([r[4] for r in records])}
    np.savez_compressed(args.output / "trace.npz", **trace)
    quiet_mask = trace["time"] >= data.time - 3
    q, v = trace["qpos"][quiet_mask], trace["qvel"][quiet_mask]
    quiet = {"xy_span": float(np.linalg.norm(np.ptp(q[:, :2], axis=0))), "root_speed_max": float(np.linalg.norm(v[:, :3], axis=1).max()), "joint_speed_p95": float(np.quantile(np.abs(v[:, 6:]), .95)), "joint_speed_max": float(np.abs(v[:, 6:]).max())}
    report = {"source": "public G1 Edu+ 1.4.5; ai_sport 8.4.2.222; installed PC1 version unverified", "model": str(policy.model_path), "model_sha256": hashlib.sha256(policy.model_path.read_bytes()).hexdigest(), "controller": "factory 23-joint mimic, phase-only prepared motion; not live teleop", "stand_only": args.stand_only, "simulation_seconds": float(data.time), "requested_seconds": controls * .02, "wall_seconds": wall, "termination": termination, "completed": termination == "completed", "strict_physical_pass": first_rejection is None, "first_rejection": first_rejection, "min_height": min_height, "max_tilt": max_tilt, "max_speed_ratio": max_speed, "max_effort_ratio": max_effort, "max_joint_bound_excess": max_bound, "max_target_clip": max_target_clip, "inference_ms_p50_p95_max": np.quantile(inference_ms, [.5, .95, 1]).tolist(), "initial_qpos": initial_qpos.tolist(), "initial_vx": args.initial_vx, "last_3s": quiet, "timing": "unpaced 500 Hz physics / 50 Hz control; independent wall-clock deadline gate not exercised"}
    report["target_joint_margin_rad"] = args.joint_margin
    report["limit_brake"] = args.limit_brake
    report['executed_onnx']=str(args.onnx) if args.onnx else None
    report['executed_onnx_sha256']=hashlib.sha256(args.onnx.read_bytes()).hexdigest() if args.onnx else None
    report['initial_benchmark']=args.initial_benchmark
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    if args.render:
        render(model, trace["qpos"], args.output / "factory_native23.mp4")


if __name__ == "__main__":
    main()
