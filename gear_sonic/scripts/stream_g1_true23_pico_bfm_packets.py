"""Bridge causal PICO/SOMA body poses into BFM-Zero teleoperation packets.

This publisher is deliberately read-only with respect to the robot.  The live
input is the existing ``stream_g1_23dof_pico_causal_zmq`` publisher started
with ``--native23-body-packets``; that process remains the sole owner of the
PICO capture worker and pinned SOMA solver.  This process only converts its
current full-body extension to BFM's 23-DoF packet contract, applies the
declared causal floor transform, and publishes ZMQ JSON for
``run_g1_true23_bfm_teleop_sim consume``.

``--recorded-motion`` is a hardware-free source for the same bridge.  It is a
recorded PICO-derived native23 motion, not an alternate retargeter.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time
from typing import Any, Iterable

import mujoco
import numpy as np
from scipy.linalg import expm
from scipy.spatial.transform import Rotation
import zmq

from gear_sonic.scripts.run_g1_true23_bfm_teleop_sim import json_packet
from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES
from gear_sonic.utils.g1_23dof_pico_retargeted_producer import SOMA_MJ29_JOINT_NAMES
from gear_sonic.utils.g1_true23_bfmzero_inference import load_contract
from gear_sonic.utils.g1_true23_bfmzero_stream import DT, Packet, PacketGate, validate_fields
from gear_sonic.utils.g1_true23_pico_body_adapter import Native23PicoBodyAdapter

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = ROOT / "artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1/contract.json"
DEFAULT_BFM_CONTRACT = ROOT / "artifacts/g1_true23_six_hour_replan_20260910_v1/bfmzero_inspect_v1/config.yaml"
DEFAULT_MODEL = ROOT / "gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml"

FLOOR_OMEGA_PER_SECOND = 15.0
FLOOR_BUFFER_M = 0.020
FLOOR_GUARD_M = 0.000001


def _percentiles_ns(values: list[int]) -> dict[str, float | None]:
    if not values:
        return {"p50_ms": None, "p95_ms": None, "max_ms": None}
    array = np.asarray(values, dtype=np.float64) / 1e6
    return {"p50_ms": float(np.percentile(array, 50)), "p95_ms": float(np.percentile(array, 95)),
            "max_ms": float(np.max(array))}


class NativeFootClearance:
    """Exact native-model foot-sphere clearance for a current 23-DoF pose."""

    def __init__(self, model_path: Path):
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        if self.model.nq != 30 or self.model.nbody != 25:
            raise ValueError("floor model is not the native 23-DoF free-base model")
        self.data = mujoco.MjData(self.model)
        floor = self.model.geom("floor").id
        if self.model.geom_type[floor] != mujoco.mjtGeom.mjGEOM_PLANE:
            raise ValueError("native floor geometry is not a plane")
        self.floor = floor
        foot_bodies = [self.model.body(f"{side}_ankle_roll_link").id for side in ("left", "right")]
        self.foot_geoms = [
            [
                geom for geom in range(self.model.ngeom)
                if self.model.geom_bodyid[geom] == body
                and self.model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_SPHERE
                and ((self.model.geom_contype[geom] & self.model.geom_conaffinity[floor])
                     or (self.model.geom_contype[floor] & self.model.geom_conaffinity[geom]))
            ]
            for body in foot_bodies
        ]
        if not all(len(group) == 4 for group in self.foot_geoms):
            raise ValueError("native model does not expose foot collision geometry")

    def minimum(self, fields: dict[str, np.ndarray]) -> float:
        self.data.qpos[:] = np.r_[fields["body_pos_w"][0], fields["body_quat_w"][0], fields["joint_pos"]]
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        floor_origin = self.data.geom_xpos[self.floor]
        normal = self.data.geom_xmat[self.floor].reshape(3, 3)[:, 2]
        clearances: list[float] = []
        for group in self.foot_geoms:
            for geom in group:
                clearances.append(float((self.data.geom_xpos[geom] - floor_origin) @ normal - self.model.geom_size[geom, 0]))
        return min(clearances)


class CausalFloorLift:
    """The v4 critically damped 50-Hz floor correction, with no preview."""

    def __init__(self, clearance: NativeFootClearance):
        continuous = np.array([[0.0, 1.0], [-FLOOR_OMEGA_PER_SECOND**2, -2.0 * FLOOR_OMEGA_PER_SECOND]])
        self.transition = expm(continuous * DT)
        self.control = np.linalg.solve(
            continuous, (self.transition - np.eye(2)) @ np.array([0.0, FLOOR_OMEGA_PER_SECOND**2])
        )
        self.clearance = clearance
        self.state: np.ndarray | None = None

    def apply(self, fields: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        before = self.clearance.minimum(fields)
        raw_required = max(0.0, -before)
        if self.state is None:
            self.state = np.array([raw_required, 0.0])
        self.state = self.transition @ self.state + self.control * raw_required
        filtered_required = float(self.state[0])
        lift = max(raw_required + FLOOR_GUARD_M, filtered_required + FLOOR_BUFFER_M)
        result = {name: np.asarray(value, dtype=np.float64).copy() for name, value in fields.items()}
        result["body_pos_w"][:, 2] += lift
        after = self.clearance.minimum(result)
        if after < FLOOR_GUARD_M - 1e-10:
            raise RuntimeError(f"causal floor correction did not clear the floor: {after}")
        return result, {"before_m": before, "after_m": after, "raw_required_lift_m": raw_required,
                         "filtered_required_lift_m": filtered_required, "applied_lift_m": lift}


class BFMPicoPacketBridge:
    """Adapts one live upstream frame at a time and re-derives causal velocities."""

    def __init__(self, contract: dict[str, Any], floor: CausalFloorLift, *, epoch: int = 0):
        self.adapter = Native23PicoBodyAdapter(contract, epoch=epoch)
        self.floor = floor
        self.previous: dict[str, np.ndarray] | None = None
        self.epoch = epoch

    def rearm(self, epoch: int) -> None:
        self.adapter.rearm(epoch)
        self.previous = None
        self.epoch = epoch

    def adapt(self, wire: dict[str, Any], *, received_monotonic_ns: int, final: bool = False) -> tuple[Packet, dict[str, Any]]:
        adapted = self.adapter.adapt(wire, received_monotonic_ns=received_monotonic_ns)
        corrected, floor = self.floor.apply(adapted.packet.fields)
        previous = self.previous
        if previous is None:
            corrected["joint_vel"] = np.zeros(23, dtype=np.float64)
            corrected["body_lin_vel_w"] = np.zeros((24, 3), dtype=np.float64)
            corrected["body_ang_vel_w"] = np.zeros((24, 3), dtype=np.float64)
        else:
            corrected["joint_vel"] = (corrected["joint_pos"] - previous["joint_pos"]) / DT
            corrected["body_lin_vel_w"] = (corrected["body_pos_w"] - previous["body_pos_w"]) / DT
            current = Rotation.from_quat(corrected["body_quat_w"][:, [1, 2, 3, 0]])
            prior = Rotation.from_quat(previous["body_quat_w"][:, [1, 2, 3, 0]])
            corrected["body_ang_vel_w"] = (current * prior.inv()).as_rotvec() / DT
        # The adapter has already normalized source rotations.  Keep the BFM
        # tolerance explicit at the packet boundary rather than trusting JSON.
        corrected["body_quat_w"] /= np.linalg.norm(corrected["body_quat_w"], axis=1)[:, None]
        validate_fields(corrected)
        packet = Packet(adapted.packet.epoch, adapted.packet.sequence, adapted.packet.source_time, corrected, final)
        self.previous = {name: value.copy() for name, value in corrected.items()}
        source = dict(adapted.source, floor=floor,
                      derivative="current_minus_previous_corrected_pose_over_20ms",
                      first_derivative_unavailable=previous is None)
        return packet, source


def _recorded_wires(path: Path, contract: dict[str, Any], *, start_frame: int = 0,
                    stop_frame: int | None = None) -> Iterable[dict[str, Any]]:
    """Expose a saved PICO-derived native23 motion as current-body source wires."""
    with np.load(path, allow_pickle=False) as archive:
        required = {"joint_pos", "body_pos_w", "body_quat_w"}
        if not required <= set(archive.files):
            raise ValueError("recorded motion lacks native23 position/body fields")
        q = np.asarray(archive["joint_pos"], dtype=np.float64)
        pos = np.asarray(archive["body_pos_w"], dtype=np.float64)
        quat = np.asarray(archive["body_quat_w"], dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 23 or pos.shape != (len(q), 24, 3) or quat.shape != (len(q), 24, 4):
        raise ValueError("recorded motion is not a 50-Hz native23 body reference")
    if not np.isfinite(q).all() or not np.isfinite(pos).all() or not np.isfinite(quat).all():
        raise ValueError("recorded motion contains non-finite values")
    if np.max(np.abs(np.linalg.norm(quat, axis=-1) - 1.0)) > 1e-5:
        raise ValueError("recorded motion contains non-unit body quaternions")
    if start_frame < 0 or start_frame >= len(q):
        raise ValueError("recorded-start-frame is outside the recorded motion")
    stop = len(q) if stop_frame is None else stop_frame
    if stop <= start_frame or stop > len(q):
        raise ValueError("recorded-stop-frame must be inside the recorded motion and after start")
    names = list(contract["body_names"])
    hardware_ids = [SOMA_MJ29_JOINT_NAMES.index(name) for name in HARDWARE_23_JOINT_NAMES]
    start_ns = time.monotonic_ns()
    for sequence, index in enumerate(range(start_frame, stop)):
        all_q = np.zeros(29, dtype=np.float64)
        all_q[hardware_ids] = q[index]
        body_xyzw = quat[index][:, [1, 2, 3, 0]]
        task_ids = [names.index(name) for name in ("left_wrist_roll_rubber_hand", "right_wrist_roll_rubber_hand", "torso_link")]
        stamp = start_ns + sequence * int(DT * 1e9)
        body = {
            "schema_version": 1, "kind": "native23_current_original29_body_pose", "original29_axes_preserved": True,
            "source_frame_index": sequence, "reference_monotonic_ns": stamp, "capture_monotonic_ns": stamp,
            "joint_names": list(SOMA_MJ29_JOINT_NAMES), "joint_position": all_q.tolist(), "body_names": names,
            "body_position_w": pos[index].tolist(), "body_quaternion_xyzw": body_xyzw.tolist(),
            "task_names": ["left_hand", "right_hand", "head"],
            "task_position_w": pos[index, task_ids].tolist(), "task_quaternion_xyzw": body_xyzw[task_ids].tolist(),
        }
        yield {"control_source_frame_index": sequence, "control_monotonic_ns": stamp, "native23_body_pose": body}


def _live_wires(endpoint: str, *, timeout_seconds: float) -> Iterable[dict[str, Any]]:
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(endpoint)
    try:
        while True:
            if not socket.poll(int(timeout_seconds * 1000)):
                raise TimeoutError("timed out waiting for the causal PICO upstream stream")
            wire = socket.recv_json()
            if not isinstance(wire, dict):
                raise ValueError("causal PICO upstream packet is not an object")
            yield wire
    finally:
        socket.close()
        context.term()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if (args.recorded_motion is None) == (args.upstream_endpoint is None):
        raise ValueError("choose exactly one of --recorded-motion or --upstream-endpoint")
    if args.max_packets is not None and args.max_packets <= 0:
        raise ValueError("max-packets must be positive")
    contract = json.loads(args.contract.read_text())
    floor = CausalFloorLift(NativeFootClearance(args.model))
    bridge = BFMPicoPacketBridge(contract, floor, epoch=args.epoch)
    source = (_recorded_wires(args.recorded_motion, contract, start_frame=args.recorded_start_frame,
                              stop_frame=args.recorded_stop_frame) if args.recorded_motion is not None
              else _live_wires(args.upstream_endpoint, timeout_seconds=args.input_timeout_seconds))
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(args.bind)
    if args.subscriber_warmup_s:
        time.sleep(args.subscriber_warmup_s)
    published = failed = 0
    reasons: Counter[str] = Counter()
    latency_ns: list[int] = []
    before_clearance: list[float] = []
    after_clearance: list[float] = []
    admission = Counter()
    bfm_contract = load_contract(args.bfm_contract) if args.gate_audit else None
    gate = PacketGate(bfm_contract, now=0.0) if bfm_contract is not None else None
    recorded_clock_start: float | None = None
    try:
        wires = iter(source)
        buffered: dict[str, Any] | None = None
        while True:
            if buffered is None:
                try:
                    wire = next(wires)
                except StopIteration:
                    break
            else:
                wire, buffered = buffered, None
            if args.recorded_motion is not None:
                capture_ns = int(wire["native23_body_pose"]["capture_monotonic_ns"])
                if recorded_clock_start is None:
                    recorded_clock_start = capture_ns / 1e9
                deadline = recorded_clock_start + published * DT
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        break
                    time.sleep(min(remaining, 0.002))
            final = False
            if args.recorded_motion is not None:
                try:
                    buffered = next(wires)
                except StopIteration:
                    final = True
            received_ns = time.monotonic_ns()
            try:
                packet, source_info = bridge.adapt(wire, received_monotonic_ns=received_ns, final=final)
                validate_fields(packet.fields)
                if args.gate_audit:
                    assert gate is not None
                    accepted = gate.receive(packet, packet.source_time)
                    admission["admitted" if accepted else "rejected"] += 1
                    if not accepted:
                        reason = None if gate.fault is None else gate.fault["reason"]
                        raise ValueError("PacketGate.receive: " + str(reason))
                socket.send_json(json_packet(packet))
                published_ns = time.monotonic_ns()
                capture_ns = int(source_info["capture_monotonic_ns"])
                if published_ns < capture_ns:
                    raise RuntimeError("publication timestamp precedes capture timestamp")
                latency_ns.append(published_ns - capture_ns)
                before_clearance.append(float(source_info["floor"]["before_m"]))
                after_clearance.append(float(source_info["floor"]["after_m"]))
                published += 1
                print(json.dumps({"sequence": packet.sequence, "epoch": packet.epoch,
                                  "capture_to_publication_ms": latency_ns[-1] / 1e6,
                                  "floor_after_m": after_clearance[-1]}), flush=True)
            except (ValueError, RuntimeError, KeyError, TypeError, IndexError) as exc:
                failed += 1
                reasons[str(exc)] += 1
                raise RuntimeError(f"bridge refuses to publish invalid packet: {exc}") from exc
            if args.max_packets is not None and published >= args.max_packets:
                break
    finally:
        socket.close()
        context.term()
    report = {
        "kind": "g1_true23_pico_bfm_packet_bridge_v1", "source": "recorded_pico_derived_motion" if args.recorded_motion else "live_pico_causal_upstream",
        "bind": args.bind, "epoch": args.epoch, "packets_published": published,
        "recorded_source_frame_range": None if args.recorded_motion is None else [args.recorded_start_frame, args.recorded_stop_frame],
        "validation_failures_before_publication": failed, "validation_failure_reasons": dict(reasons),
        "capture_to_publication": _percentiles_ns(latency_ns),
        "floor": {"method": "v4_critically_damped_causal_filter", "omega_per_second": FLOOR_OMEGA_PER_SECOND,
                  "fixed_buffer_m": FLOOR_BUFFER_M, "clearance_guard_m": FLOOR_GUARD_M,
                  "minimum_before_m": None if not before_clearance else float(min(before_clearance)),
                  "minimum_after_m": None if not after_clearance else float(min(after_clearance)),
                  "frames_below_floor_after": int(sum(value < 0.0 for value in after_clearance))},
        "packet_gate_audit": None if not args.gate_audit else {
            **dict(admission), "rejected_reasons": {} if gate is None or gate.fault is None else {gate.fault["reason"]: gate.rejected},
        },
        "authorization": {"dds_opened": False, "lowcmd_opened": False, "robot_commands_published": False,
                          "hardware_authorized": False},
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise FileExistsError("bridge report refuses overwrite")
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Causal live-PICO to BFM-Zero packet bridge; no robot control API")
    parser.add_argument("--bind", default="tcp://127.0.0.1:5591")
    parser.add_argument("--upstream-endpoint", help="existing causal PICO publisher with --native23-body-packets")
    parser.add_argument("--recorded-motion", type=Path, help="recorded PICO-derived native23 .npz for no-headset test")
    parser.add_argument("--recorded-start-frame", type=int, default=0)
    parser.add_argument("--recorded-stop-frame", type=int)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--bfm-contract", type=Path, default=DEFAULT_BFM_CONTRACT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--max-packets", type=int)
    parser.add_argument("--subscriber-warmup-s", type=float, default=0.25)
    parser.add_argument("--input-timeout-seconds", type=float, default=2.0)
    parser.add_argument("--gate-audit", action="store_true", help="also pass each output through PacketGate.receive")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    for name in ("contract", "model"):
        value = getattr(args, name)
        if not value.is_file():
            raise FileNotFoundError(f"{name} is missing: {value}")
    if args.gate_audit and not args.bfm_contract.is_file():
        raise FileNotFoundError(f"BFM contract is missing: {args.bfm_contract}")
    print(json.dumps(run(args), indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
