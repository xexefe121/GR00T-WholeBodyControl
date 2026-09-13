"""CPU Batch/single-MuJoCo parity and throughput on native23 nominal physics.

Run in the isolated mjbatch environment. --plain-only also runs under the
existing MuJoCo environment for an explicit cross-version comparison.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def prepare_native():
    """Same physical assignments as prepare_mujoco_model, with no Torch import."""
    path = ROOT.parent / "GR00T-WholeBodyControl/gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml"
    config_path = ROOT / "gear_sonic/config/sim_validation/g1_23dof_mujoco_sim2sim.json"
    config = json.loads(config_path.read_text())
    model = mujoco.MjModel.from_xml_path(str(path))
    assert (model.nq, model.nv, model.nu) == (30, 29, 23)
    physics = config["physics"]
    assert model.opt.timestep == physics["timestep_s"] == .002
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER
    np.testing.assert_array_equal(model.opt.gravity, physics["gravity_mps2"])
    ids = np.arange(1, model.njnt)
    addresses = model.jnt_dofadr[ids]
    for field, key in (("dof_armature", "armature_hardware"), ("dof_damping", "joint_damping_hardware"),
                       ("dof_frictionloss", "joint_frictionloss_hardware")):
        getattr(model, field)[addresses] = physics[key]
    effort = np.asarray(physics["effort_limit_hardware_nm"])
    model.jnt_actfrclimited[ids] = 1
    model.jnt_actfrcrange[ids] = np.column_stack((-effort, effort))
    mujoco.mj_setConst(model, mujoco.MjData(model))
    return model, config, {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (path, config_path)}


def initial_states(config, count):
    initial = config["initial_state"]
    qpos = np.tile(np.r_[initial["base_position_m"], initial["base_quaternion_wxyz"],
                         initial["joint_position_hardware_rad"]], (count, 1))
    qvel = np.zeros((count, 29))
    rng = np.random.default_rng(20260911)
    qpos[:, 7:] += rng.uniform(-.002, .002, (count, 23))
    qvel[:, 6:] = rng.uniform(-.01, .01, (count, 23))
    return qpos, qvel


def target_sequence(config, count, steps):
    default = np.asarray(config["initial_state"]["joint_position_hardware_rad"])
    knots = np.arange(steps) // 10
    offset = .008 * np.sin(knots[:, None, None] * .08 + np.arange(count)[None, :, None] * .1
                          + np.arange(23)[None, None, :] * .15)
    return default + offset


def run_plain(model, config, count, steps, *, capture=False):
    qpos, qvel = initial_states(config, count)
    data = [mujoco.MjData(model) for _ in range(count)]
    for i, value in enumerate(data):
        value.qpos[:], value.qvel[:] = qpos[i], qvel[i]
        mujoco.mj_forward(model, value)
    physics = config["physics"]
    kp, kd, effort = [np.asarray(physics[key]) for key in ("kp_hardware", "kd_hardware", "effort_limit_hardware_nm")]
    targets = target_sequence(config, count, steps)
    traces = {key: [] for key in ("qpos", "qvel", "ctrl", "qfrc_actuator", "time")}
    started = time.perf_counter()
    for step in range(steps):
        q = np.asarray([value.qpos[7:] for value in data])
        dq = np.asarray([value.qvel[6:] for value in data])
        ctrl = np.clip(kp * (targets[step] - q) - kd * dq, -effort, effort)
        for i, value in enumerate(data):
            value.ctrl[:] = ctrl[i]
            mujoco.mj_step(model, value)
        if capture:
            for key in traces:
                traces[key].append(np.asarray([np.array(getattr(value, key), copy=True) for value in data]))
    elapsed = time.perf_counter() - started
    return elapsed, {key: np.asarray(value) for key, value in traces.items()} if capture else None


def run_batch(model, config, count, steps, threads, *, capture=False):
    from mjbatch import Batch
    batch = Batch(model, count, num_threads=threads)
    fields = {key: batch.bind(key) for key in ("qpos", "qvel", "ctrl")}
    if capture:
        fields.update({key: batch.bind(key) for key in ("qfrc_actuator", "time")})
    fields["qpos"][:], fields["qvel"][:] = initial_states(config, count)
    batch.forward()
    physics = config["physics"]
    kp, kd, effort = [np.asarray(physics[key]) for key in ("kp_hardware", "kd_hardware", "effort_limit_hardware_nm")]
    targets = target_sequence(config, count, steps)
    traces = {key: [] for key in fields}
    started = time.perf_counter()
    for step in range(steps):
        fields["ctrl"][:] = np.clip(kp * (targets[step] - fields["qpos"][:, 7:]) - kd * fields["qvel"][:, 6:], -effort, effort)
        batch.step()
        if capture:
            for key in traces:
                traces[key].append(fields[key].copy())
    elapsed = time.perf_counter() - started
    return elapsed, {key: np.asarray(value) for key, value in traces.items()} if capture else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plain-only", action="store_true")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 8, 32, 64])
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--compare-trace", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("choose a new benchmark output folder")
    args.output.mkdir(parents=True)
    model, config, sources = prepare_native()
    report = dict(python=platform.python_version(), executable=os.sys.executable, mujoco=mujoco.__version__,
                  numpy=np.__version__, logical_cpus=os.cpu_count(), native_dimensions=[model.nq,model.nv,model.nu],
                  physics_timestep_s=model.opt.timestep, pd_substep_recompute=True, target_hold_controls=10,
                  model_and_config_sha256=sources, concurrent_gpu_training_and_cpu_jobs=True, benchmarking_only=True)
    _, plain = run_plain(model, config, 8, 100, capture=True)
    np.savez_compressed(args.output / "plain_parity_trace.npz", **plain)
    if args.compare_trace:
        with np.load(args.compare_trace, allow_pickle=False) as other:
            report["cross_version_trace_difference"] = {key: float(np.max(np.abs(plain[key]-other[key]))) for key in plain}
    if not args.plain_only:
        report["mjbatch"] = importlib.metadata.version("mjbatch")
        _, batched = run_batch(model, config, 8, 100, min(args.threads,8), capture=True)
        report["batch_single_parity"] = {key: dict(max_abs=float(np.max(np.abs(plain[key]-batched[key]))),
                                                  bit_exact=bool(np.array_equal(plain[key],batched[key]))) for key in plain}
        for key in plain:
            np.testing.assert_array_equal(plain[key], batched[key], key)
    rows = []
    for count in args.sizes:
        plain_times = [run_plain(model, config, count, args.steps)[0] for _ in range(args.repeats)]
        row = dict(batch_size=count, physics_steps_per_sim=args.steps, repeats=args.repeats,
                   plain_seconds_median=float(np.median(plain_times)),
                   plain_physics_steps_per_second=count*args.steps/float(np.median(plain_times)), batches=[])
        if not args.plain_only:
            for threads in sorted(set((1, min(args.threads,count)))):
                times = [run_batch(model, config, count, args.steps, threads)[0] for _ in range(args.repeats)]
                median = float(np.median(times))
                row["batches"].append(dict(threads=threads, seconds_median=median,
                                           physics_steps_per_second=count*args.steps/median,
                                           speedup_vs_plain=float(np.median(plain_times))/median,
                                           equivalent_50hz_controls_per_second=count*args.steps/(10*median)))
        rows.append(row)
        print(json.dumps(row), flush=True)
    report["throughput"] = rows
    with (args.output / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({key:value for key,value in report.items() if key != "throughput"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
