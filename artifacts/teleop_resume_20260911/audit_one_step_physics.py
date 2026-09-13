"""Independently replay every nominal and policy branch through strict physics.

No actor, backward model, residual head, optimizer or new training labels.
Input manifest key names are explicit; the collector supplies no audit logic.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np

from branch_audit_oracle import load_oracle_with_endpoint


ROOT = Path(__file__).resolve().parents[2]
NEW = Path("/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911")
OLD = NEW.parent / "sonic23_teleop_six_hour_20260910"
TRACE_PATHS = [
    OLD / "bfm_online_intent_v2/walk003_terminal_bfm_yaw4_hybrid_v1/trace.npz",
    NEW / "student_actual_oracle_control1_resume1001_v1/nominal/trace.npz",
    NEW / "bfm_entry250_actual_oracle_v1/nominal/trace.npz",
]
TRACE_HASHES = [
    "93e108071973fb69746d22987bcd1918a20e75c14c17ddcbef8f7aa21f091488",
    "85de329f57b4582c61f5b7055e7e567f3b4a971d71fbcde1673998197eef149a",
    "721a44a88ba3c8d1c9a42c23ab3ec1d0b39b529c78abaa54df19f07ebf5a886a",
]
FIELDS = {
    "qpos": "physics_qpos", "qvel": "physics_qvel",
    "time": "physics_time", "command_torque": "physics_torque",
    "actuator_force": "physics_actuator_force",
    "warning_counts": "warning_counts", "warning_lastinfo": "warning_lastinfo",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def archive(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key].copy() for key in source.files}


def exact(a, b, label):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype or a.tobytes() != b.tobytes():
        raise AssertionError("Independent branch physics: " + label)


def array_digest(value):
    value = np.asarray(value)
    h = hashlib.sha256()
    h.update(str((value.shape, value.dtype.str)).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def atomic(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return "NaN" if np.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    return value


def main(args):
    assert np.__version__ == "1.26.4"
    import mujoco
    assert mujoco.__version__ == "3.2.3"
    frozen = NEW / "preserved_walk_demo_v1/repo"
    sys.path.insert(0, str(frozen))
    from gear_sonic.utils.g1_true23_mjbatch_mpc import load_native_bundle
    bundle = ROOT / "artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1"
    referee = ROOT / "gear_sonic/utils/g1_true23_feasibility_referee.py"
    oracle = load_oracle_with_endpoint(referee)
    assert sha(args.collection / "report.json") == args.collection_report_sha256
    producer = read(args.collection / "report.json")
    assert producer["completed"] and producer["passed"] and producer["rows"] == 3054
    assert producer["nominal_verified"] == producer["policy_branches"] == 3054
    assert sha(args.old_snapshots.parent / "report.json") == args.old_report_sha256
    capture_report = read(args.old_snapshots.parent / "report.json")
    assert capture_report["passed"] and capture_report["completed_native_steps"] == 12680
    assert capture_report["snapshots_sha256"] == sha(args.old_snapshots)
    array_dir = args.collection / "data"
    manifest_path = array_dir / "manifest.json"
    manifest = read(manifest_path)
    assert manifest["complete"] and manifest["rows"] == 3054
    assert manifest["request_sha256"] == producer["request_sha256"]
    assert sha(manifest_path) == producer["manifest_sha256"]
    native_ledger_path = args.collection / "native_rows.jsonl"
    assert sha(native_ledger_path) == producer["native_rows_sha256"] == manifest["native_rows_sha256"]
    producer_segments = {}
    observed_order = []
    with native_ledger_path.open() as stream:
        for line in stream:
            entry = json.loads(line)
            key = (entry["phase"], entry["row"])
            assert key not in producer_segments
            producer_segments[key] = entry
            observed_order.append(key)
    expected_order = [("nominal", row) for row in range(3054)] + [
        ("policy", row) for row in [2036] + [r for r in range(3054) if r != 2036]]
    assert observed_order == expected_order
    arrays = {}
    paths = [manifest_path, args.collection / "report.json", native_ledger_path, args.old_snapshots,
             args.old_snapshots.parent / "report.json", referee, Path(__file__),
             Path(__file__).with_name("branch_audit_oracle.py"),
             bundle / "contract.json", bundle / "native_prepared.xml",
             bundle / "prepared_model_arrays.npz",
             frozen / "gear_sonic/utils/g1_true23_mjbatch_mpc.py"]
    paths.extend(sorted((frozen / "gear_sonic/utils").glob("*.py")))
    paths.extend(sorted(Path(mujoco.__file__).parent.glob("*.so*")))
    paths.append(Path(mujoco.__file__))
    for key, entry in manifest["arrays"].items():
        path = (array_dir / entry["path"]).resolve()
        if not path.is_relative_to(array_dir.resolve()):
            raise ValueError("Array path leaves collection")
        assert sha(path) == entry["sha256"], key
        arrays[key] = np.load(path, mmap_mode="r", allow_pickle=False)
        assert list(arrays[key].shape) == entry["shape"], key
        assert arrays[key].dtype == np.dtype(entry["dtype"]), key
        paths.append(path)
    row_dataset = np.repeat(np.arange(3, dtype=np.int64), 1018)
    row_control = np.tile(np.arange(250, 1268, dtype=np.int64), 3)
    row_center = row_dataset * 1019 + row_control - 250
    for key, expected in (("dataset", row_dataset), ("start_control", row_control),
                          ("successor_control", row_control + 1), ("center_index", row_center)):
        exact(arrays[key], expected, key)
    traces = []
    for path, digest in zip(TRACE_PATHS, TRACE_HASHES):
        assert sha(path) == digest, str(path)
        traces.append(archive(path))
        paths.append(path)
    snapshot = archive(args.old_snapshots)
    exact(snapshot["control"], np.arange(1269, dtype=np.int64), "old boundary controls")
    starts = [snapshot["control_integration_before"]] + [
        trace["control_integration_before"] for trace in traces[1:]]
    witness_path = NEW / "velocity_chord_student_evaluation_v1/nominal/trace.npz"
    assert sha(witness_path) == "1e8e44a6c405a86138558ea89681fb225b73b2be969e2e7a608903ebb5f80872"
    witness = archive(witness_path)
    paths.append(witness_path)
    model, contract, *_ = load_native_bundle(bundle, "walk003")
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    assert mujoco.mj_stateSize(model, spec) == 291
    source = mujoco.MjData(model)
    roundtrip = np.empty(291)
    args.output.mkdir(parents=True, exist_ok=False)
    pins = {str(path): sha(path) for path in paths}
    request = dict(kind="independent_all_one_control_branch_native_replay",
                   requested_rows=3054, requested_nominal_segments=3054,
                   requested_policy_segments=3054, maximum_native_steps=61080,
                   model_evaluations=0, optimizer_updates=0, hashes=pins,
                   collection_report_sha256=args.collection_report_sha256,
                   old_report_sha256=args.old_report_sha256,
                   independent_oracle_retains_final_integration=True)
    atomic(args.output / "request.json", request)
    request_hash = sha(args.output / "request.json")
    endpoints = {kind: np.lib.format.open_memmap(
        args.output / (kind + "_final_integration.npy"), mode="w+",
        dtype=np.float64, shape=(3054, 291)) for kind in ("nominal", "policy")}
    for value in endpoints.values():
        value.fill(np.nan)
    counts = dict(nominal_segments=0, policy_segments=0, native_steps=0,
                  policy_feasible=0, policy_failed=0)
    active = {}
    first_witness = False
    log_path = args.output / "segments.jsonl"
    with log_path.open("x", encoding="utf-8") as log:
        try:
            for kind in ("nominal", "policy"):
                # A fixed calibration row first in the policy phase, once only.
                order = (list(range(3054)) if kind == "nominal" else
                         [2036] + [row for row in range(3054) if row != 2036])
                for row in order:
                    dataset, control = int(row_dataset[row]), int(row_control[row])
                    original = traces[dataset]
                    original_warning_key = ("physics_warning_counts" if dataset == 0
                                            else "physics_warning_number")
                    assert not np.any(original[original_warning_key][control * 10])
                    assert not np.any(original["physics_warning_lastinfo"][control * 10])
                    initial = starts[dataset][control]
                    active = dict(kind=np.asarray(kind), row=np.int64(row),
                                  dataset=np.int64(dataset), control=np.int64(control),
                                  initial=initial.copy())
                    exact(arrays[kind + "_start_integration"][row], initial, "source full291")
                    target = (original["target"][control] if kind == "nominal"
                              else arrays[args.policy_target_key][row])
                    active["target"] = target.copy()
                    mujoco.mj_setState(model, source, initial, spec)
                    source.warning.number[:] = 0
                    source.warning.lastinfo[:] = 0
                    mujoco.mj_forward(model, source)
                    assert not np.any(source.warning.number) and not np.any(source.warning.lastinfo)
                    mujoco.mj_setState(model, source, initial, spec)
                    source.warning.number[:] = 0
                    source.warning.lastinfo[:] = 0
                    mujoco.mj_getState(model, source, roundtrip, spec)
                    exact(roundtrip, initial, "native start roundtrip")
                    exact(source.qpos, original["qpos"][control], "starting measured qpos")
                    exact(source.qvel, original["qvel"][control], "starting measured qvel")
                    result, trace = oracle(model, source, target[None], contract,
                                           stop_on_failure=True, retain_trace=True)
                    steps = result["physics_steps"]
                    counts["native_steps"] += steps
                    active.update({"independent_" + key: value for key, value in trace.items()})
                    produced_segment = producer_segments[(kind, row)]
                    for key in ("feasible", "first_failure", "physics_steps", "maximum",
                                "minimum_root_height_m", "initial_time", "final_time",
                                "final_warning_counts", "original_data_unchanged"):
                        assert json_safe(result[key]) == produced_segment["report"][key], (kind, row, key)
                    for key in ("attempted", "returned", "captured"):
                        assert produced_segment[key] == steps, (kind, row, key)
                    assert steps == int(arrays[kind + "_valid_steps"][row])
                    assert steps == int(arrays[kind + "_attempted_steps"][row])
                    hashes = {}
                    for key, oracle_key in FIELDS.items():
                        length = steps if key in ("command_torque", "actuator_force") else steps + 1
                        exact(trace[oracle_key], arrays[kind + "_" + key][row, :length], key)
                        hashes[key] = array_digest(trace[oracle_key])
                        if kind == "nominal":
                            saved_key = ({"warning_counts": original_warning_key,
                                          "warning_lastinfo": "physics_warning_lastinfo",
                                          "actuator_force": ("physics_actuator_torque" if dataset == 0
                                                             else "physics_actuator_force")}.get(key, oracle_key))
                            recorded = original[saved_key][control * 10:control * 10 + length]
                            value = trace[oracle_key]
                            if key.startswith("warning"):
                                expected_dtype = np.dtype(np.int64 if dataset == 0 else np.int32)
                                assert recorded.dtype == expected_dtype
                                value = value.astype(expected_dtype)
                            exact(value, recorded, "original nominal " + key)
                    expected_clock = float(source.time)
                    clocks = [expected_clock]
                    for _ in range(steps):
                        expected_clock += .002
                        clocks.append(expected_clock)
                    exact(np.asarray(clocks), arrays[kind + "_expected_time"][row, :steps + 1],
                          "independent repeated clock")
                    exact(trace["final_integration"], arrays[kind + "_end_integration"][row],
                          "final full291")
                    endpoints[kind][row] = trace["final_integration"]
                    if kind == "nominal":
                        assert result["feasible"] and steps == 10
                        exact(trace["final_integration"], starts[dataset][control + 1],
                              "original nominal successor full291")
                    else:
                        assert bool(arrays["label_valid"][row]) == result["feasible"]
                        counts["policy_feasible" if result["feasible"] else "policy_failed"] += 1
                        if row == 2036:
                            exact(target, witness["target"][250], "actual250 command witness")
                            assert result["feasible"] and steps == 10
                            exact(trace["final_integration"], witness["control_integration_before"][251],
                                  "actual251 full291 witness")
                            for key, oracle_key in FIELDS.items():
                                length = 10 if key in ("command_torque", "actuator_force") else 11
                                saved_key = ({"warning_counts": "physics_warning_counts",
                                              "warning_lastinfo": "physics_warning_lastinfo",
                                              "actuator_force": "physics_actuator_torque"}.get(key, oracle_key))
                                exact(trace[oracle_key], witness[saved_key][2500:2500 + length],
                                      "actual250 ten-step witness " + key)
                            np.savez_compressed(args.output / "actual251_witness.npz", **trace)
                            first_witness = True
                    counts[kind + "_segments"] += 1
                    log.write(json.dumps(dict(kind=kind, row=row, dataset=dataset, control=control,
                              oracle=json_safe(result), actual_sample_hashes=hashes,
                              final_integration_hash=array_digest(trace["final_integration"])),
                              allow_nan=False) + "\n")
                    if counts[kind + "_segments"] % 200 == 0:
                        log.flush()
                        atomic(args.output / "progress.json", dict(stage=kind, **counts))
                        print(json.dumps(dict(stage=kind, **counts)), flush=True)
            assert counts["nominal_segments"] == counts["policy_segments"] == 3054
            assert first_witness and counts["native_steps"] <= 61080
            assert counts["native_steps"] == producer["native_returned"] == producer["native_attempted"]
            assert counts["policy_feasible"] == producer["labels_valid"]
            assert counts["policy_failed"] == producer["strict_failed"]
            for value in endpoints.values():
                value.flush()
            log.flush()
            for path, expected in pins.items():
                assert sha(path) == expected, path
            assert sha(args.output / "request.json") == request_hash
            report = dict(passed=True, kind="independent_all_one_control_branch_native_replay",
                          **counts, all_saved_samples_and_final291_byteexact=True,
                          actual_query250_to251_witness_byteexact=True,
                          original_requested_rows=3054, maximum_native_steps=61080,
                          model_evaluations=0, optimizer_updates=0, new_labels=0,
                          live_controller_qualification=False, request_sha256=request_hash,
                          hashes_unchanged=True, segments_sha256=sha(log_path))
            atomic(args.output / "report.json", report)
            print(json.dumps(report), flush=True)
        except BaseException as error:
            log.flush()
            for value in endpoints.values():
                value.flush()
            np.savez_compressed(args.output / "failure_active.npz", **active)
            atomic(args.output / "failure.json", dict(passed=False, **counts,
                   exception=repr(error), traceback=traceback.format_exc(),
                   original_requested_rows=3054, maximum_native_steps=61080,
                   completed_replay_counts_exclude_any_unreturned_oracle_call=True))
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("collection", "old-snapshots", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--collection-report-sha256", required=True)
    parser.add_argument("--old-report-sha256", required=True)
    parser.add_argument("--policy-target-key", default="policy_applied_target")
    main(parser.parse_args())
