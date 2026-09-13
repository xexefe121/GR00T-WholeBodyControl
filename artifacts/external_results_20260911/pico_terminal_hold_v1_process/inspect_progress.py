"""Saved-output monitor and request/provenance comparison; no dynamics."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
OUTPUT = BASE / "pico_terminal_hold_v1"
def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()
previous = json.loads((BASE / "pico_full_control_lm_v1/request.json").read_text())
request = json.loads((OUTPUT / "request.json").read_text())
fields = ['clip', 'probe', 'horizon', 'commit', 'iterations', 'finite_difference_epsilon', 'batch_threads',
          'feedback_correction_clip_rad', 'costs', 'mujoco', 'restoration', 'hard_feasibility',
          'recorded_target_seed', 'fresh_bfm_seed', 'motion_override', 'source_clock_hz', 'physics_hz',
          'source_reference_lift_m', 'source_frame_removal', 'actual_execution', 'planning_actuators',
          'feedback_gain_semantics', 'exact_solver_reuse']
checks = {key: previous[key] == request[key] for key in fields}
assert all(checks.values()), checks
receipt = dict(checks=checks, all_same=True,
               canonical_request_sha256=sha(BASE / "pico_full_control_lm_v1/request.json"),
               hold_request_sha256=sha(OUTPUT / "request.json"))
(HERE / "actual_request_verification.json").write_text(json.dumps(receipt, indent=2))
status = dict(utc=datetime.now(timezone.utc).isoformat(), request23fields_equal=True,
              boundary_verified=json.loads((OUTPUT / "boundary_verification.json").read_text())["verified"])
if (OUTPUT / "report.json").exists():
    report = json.loads((OUTPUT / "report.json").read_text())
    status.update({key: report[key] for key in ('completed_controls', 'requested_controls', 'physics_steps',
                                               'failure', 'probe_completed', 'simulation_time')})
    launch = json.loads((HERE / "launch_receipt.json").read_text())
    comparisons = {path: sha(Path(path)) == digest for path, digest in launch['verified_hashes'].items()}
    assert all(comparisons.values()), comparisons
    (HERE / "postrun_provenance.json").write_text(json.dumps(dict(all_exact=True, count=len(comparisons), checks=comparisons), indent=2))
    status['all104_launch_hashes_unchanged'] = True
    status['trace_sha256'] = sha(OUTPUT / 'trace.npz')
    status['report_sha256'] = sha(OUTPUT / 'report.json')
elif (OUTPUT / "trace.partial.npz").exists():
    with np.load(OUTPUT / "trace.partial.npz", allow_pickle=False) as archive:
        metadata = json.loads(str(archive['checkpoint_metadata']))
        status.update({key: metadata[key] for key in ('completed_controls', 'completed_global_controls',
                                                      'physics_steps', 'simulation_time', 'failure')})
with (HERE / 'monitor.jsonl').open('a') as stream:
    stream.write(json.dumps(status) + '\n')
print(json.dumps(status))
