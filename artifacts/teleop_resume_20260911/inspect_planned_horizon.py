"""Check saved planned targets on a full native integration-state continuation."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from verify_feasibility_referee import archive, restore, oracle, BUNDLE, NEW, PROBE, ORACLE
from gear_sonic.utils.g1_true23_mjbatch_mpc import load_native_bundle, sha256
import numpy as np

native, contract, *_ = load_native_bundle(BUNDLE, 'pico')
fixture_path = PROBE / 'actual_3740_integration_state.npz'
plan_path = NEW / 'hard_feasibility_3740_v1/trace.npz'
fixture, plan = archive(fixture_path), archive(plan_path)
data = restore(native, fixture)
report, trace = oracle.inspect_native_segment(native, data, plan['planned_targets'], contract)
count = report['complete_controls']
actual = np.concatenate((trace['physics_qpos'][::10], trace['physics_qvel'][::10]), axis=1)
report.update(
    kind='independent_native_continuation_of_saved_planned_targets',
    predicted_vs_continuous_native_qpos_max_abs=float(np.max(np.abs(actual[:, :30] - plan['planned_states'][:count + 1, :30]))),
    predicted_vs_continuous_native_qvel_max_abs=float(np.max(np.abs(actual[:, 30:] - plan['planned_states'][:count + 1, 30:]))),
    feedback_applied=False, controller_replanning_performed=False,
    full_source_qualified=False, recovery_qualified=False,
    purpose='Check whether this accepted nominal horizon also stays feasible under continuous native warmstart; it is not a new controller trial.',
    input_hashes={str(p): sha256(p) for p in (fixture_path, plan_path, ORACLE, Path(__file__))},
)
output = NEW / 'hard_feasibility_3740_nominal_native_horizon_v1'
output.mkdir(exist_ok=False)
np.savez_compressed(output / 'trace.npz', **trace)
(output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
(output / 'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(report), flush=True)
