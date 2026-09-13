"""Read-only source/array assessment. No policy, optimizer, or dynamics calls."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT = Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof')
NEW = Path('E:/codex-artifacts/sonic23_teleop_resume_20260911')
OLD = Path('E:/codex-artifacts/sonic23_teleop_six_hour_20260910')
OUT = Path(__file__).parent
BUNDLE = ROOT / 'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1'
RUNNER = OLD / 'bfm_online_intent_v2/walk003_terminal_bfm_yaw4_hybrid_v1/runner_snapshot.py'
FROZEN = NEW / 'preserved_walk_demo_v1/repo'
MAIN = NEW / 'walk002_full_control_lm_v1'
REFERENCE = OLD / 'mjbatch_intent_floor_inputs_v1/walk002/reference.npz'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def read(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}

def exact(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()

def write(p, data):
    p.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')

trace = read(MAIN / 'trace.npz')
oracle = read(NEW / 'walk002_full_control_lm_independent_physics_v1/trace.npz')
intent = json.loads((NEW / 'walk002_full_control_lm_independent_intent_v1/report.json').read_text())
contract = json.loads((BUNDLE / 'contract.json').read_text())
timeline = json.loads((BUNDLE / 'walk002/timeline.json').read_text())
phases = {p['name']: p for p in timeline['phases']}
switch = phases['returned_standing']['control_start']
count = len(trace['target'])
assert switch == 1117 and count == 1417
assert phases['source_motion']['control_start'] == 350
assert phases['source_motion']['control_stop'] == 1017
assert intent['full_lifecycle_source_intent_pass'] and not intent['requested_segment_quiet_pass']

# Execute only the archived pure-array independent history algebra. No module
# import: the archived module also imports MuJoCo and contains an executable trial.
tree = ast.parse(RUNNER.read_text())
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'direct_history')
scope = {'np': np}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(RUNNER), 'exec'), scope)
state, previous, history = scope['direct_history'](trace, contract, switch + 1)
observation_path = FROZEN / 'gear_sonic/utils/g1_true23_bfm_seed_observations.py'
spec = importlib.util.spec_from_file_location('frozen_observation_array_contract', observation_path)
observation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observation)
actual_history = observation.BFMHistory()
exact_states, exact_histories = [], []
for control in range(switch + 1):
    q, dq = trace['qpos'][control], trace['qvel'][control]
    st, terms = observation.state_and_terms(q[7:], dq[6:], q[3:7], dq[3:6], previous[control], np.asarray(contract['default_q']))
    exact_states.append(st)
    exact_histories.append(actual_history.before_update(terms))
exact_states, exact_histories = np.asarray(exact_states), np.asarray(exact_histories)
signed_zero_only = np.array_equal(history, trace['fresh_seed_measured_history'][:switch+1])
signed_zero_difference_count = int(np.count_nonzero(history.view(np.uint32) != trace['fresh_seed_measured_history'][:switch+1].view(np.uint32)))
checks = {
    'all_1118_precontrol_previous_actions_bitexact': exact(previous, trace['fresh_seed_previous_action'][:switch+1]),
    'independent_direct_algebra_histories_value_exact': signed_zero_only,
    'all_1118_frozen_contract_flat300_histories_bitexact': exact(exact_histories, trace['fresh_seed_measured_history'][:switch+1]),
    'independent_direct_algebra_actor_states_value_exact': bool(np.array_equal(state, exact_states)),
    'prefix1117_all_native_substeps_complete': bool(np.all(trace['physics_substeps'][:switch] == 10)),
    'prefix_source_frames11_through1127_exact': exact(trace['source_frame'][:switch], np.arange(11, switch+11, dtype=trace['source_frame'].dtype)),
    'original_source667_finished_before_switch': phases['source_motion']['control_stop'] <= switch,
    'return100_finished_at_original_switch': phases['return_ramp']['control_stop'] == switch,
    'terminal250_plus_original_margin50': phases['returned_standing']['requested_controls'] == 250 and phases['standing_proof_margin']['requested_controls'] == 50,
}
for source_key, oracle_key in (
    ('physics_qpos', 'physics_qpos'), ('physics_qvel', 'physics_qvel'),
    ('physics_torque', 'physics_torque'), ('physics_actuator_force', 'physics_actuator_force'),
    ('physics_time', 'physics_time'), ('physics_warning_number', 'warning_counts'),
    ('physics_warning_lastinfo', 'warning_lastinfo')):
    n = switch * 10 + int(source_key not in ('physics_torque', 'physics_actuator_force'))
    checks['root_oracle_prefix_' + source_key + '_bitexact'] = exact(trace[source_key][:n], oracle[oracle_key][:n])
assert all(checks.values()), checks

np.savez_compressed(OUT / 'saved_boundary1117_comparison_only.npz',
    qpos=trace['qpos'][switch], qvel=trace['qvel'][switch],
    time=trace['physics_time'][switch*10], state=exact_states[-1],
    previous_action=previous[-1], measured_history=exact_histories[-1],
    warning_number=trace['physics_warning_number'][switch*10],
    warning_lastinfo=trace['physics_warning_lastinfo'][switch*10],
    comparison_only=np.asarray(True), physical_initialization_allowed=np.asarray(False),
    original_trace_sha256=np.asarray(sha(MAIN / 'trace.npz')))

paths = [Path(__file__), RUNNER, FROZEN.parent / 'mapping.json',
    MAIN / 'trace.npz', MAIN / 'report.json', MAIN / 'request.json',
    NEW / 'walk002_full_control_lm_v1_process/exit_status.json',
    NEW / 'walk002_full_control_lm_v1_process/postrun_provenance.json',
    NEW / 'walk002_full_control_lm_v1_process/actual_request_verification.json',
    NEW / 'walk002_full_control_lm_independent_physics_v1/report.json',
    NEW / 'walk002_full_control_lm_independent_physics_v1/trace.npz',
    NEW / 'walk002_full_control_lm_independent_intent_v1/report.json',
    NEW / 'walk002_full_endpoint_independent_v1/report.json',
    NEW / 'walk002_full_endpoint_independent_v1/endpoint.npz',
    REFERENCE, REFERENCE.parent / 'portable_receipt.json',
    OUT / 'saved_boundary1117_comparison_only.npz']
paths += [BUNDLE / p for p in ('manifest.json', 'contract.json', 'native_prepared.xml', 'prepared_model_arrays.npz',
    'walk002/timeline.json', 'walk002/native_original.npz', 'walk002/original29.npz')]
paths += [ROOT / 'artifacts/teleop_six_hour_20260910/bfm_onnx_v2' / p for p in ('manifest.json', 'actor.onnx', 'backward.onnx')]
paths += [p for p in FROZEN.rglob('*.py')]

proposal = dict(
    kind='PREPARE_ONLY_one_offline_saved_MPC_prefix_then_actual_BFM_terminal_hybrid',
    execution_authorized=False, controller_source_ready=False,
    clip='walk002', original_main_physics_and_all667_source_pass=True,
    original_main_quiet_pass=False, original_quiet_metrics=intent['quiet_last_three_seconds'],
    checks=checks, saved_array_checks_pass=all(checks.values()),
    independent_algebra_difference=dict(numerically_equal=signed_zero_only, signed_zero_history_components=signed_zero_difference_count,
        note='Archived vector gravity expression differs in eight signed-zero components at initial history only. Exact frozen observation expression matches every source byte; runtime gate will use that expression. No tolerance introduced.'),
    main_trace_sha256=sha(MAIN/'trace.npz'),
    source_metrics=intent['source_metrics'],
    intended_lifecycle_controls=1417, intended_extension_controls=250,
    prefix=dict(global_controls_inclusive=[0,1116], physics_steps=11170,
        execution='recompute native manual PD from original applied MPC targets, one fresh canonical MjData at unchanged v4 frame10',
        origin='qualified exact-controller walk002 full trace; never call this fresh MPC',
        source_controls=667, return_controls=100),
    terminal=dict(global_controls_inclusive=[1117,1416], actual_BFM_controls=300,
        original_terminal_start=1117, reference_frame_first=1128, reference_frame_last=1427,
        yaw_gain=4., position_gain=1., horizon=8, inference_threads=1,
        reference='unchanged walk002 native_original.npz; unchanged v4 reference remains evaluation/initialization only',
        controller='archived walk003 terminal_goal_yaw4 plus unchanged original BFM actor/backward'),
    extension=dict(global_controls_inclusive=[1417,1666], actual_BFM_controls=250,
        reference_frame=1427, continuity='same live MjData, controller history, previous raw action and repeated clock; separate trace',
        launch_condition='complete lifecycle and every strict physical gate; preserve original lifecycle quiet failure even if extension later passes'),
    required_adapter_changes=[
        'Explicit walk002 and fixed timeline assertions: source350:1017, return1017:1117, terminal1117:1417, separate250.',
        'Bind root current report schema and hashes; replace old walk003 baseline schema assumptions.',
        'Before first BFM command compare every native prefix q/dq/requested+applied torque/actual force/time/warning count+lastinfo, every actual target/source frame/history/prior action to qualified trace and root audit; mismatch stops before inference.',
        'Independent expected clock starts zero and adds .002 each physical step; never assign it from observed data.time.',
        'Snapshot complete integration8191/291, qacc_warmstart, ctrl, warning ledger and named controller history at switch1117 and lifecycle boundary1417; comparison/recording only, no setter or forward at handoff.',
        'Copy live measured history at switch; prior raw action comes from actual MPC target normalized with training effort. After switch preserve actor raw*5 before native target clipping.',
        'Preserve goal/observation/actor numerical expressions exactly. Do not append reference rows or change latent averaging at EOF.',
        'Abort at first native range excess>1e-6 (old runner waited .01), speedratio>1, effortratio>1+1e-9, warning, nonfinite/fall or repeated-clock mismatch; save partial current substep and first failure.',
        'Canonical trace schema with physics_requested_torque, physics_actuator_force, physics_expected_time, physics_warning_number/lastinfo, source_frame/global_control/controller_mode and actual policy inputs; normalize empty array trailing shapes.',
        'Save final main and initial extension complete291/history/clock exactly equal; never use extra250 to replace the original300 quiet window.',
        'Freeze artifact-local runner and import tree; preflight input hashes; durable hidden launcher with start/exit/command/provenance, no duplicate output; root independent full source and both quiet reports afterward.'
    ],
    permitted_claim_on_success='one offline hybrid original-source lifecycle and separate quiet hold; recorded MPC prefix with actual terminal BFM',
    fresh_MPC_or_live_or_realtime_claim=False, physics_or_inference_executed=False,
    hardware_authorized=False, input_hashes={str(p):sha(p) for p in paths})
write(OUT / 'proposal.json', proposal)
print(json.dumps(dict(proposal_sha256=sha(OUT/'proposal.json'), checks=len(checks), all_pass=all(checks.values()), input_hashes=len(paths), boundary_sha256=sha(OUT/'saved_boundary1117_comparison_only.npz'))))
