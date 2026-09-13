"""Pure saved-array audit. No policy, MuJoCo, optimizer, or repository imports."""
from pathlib import Path
import hashlib
import json
import numpy as np

OUT = Path(__file__).resolve().parent
OLD = Path('E:/codex-artifacts/sonic23_teleop_six_hour_20260910')
NEW = OUT.parent
REPO = Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof')
BUNDLE = REPO / 'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1'
WALK = OLD / 'bfm_online_intent_v2/walk003_terminal_bfm_yaw4_hybrid_v1'
PICO = OLD / 'mjbatch_full_v1/pico_v4_native323_freshseed_allmargin_5iter_full_v1'
CONT = NEW / 'hard_feasibility_restoration_continuation_3740_v1'
FIXTURE = NEW / 'recovery_probe_v1'
DT = .02
contract = json.loads((BUNDLE / 'contract.json').read_text())
default, kp, training_effort, velocity = [np.asarray(contract[k], float) for k in
                                       ('default_q', 'kp', 'training_effort', 'native_velocity')]
names = contract['joint_names']
groups = {'all':slice(None), 'legs':slice(0, 12), 'waist':slice(12, 13), 'arms':slice(13, 23)}
inputs = []

def load(path, keys=None):
    inputs.append(path)
    with np.load(path, allow_pickle=False) as a:
        return {k:a[k].copy() for k in (a.files if keys is None else keys)}

def action(target):
    # Exact recorded-control BFM normalization, preserving operation order.
    return ((target-default)*kp/(.25*training_effort)).astype(np.float32)

def stats(value):
    a = np.abs(np.asarray(value, float))
    return dict(p50_p95_p99_max=np.percentile(a, [50,95,99,100]).tolist(),
                rms=float(np.sqrt(np.mean(a*a))))

def band(value, limit=5.):
    a = np.abs(np.asarray(value, float))
    return dict(**stats(a), fraction_components_outside=float(np.mean(a>limit)),
                fraction_controls_with_any_outside=float(np.mean(np.any(a.reshape(len(a),-1)>limit,axis=1))))

def record(name, a, first, stop, global_start, previous_target, pre_actions, histories):
    u, q, dq = a['target'][first:stop], a['qpos'][first:stop,7:], a['qvel'][first:stop,6:]
    next_q = a['qpos'][first+1:stop+1,7:]
    prior = np.vstack((previous_target, a['target'][first:stop-1]))
    delta = u-prior
    qrate = (next_q-q)/DT
    indices = np.arange(first, stop)
    controls = global_start+indices
    assert np.array_equal(a['source_frame'][first:stop], controls+11)
    act = action(u)
    previous = pre_actions[first:stop]
    memory = histories[first:stop]
    assert memory.shape == (len(controls),300)
    np.testing.assert_array_equal(previous, action(prior))
    np.testing.assert_array_equal(a['qpos'][first:stop+1], a['physics_qpos'][first*10:stop*10+1:10])
    physical = a['physics_qvel'][first*10:stop*10+1,6:]
    rows = {}
    for group, columns in groups.items():
        group_joints = np.arange(23)[columns]
        jump = delta[:,columns]
        slew = np.abs(jump)/DT
        rate_ratio = slew/velocity[columns]
        max_flat = np.argmax(slew)
        control_row, joint_column = np.unravel_index(max_flat, slew.shape)
        joint = int(group_joints[joint_column])
        rows[group] = dict(
            target_jump_rad=stats(jump), target_slew_rad_per_s=stats(slew),
            target_slew_over_native_joint_speed=band(rate_ratio, 1.),
            measured_joint_average_rate_rad_per_s=stats(qrate[:,columns]),
            measured_joint_instantaneous_rate_rad_per_s=stats(dq[:,columns]),
            every2ms_measured_speed_ratio=stats(physical[:,columns]/velocity[columns]),
            target_minus_precontrol_measured_joint_rad=stats((u-q)[:,columns]),
            normalized_applied_target_action=band(act[:,columns]),
            previous_action_as_supplied=band(previous[:,columns]),
            largest_slew=dict(global_control=int(controls[control_row]), source_time_s=float((controls[control_row]-350)*DT),
                              joint=names[joint], target_before_rad=float(prior[control_row,joint]),
                              target_after_rad=float(u[control_row,joint]), target_jump_rad=float(delta[control_row,joint]),
                              target_slew_rad_per_s=float(slew[control_row,joint_column]),
                              measured_joint_before_rad=float(q[control_row,joint]),
                              measured_joint_after_rad=float(next_q[control_row,joint])),
        )
    result = dict(name=name, controls=len(controls), global_controls_first_last=[int(controls[0]),int(controls[-1])],
                  source_interval_s=[float((controls[0]-350)*DT),float((controls[-1]+1-350)*DT)],
                  source_frames_first_last=[int(a['source_frame'][first]),int(a['source_frame'][stop-1])],
                  boundary_previous_target_included=True,
                  source_only=True, groups=rows,
                  action_history=band(memory[:,:92]),
                  position_history_relative_default_rad=stats(memory[:,104:196]),
                  velocity_history_over_native_limit=stats(memory[:,196:288].reshape(-1,4,23)/velocity),
                  saved_previous_action_matches_normalized_applied_target=True,
                  precontrol_qpos_matches_every_tenth_physics_state=True)
    return result

w = load(WALK/'trace.npz', ('target','qpos','qvel','physics_qpos','physics_qvel','source_frame','history','previous_action','action'))
p = load(PICO/'trace.npz', ('target','qpos','qvel','physics_qpos','physics_qvel','source_frame','fresh_seed_previous_action','fresh_seed_measured_history'))
r = load(CONT/'trace.npz')
f = load(FIXTURE/'actual_3740_integration_state.npz')
h = load(FIXTURE/'actual_3740_bfm_history.npz')
np.testing.assert_array_equal(w['action'][:1269], action(w['target'][:1269]))
np.testing.assert_array_equal(p['qpos'][3740], f['qpos'])
np.testing.assert_array_equal(p['qvel'][3740], f['qvel'])
np.testing.assert_array_equal(r['qpos'][0], f['qpos'])
np.testing.assert_array_equal(r['qvel'][0], f['qvel'])
np.testing.assert_array_equal(h['previous_action'], action(p['target'][3739]))
np.testing.assert_array_equal(h['memory_actions'], action(p['target'][[3738,3737,3736,3735]]))
np.testing.assert_array_equal(r['fresh_seed_previous_action'][0], h['previous_action'])
np.testing.assert_array_equal(r['fresh_seed_measured_history'][0,:92], h['memory_actions'].reshape(-1))
np.testing.assert_array_equal(r['final_previous_action'], action(r['target'][-1]))

# Validate action-memory lag independently: before processing c, the four
# history actions correspond to targets c-2,c-3,c-4,c-5; previous_action is c-1.
for a, selected, origin, key in [(w,np.arange(350,1169),0,'history'),
                                (p,np.arange(350,3740),0,'fresh_seed_measured_history')]:
    expected = action(a['target'][selected[:,None]-np.arange(2,6)[None,:]])
    np.testing.assert_array_equal(a[key][selected,:92].reshape(-1,4,23),expected)
joined = np.concatenate((p['target'][:3740],r['target']))
expected = action(joined[(3740+np.arange(65))[:,None]-np.arange(2,6)[None,:]])
np.testing.assert_array_equal(r['fresh_seed_measured_history'][:,:92].reshape(-1,4,23), expected)

records = [
    record('passing_walk003_full_source',w,350,1169,0,w['target'][349],w['previous_action'],w['history']),
    record('original_pico_source_prefix_to_fixture',p,350,3740,0,p['target'][349],p['fresh_seed_previous_action'],p['fresh_seed_measured_history']),
    record('original_pico_last65_before_fixture',p,3675,3740,0,p['target'][3674],p['fresh_seed_previous_action'],p['fresh_seed_measured_history']),
    record('restoration_continuation65',r,0,65,3740,p['target'][3739],r['fresh_seed_previous_action'],r['fresh_seed_measured_history']),
]
fixture = dict(global_control=3740, source_frame=3751, source_time_s=67.8,
               qpos_matches_original_and_continuation=True,
               qvel_matches_original_and_continuation=True,
               previous_action=band(h['previous_action'][None]),
               action_history=band(h['memory_actions'][None]),
               previous_action_joint_values={k:float(v) for k,v in zip(names,h['previous_action'])},
               history_action_max_abs_by_joint={k:float(v) for k,v in zip(names,np.abs(h['memory_actions']).max(axis=0))},
               previous_target_minus_actual_joint_rad=stats(p['target'][3739]-f['qpos'][7:]),
               actual_joint_speed_ratio=stats(f['qvel'][6:]/velocity),
               fixture_warm_targets_excluded='Private future BFM proposals are not actual commands and are not included in slew statistics.')
continuation_report = json.loads((CONT/'report.json').read_text())
inputs += [BUNDLE/'contract.json',WALK/'recorded_source_audit_v2.json',WALK/'report.json',
           PICO/'report.json',CONT/'report.json',CONT/'request.json',
           FIXTURE/'actual_3740_bfm_history_receipt.json',Path(__file__)]
inventory=[]
for path in inputs:
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1<<20),b''):digest.update(block)
    inventory.append(dict(path=str(path),bytes=path.stat().st_size,sha256=digest.hexdigest()))
report=dict(kind='read_only_native23_teacher_command_and_history_audit',dt=DT,physics_dt=.002,
            groups=names,records=records,fixture=fixture,
            continuation_completed_controls=continuation_report['completed_controls'],
            continuation_requested_controls=continuation_report['requested_controls'],
            continuation_failure=continuation_report['failure'],
            previous_action_and_four_history_lags_bit_exact=True,
            command_rate_is_not_actual_motor_speed_or_a_native_physics_violation=True,
            outside_five_is_a_declared_reference_band_not_a_measured_training_distribution=True,
            causation_established=False,policy_or_physics_run=False,provenance=inventory)
(OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
compact=[]
for record_ in records:
    all_=record_['groups']['all']
    compact.append(dict(name=record_['name'],controls=record_['controls'],
                        jump_p95_max=all_['target_jump_rad']['p50_p95_p99_max'][1::2],
                        slew_p95_max=all_['target_slew_rad_per_s']['p50_p95_p99_max'][1::2],
                        measured_rate_p95_max=all_['measured_joint_average_rate_rad_per_s']['p50_p95_p99_max'][1::2],
                        lead_p95_max=all_['target_minus_precontrol_measured_joint_rad']['p50_p95_p99_max'][1::2],
                        action_p95_max=all_['normalized_applied_target_action']['p50_p95_p99_max'][1::2],
                        action_component_outside5=all_['normalized_applied_target_action']['fraction_components_outside'],
                        action_control_outside5=all_['normalized_applied_target_action']['fraction_controls_with_any_outside'],
                        history_p95_max=record_['action_history']['p50_p95_p99_max'][1::2],
                        physical_max_ratio=all_['every2ms_measured_speed_ratio']['p50_p95_p99_max'][-1]))
print(json.dumps(dict(records=compact,fixture=fixture,continuation_failure=report['continuation_failure']),indent=2))
