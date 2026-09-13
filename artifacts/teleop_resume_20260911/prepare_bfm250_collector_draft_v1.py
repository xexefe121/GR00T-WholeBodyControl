"""Prepare conditional collector source only; no inference or collection."""
from pathlib import Path
import ast,json,shutil
NEW=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911')
OLD=NEW/'fresh_expert_labels_resume_v1/source_snapshot_v2'
BASE=NEW/'bfm_entry250_labels_v1';BASE.mkdir(exist_ok=False)
SNAP=BASE/'source_draft_v1';SNAP.mkdir()
r=json.loads((OLD.parent/'collector_frozen_inputs_v2.json').read_text())
for name in r['source_sha256']:
    if name in ('collect_actual_branch_labels.py','audit_label_compatibility.py'):continue
    dest=SNAP/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(OLD/name,dest)
s=(OLD/'collect_actual_branch_labels.py').read_text()
s=s.replace("RUN = TASK / 'student_actual_oracle_control1_resume1001_v1'", "RUN = TASK / 'bfm_entry250_actual_oracle_v1'")
s=s.replace("ORIGINAL = TASK / 'student_actual_oracle_control1_v1'", "BASELINE = TASK / 'original_bfm_entry250_v1'")
s=s.replace("STUDENT = TASK / 'fast_controller_continued_fit_v1'", "STUDENT = TASK / 'fast_controller_aggregate_fit_v1'")
s=s.replace("    assert sha(RUN / 'resume_inputs/trace.partial.npz') == frozen['checkpoint1001_sha256']\n    assert sha(ORIGINAL / 'frozen_inputs_v2.json') == frozen['original_v2_receipt_sha256']\n    assert sha(RUN / 'resume_receipt.json') == frozen['resume_receipt_sha256']", "    assert sha(RUN / 'inputs/baseline250_trace.npz') == frozen['baseline250_sha256']\n    assert sha(RUN / 'frozen_inputs.json') == frozen['expert_branch_receipt_sha256']\n    assert sha(RUN / 'prelaunch_clearance.json') == frozen['expert_launch_clearance_sha256']")
s=s.replace("    checkpoint = archive(RUN / 'resume_inputs/trace.partial.npz')", "    baseline = archive(RUN / 'inputs/baseline250_trace.npz')\n    snapshot = archive(RUN / 'inputs/precontrol250.npz')")
s=s.replace("np.r_[0, np.ones(1268), np.full(300, 2)]", "np.r_[np.zeros(250), np.ones(1019), np.full(300, 2)]")
start=s.index('    for key, old in checkpoint.items():')
end=s.index('    limits = np.asarray',start)
s=s[:start]+'''    direct=('target','source_frame','global_control','controller_mode','joint_error','root_error','physics_substeps',
        'control_integration_before','control_history_before','control_previous_action_before','previous_action','action',
        'qpos','qvel','physics_qpos','physics_qvel','physics_torque','physics_time','physics_expected_time',
        'range_excess','velocity_ratio','effort_ratio','clock_error')
    mapping={key:key for key in direct}
    mapping.update(physics_actuator_force='physics_actuator_torque',physics_warning_number='physics_warning_counts',
        physics_warning_lastinfo='physics_warning_lastinfo')
    mapping.update({'control_history_'+key:'control_history_before_'+key for key in KEYS})
    for combined, original in mapping.items():
        np.testing.assert_array_equal(trace[combined][:len(baseline[original])],baseline[original])
    np.testing.assert_array_equal(trace['initial_integration'],baseline['initial_integration'])
    np.testing.assert_array_equal(trace['branch_initial_integration'],baseline['final_integration'])
    np.testing.assert_array_equal(trace['control_integration_before'][250],snapshot['integration'])
    np.testing.assert_array_equal(trace['control_previous_action_before'][250],snapshot['previous_action'])
    np.testing.assert_array_equal(trace['control_history_before'][250],snapshot['history_flat'])
    for key in KEYS:np.testing.assert_array_equal(trace['control_history_'+key][250],snapshot['history_'+key])
    accumulated=0.
    assert trace['physics_expected_time'][0]==0.
    for i in range(15690):
        accumulated+=.002
        assert accumulated==trace['physics_expected_time'][i+1]==trace['physics_time'][i+1]
''' +s[end:]
s=s.replace('        if control > 0:', '        if control >= 250:')
s=s.replace('        # Student control0 is provenance only and keeps its combined PRECLIP action.\n        previous = (trace[\'action\'][0].copy() if control == 0 else', "        # Original BFM controls0..249 are provenance only; preserve their raw actions.\n        previous = (trace['action'][control].copy() if control < 250 else")
s=s.replace("np.arange(1, 1269)","np.arange(250, 1269)")
start=s.index('    student = archive(STUDENT /')
end=s.index('    span = np.diff',start)
s=s[:start]+'''    np.testing.assert_array_equal(arrays['source_frame'],np.arange(261,1280))
    np.testing.assert_array_equal(previous,trace['control_previous_action_before'][1269])
    for key in KEYS:np.testing.assert_array_equal(history.data[key],trace['control_history_'+key][1269])
    query_parity={}
    for key, actual, expected in (
        ('previous_action',arrays['previous_action'][0],snapshot['previous_action']),
        ('history',arrays['history'][0],snapshot['history_flat']),
        ('qpos',trace['qpos'][250],snapshot['qpos']),('qvel',trace['qvel'][250],snapshot['qvel']),
        ('integration',trace['control_integration_before'][250],snapshot['integration'])):
        np.testing.assert_array_equal(actual,expected)
        query_parity[key]=dict(bitexact=True,max_abs_difference=0.)
    first_fresh=archive(RUN/'initial_seed/fresh_bfm.npz')['targets'][0]
    np.testing.assert_array_equal(np.clip(arrays['base_target'][0],limits[:,0],limits[:,1]),first_fresh)
    query_parity['clipped_base_vs_initial_fresh_seed']=dict(bitexact=True,max_abs_difference=0.)
''' +s[end:]
s=s.replace("trace['qpos'][1:1270]", "trace['qpos'][250:1270]").replace("trace['qvel'][1:1270]", "trace['qvel'][250:1270]")
s=s.replace("trace['control_integration_before'][1:1269]", "trace['control_integration_before'][250:1269]")
s=s.replace("kind='one_qualified_actual_student_state_expert_branch_labels', samples=1268", "kind='one_qualified_BFM250_actual_expert_transition_labels', samples=1019")
s=s.replace("expert_controls=[1, 1268], excluded_student_prefix_control=0", "expert_controls=[250, 1268], excluded_BFM_prefix_controls=[0,249]")
s=s.replace('query_input_student_parity=query_parity', 'query250_input_and_initial_base_parity=query_parity')
s=s.replace("        original_v2_receipt_sha256=frozen['original_v2_receipt_sha256'],\n        resume_receipt_sha256=frozen['resume_receipt_sha256'],\n        checkpoint1001_sha256=frozen['checkpoint1001_sha256'],", "        expert_branch_receipt_sha256=frozen['expert_branch_receipt_sha256'],\n        expert_launch_clearance_sha256=frozen['expert_launch_clearance_sha256'],\n        baseline250_sha256=frozen['baseline250_sha256'],")
s=s.replace('actor_inference_calls=1268', 'actor_inference_calls=1019').replace('dict(samples=1268,','dict(samples=1019,')
s=s.replace("    np.savez_compressed(destination / 'existing_normalization.npz',", "    original_normalization=archive(TASK/'fast_controller_continued_fit_v1/fit/teacher_fit.npz')\n    for key in ('feature_mean','feature_std'):np.testing.assert_array_equal(normalization[key],original_normalization[key])\n    np.savez_compressed(destination / 'existing_normalization.npz',")
ast.parse(s);(SNAP/'collect_bfm250_labels.py').write_text(s)
(BASE/'README.md').write_text('Prepared source draft only. No collector inference, physics or fit has run. Runtime refuses to proceed until collector_frozen_inputs.json and an exact root qualification receipt exist. Final source/input freeze follows completed branch qualification; trainer freeze waits collected labels and compatibility.\n')
print(BASE)
