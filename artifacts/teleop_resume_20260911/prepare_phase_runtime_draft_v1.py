"""Prepare one canonical phase runtime adapter without executing it."""
from pathlib import Path
import ast
NEW=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911')
SNAP=NEW/'fast_controller_phase_fit_v1/source_draft_v1'
s=(SNAP/'evaluate_nominal_pilot.py').read_text()
s=s.replace('def main(zero_only=False):','def main():')
start=s.index('    if zero_only:')
end=s.index("    fit=json.loads",start)
s=s[:start]+'''    baseline=archive(BASE.parent/'original_bfm_entry250_v1/entry250/trace.npz')
    baseline_report=json.loads((BASE.parent/'original_bfm_entry250_v1/entry250/report.json').read_text())
    assert baseline_report['full_segment_completed'] and baseline_report['quiet_standing_diagnostic']['quiet_standing_diagnostic_pass']
    assert sha(BASE.parent/'original_bfm_entry250_v1/entry250/trace.npz')==baseline_report['trace_sha256']
    labels=archive(BASE.parent/'bfm_entry250_labels_v1/labels/labels.npz')
    label_report=json.loads((BASE.parent/'bfm_entry250_labels_v1/labels/report.json').read_text())
    assert label_report['samples']==1019 and label_report['expert_controls']==[250,1268]
    assert sha(BASE.parent/'bfm_entry250_labels_v1/labels/labels.npz')==label_report['labels_sha256']
    np.testing.assert_array_equal(labels['control'],np.arange(250,1269))
    np.testing.assert_array_equal(labels['source_frame'],np.arange(261,1280))
''' +s[end:]
s=s.replace("    assert sha(BASE/'fit/student_head.onnx')==fit['checkpoints']['student_head.onnx']", "    assert sha(BASE/'fit/student_head.onnx')==fit['checkpoints']['student_head.onnx']\n    assert fit['steps']==65000 and fit['additional_updates']==5000\n    assert fit['full_objective_improved'] and fit['export_parity_passed'] and fit['rollout_numerical_prerequisites_pass']")
s=s.replace("controller='frozenoriginalBFM+rangecomplete linearspan residual first1269; frozenBFMyaw4 terminal'", "controller='originalBFM yaw2 disabledresidual0..249; linearspan residual250..1268; originalBFM yaw4 terminal1269 onward'")
s=s.replace("zero_parity_sha256=sha(BASE/'zero_parity/report.json')", "baseline250_trace_sha256=sha(BASE.parent/'original_bfm_entry250_v1/entry250/trace.npz')")
s=s.replace("prior_action='preclipcombinedBFM+linearresidual while learned; BFMrawactor*5 terminal; unclippedhistory'", "prior_action='rawBFM actor*5 initial250; preclipcombinedBFM+linearresidual250..1268; rawBFM actor*5 terminal; unclippedhistory'")
s=s.replace("simulation_privileged_root_pose_velocity=True,hardware_authorized=False,DAgger_queries_launched=False)", "simulation_privileged_root_pose_velocity=True,hardware_authorized=False,DAgger_queries_launched=False,\n        initial_BFM_controls=250,learned_controls=[250,1269],first_activation_query250_labels_sha256=label_report['labels_sha256'],\n        prefix_and_query_labels_comparison_only_no_injection=True,controller_mode_map={'0':'initial_BFM','1':'learned_residual','2':'terminal_BFM'})")
marker='    def segment(dest,start,count,expected_time):'
helpers=r'''    class TransitionParityError(RuntimeError):pass

    def record_parity(name,pairs):
        checks={key:dict(bitexact=bool(np.array_equal(actual,expected)),actual_shape=list(np.shape(actual)),expected_shape=list(np.shape(expected)))
            for key,actual,expected in pairs}
        passed=all(v['bitexact'] for v in checks.values())
        result=dict(passed=passed,checks=checks,comparison_only_no_injection=True,hardware_authorized=False)
        (BASE/name).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        if not passed:raise TransitionParityError(name+' failed: '+str([k for k,v in checks.items() if not v['bitexact']]))

    def verify_generated_prefix250(trace):
        fields=('qpos','qvel','target','source_frame','global_control','controller_mode','joint_error','root_error',
            'state','history','previous_action','action','base_target','delta','features',
            'physics_qpos','physics_qvel','physics_torque','physics_actuator_torque','physics_time','physics_expected_time',
            'physics_warning_counts','physics_warning_lastinfo','physics_substeps','range_excess','velocity_ratio','effort_ratio','clock_error',
            'control_integration_before','control_history_before','control_previous_action_before')
        pairs=[(key,np.asarray(trace[key])[:len(baseline[key])],baseline[key]) for key in fields]
        pairs.extend([('final_integration',get_state(native,data),baseline['final_integration']),
            ('recorded_controls',np.asarray(runtime.seed.recorded_controls),baseline['final_recorded_controls']),
            ('final_previous_action',runtime.seed.previous_action,baseline['final_previous_action'])])
        pairs.extend(('final_history_'+key,value,baseline['final_history_'+key]) for key,value in runtime.seed.history.data.items())
        record_parity('canonical_prefix250_parity.json',pairs)

    def verify_actual_query250(proposed):
        pairs=[(key,proposed[key],labels[key][0]) for key in ('features','base_target','previous_action','history','state')]
        pairs.extend([('integration',get_state(native,data),labels['control_integration_before'][0]),
            ('qpos',data.qpos,labels['teacher_qpos'][0]),('qvel',data.qvel,labels['teacher_qvel'][0])])
        record_parity('actual_query250_input_parity.json',pairs)

'''
s=s.replace(marker,helpers+marker)
s=s.replace("                proposed=runtime.propose(control,data.qpos,data.qvel,terminal=control>=switch)", "                if control==250:verify_generated_prefix250(trace)\n                proposed=runtime.propose(control,data.qpos,data.qvel,terminal=control>=switch,disable_head=control<250)\n                if control==250:verify_actual_query250(proposed)")
s=s.replace("failure=dict(reasons=['policy_inference_fault'],global_control=control", "failure=dict(reasons=['transition_parity_fault' if isinstance(exc,TransitionParityError) else 'policy_inference_fault'],global_control=control")
s=s.replace("controller_mode=1 if control>=switch else 0", "controller_mode=2 if control>=switch else (0 if control<250 else 1)")
s=s.replace("DAgger_not_launched=True,one_fixed_nominal_fit=True", "new_expert_query_launched=False,one_fixed_phase_only_fit=True,ordinary_final_global_step=65000")
start=s.index("if __name__=='__main__':")
s=s[:start]+"if __name__=='__main__':main()\n"
s=s.replace('"""One native323 student lifecycle, then a separately stored continuous5s hold."""', '"""One freshcanonical BFM250, learned1019, terminal300 and250 lifecycle."""')
ast.parse(s)
(SNAP/'evaluate_phase_student.py').write_text(s,encoding='utf-8')
print(SNAP/'evaluate_phase_student.py')
