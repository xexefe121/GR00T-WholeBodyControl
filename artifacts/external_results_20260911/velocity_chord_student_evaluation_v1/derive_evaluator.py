"""Artifact-local reporting/gating edits to the preserved original evaluator."""
import ast
import hashlib
import json
from pathlib import Path

BASE=Path(__file__).parent
SOURCE=BASE/'source_snapshot_v1'
original=SOURCE/'evaluate_phase_student.py'
text=original.read_text()
changes=[]


def replace(old,new):
    global text
    if text.count(old)!=1:raise ValueError('Evaluator substitution not unique: '+old)
    text=text.replace(old,new)
    changes.append(dict(old=old,new=new))


replace('"""One freshcanonical BFM250, learned1019, terminal300 and250 lifecycle."""',
        '"""One selected canonical final70000 evaluation; original raw-history controller."""')
replace('from quiet_metrics import standing_windows,quiet_diagnostic,yaw',
        'from quiet_metrics import standing_windows,quiet_diagnostic,yaw\nfrom evaluation_gate import require_ready\nfrom proposal_evidence import ProposalEvidence,exact,trace_arrays')
replace("    frozen=assert_frozen();assert mujoco.__version__=='3.2.3'",
        "    evaluation_binding=require_ready(BASE)\n    HEAD=evaluation_binding['head']\n    frozen=assert_frozen();assert mujoco.__version__=='3.2.3'")
replace("    fit=json.loads((BASE/'fit/report.json').read_text())\n    assert sha(BASE/'fit/student_head.onnx')==fit['checkpoints']['student_head.onnx']\n    assert fit['steps']==65000 and fit['additional_updates']==5000\n    assert fit['full_objective_improved'] and fit['export_parity_passed'] and fit['rollout_numerical_prerequisites_pass']",
        "    fit=evaluation_binding['fit_report']\n    assert fit['ordinary_final_step']==70000 and fit['additional_updates']==5000")
replace("    runtime=LinearStudentRuntime(native,c,original,motion,original29,BASE/'fit/student_head.onnx')",
        "    runtime=LinearStudentRuntime(native,c,original,motion,original29,HEAD)\n    evidence=ProposalEvidence(runtime,c)")
replace("        head_sha256=sha(BASE/'fit/student_head.onnx'),baseline250_trace_sha256=sha(BASE.parent/'original_bfm_entry250_v1/entry250/trace.npz'),",
        "        head_sha256=sha(HEAD),baseline250_trace_sha256=sha(BASE.parent/'original_bfm_entry250_v1/entry250/trace.npz'),\n        ordinary_final_step=70000,evaluation_binding_sha256=evaluation_binding['binding_sha256'],\n        proposal_diagnostics='raw base+delta and inverse actual target are recorded only; original raw combined-action history remains unchanged',")
replace("checks={key:dict(bitexact=bool(np.array_equal(actual,expected)),actual_shape=list(np.shape(actual)),expected_shape=list(np.shape(expected)))",
        "checks={key:dict(bitexact=exact(actual,expected),actual_shape=list(np.shape(actual)),expected_shape=list(np.shape(expected)))")
replace("        record_parity('actual_query250_input_parity.json',pairs)",
        "        record_parity('actual_query250_input_parity.json',pairs)\n        witness=evaluation_binding['first_export']\n        record_parity('actual_query250_ownexport_output_parity.json',[(\n            'features',proposed['features'],witness['features']),('onnx_delta',proposed['delta'],witness['onnx_delta'])])")
replace("        trace=new_trace(data);trace['physics_expected_time'][0]=expected_time",
        "        trace=new_trace(data);trace['physics_expected_time'][0]=expected_time\n        trace['raw_proposal']=[];trace['actual_normalized_action']=[]")
replace("            trace['control_previous_action_before'].append(runtime.seed.previous_action.copy())",
        "            trace['control_previous_action_before'].append(runtime.seed.previous_action.copy())\n            evidence.begin(control,data.qpos,data.qvel,trace['control_integration_before'][-1],expected_time,data.warning.number,data.warning.lastinfo)")
replace("                proposed=runtime.propose(control,data.qpos,data.qvel,terminal=control>=switch,disable_head=control<250)",
        "                proposed=runtime.propose(control,data.qpos,data.qvel,terminal=control>=switch,disable_head=control<250)\n                evidence.accepted_proposal(proposed)")
replace("                    substep=0,time=float(data.time),exception_type=type(exc).__name__,message=str(exc))\n                break",
        "                    substep=0,time=float(data.time),exception_type=type(exc).__name__,message=str(exc))\n                evidence.preserve(dest/'rejected_precontrol.npz',repr(exc),data.qpos,data.qvel,get_state(native,data),expected_time,data.warning.number,data.warning.lastinfo)\n                break")
replace("                joint_error=data.qpos[7:]-motion['joint_pos'][frame],root_error=data.qpos[:3]-motion['body_pos_w'][frame,0],**proposed)",
        "                joint_error=data.qpos[7:]-motion['joint_pos'][frame],root_error=data.qpos[:3]-motion['body_pos_w'][frame,0],\n                raw_proposal=evidence.last_raw_proposal.copy(),actual_normalized_action=evidence.last_actual_action.copy(),**proposed)")
replace("            if failure:break\n        arrays={k:np.asarray(v) for k,v in trace.items()}",
        "            if failure:\n                evidence.preserve(dest/'strict_failure_state.npz',failure,data.qpos,data.qvel,get_state(native,data),expected_time,data.warning.number,data.warning.lastinfo)\n                break\n        arrays=trace_arrays(trace)")
replace("                mujoco.mj_step(native,data);expected_time+=.002;actual_substeps+=1",
        "                try:\n                    mujoco.mj_step(native,data)\n                except BaseException as exc:\n                    evidence.current['attempted_native_ctrl']=data.ctrl.copy()\n                    evidence.current['native_step_attempt_substep']=np.asarray(sub,np.int64)\n                    evidence.preserve(dest/'native_exception_state.npz',repr(exc),data.qpos,data.qvel,get_state(native,data),expected_time,data.warning.number,data.warning.lastinfo)\n                    np.savez_compressed(dest/'native_exception_completed_samples.npz',**{key:np.asarray(value) for key,value in trace.items()})\n                    (dest/'fatal_failure.json').write_text(json.dumps(dict(reason='native_step_exception',exception_type=type(exc).__name__,message=str(exc),global_control=control,attempted_substep=sub,completed_prior_substeps=actual_substeps,requested_controls=count,full_segment_completed=False,unclassified_attempted_step=True),indent=2)+'\\n')\n                    raise\n                expected_time+=.002;actual_substeps+=1")
replace("            attempted_controls=len(arrays['target']),partial_substeps=len(arrays['physics_torque'])%10,physics_steps=len(arrays['physics_torque']),",
        "            attempted_controls=len(arrays['target']),partial_substeps=len(arrays['physics_torque'])%10,physics_steps=len(arrays['physics_torque']),\n            attempted_precontrols=len(arrays['control_integration_before']),controller_recorded_controls=runtime.seed.recorded_controls,\n            raw_proposal_and_actual_normalized_action_recorded=True,actual_normalized_action_used_for_history=False,")
replace("one_fixed_phase_only_fit=True,ordinary_final_global_step=65000",
        "one_fixed_velocity_chord_fit=True,ordinary_final_global_step=70000")
replace("original lifecycle stopped at firststrictphysicalfailure",
        "original lifecycle stopped at first failure; saved failure identifies policy, parity or native limit")
oldtree,newtree=ast.parse(original.read_text()),ast.parse(text)
for name in ('get_state','assess','new_trace','source_metrics','zero_parity'):
    oldnode=next(n for n in oldtree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    newnode=next(n for n in newtree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    assert ast.dump(oldnode,include_attributes=False)==ast.dump(newnode,include_attributes=False),name
output=SOURCE/'evaluate_velocity_chord_student.py';output.write_text(text)
receipt=dict(original_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
    derived_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),substitutions=changes,
    strict_oracle_ast_unchanged=True,original_runtime_file_unchanged=True,
    draft_only=True,final_head_bound=False,physics_steps=0,inference_calls=0)
(BASE/'evaluator_derivation.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='substitutions'}))
