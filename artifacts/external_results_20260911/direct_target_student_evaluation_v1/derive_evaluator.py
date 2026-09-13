"""Derive instrumentation/driver while preserving strict native helper and PD AST."""
from pathlib import Path
import ast
import json
import hashlib
BASE=Path(__file__).resolve().parent;SOURCE=BASE/'source_draft_v1'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
evidence=(SOURCE/'original_proposal_evidence.py').read_text()
evidence=evidence.replace('features=(1069,)','features=(1000,),normalized_head=(23,)')
evidence=evidence.replace("'actual_normalized_action','control_history_before'","'actual_normalized_action','normalized_head','control_history_before'")
evidence=evidence.replace("'base_target','delta','features','raw_proposal'","'base_target','delta','features','normalized_head','raw_proposal'")
evidence=evidence.replace('content=dict(self.current)', "content=dict(self.current)\n        content.update({'runtime_'+k:np.asarray(v).copy() for k,v in self.runtime.context.items()})\n        content.update({'count_'+k:np.asarray(v,np.int64) for k,v in self.runtime.counts.items()})\n        content['uncommitted_proposal']=np.asarray(self.runtime.pending is not None)\n        content['forbidden_inference_calls']=np.asarray(self.runtime.forbidden_calls,np.int64)")
evidence=evidence.replace("# Both values are diagnostics. Neither is fed into the controller's raw\n        # combined-action prior or its history, nor substituted for its target.",
    "# Evidence reconstructs the output. Runtime independently owns its phase\n        # action convention: actual-target inverse only in the learned phase.")
evidence=evidence.replace('Observation-only proposal diagnostics around the unchanged runtime.',
                          'Proposal diagnostics and schema for the direct absolute-target adapter.')
(SOURCE/'proposal_evidence.py').write_text(evidence)
text=(SOURCE/'original_evaluator.py').read_text()
node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='zero_parity')
text=text.replace(ast.get_source_segment(text,node),'')
edits=[
 ('One selected canonical final75000 evaluation; original raw-history controller.',
  'Future selected canonical direct-target evaluation; source preparation only.'),
 ('from student_linear_runtime import *','from direct_runtime import *'),
 ('def source_metrics(native,arrays,motion,original29,audit):','def source_metrics(native,arrays,motion,original29,audit,timeline):'),
 ("selected=complete&(arrays['global_control']>=350)&(arrays['global_control']<1169)",
  "source_phase=next(p for p in timeline['phases'] if p['name']=='source_motion')\n    selected=complete&(arrays['global_control']>=source_phase['control_start'])&(arrays['global_control']<source_phase['control_stop'])"),
 ('requested_source_controls=819,',"requested_source_controls=source_phase['requested_controls'],"),
 ("frozen=assert_frozen();assert mujoco.__version__=='3.2.3'", "assert mujoco.__version__=='3.2.3'"),
 ("assert fit['ordinary_final_step']==75000 and fit['additional_updates']==5000", "assert fit['features']==1000 and fit['head_output']=='normalized_target'"),
 ('runtime=LinearStudentRuntime(native,c,original,motion,original29,HEAD)',
  "runtime=DirectStudentRuntime.from_native(native,c,original,motion,original29,HEAD,evaluation_binding['span'],timeline)\n    assert runtime.ort_binary_sha256==str(evaluation_binding['first_export']['runtime_binary_sha256'].item())"),
 ("switch=next(p['control_start'] for p in timeline['phases'] if p['name']=='returned_standing')", "switch=runtime.phases.terminal_start"),
 ("assert switch==1269 and len(motion['joint_pos'])-11==1569", "assert len(motion['joint_pos'])-runtime.phases.frame_offset==runtime.phases.total"),
 ("controller='originalBFM yaw2 disabledresidual0..249; linearspan residual250..1268; originalBFM yaw4 terminal1269 onward'",
  "controller='originalBFM yaw2 startup; direct normalized absolute targets acquisition/source/return; originalBFM yaw4 terminal'"),
 ("ordinary_final_step=75000,evaluation_binding_sha256", "ordinary_final_step=fit.get('ordinary_final_step'),evaluation_binding_sha256"),
 ("proposal_diagnostics='raw base+delta and inverse actual target are recorded only; original raw combined-action history remains unchanged'",
  "proposal_diagnostics='learned base_target is default_q, delta is span64*normalized_head64; BFM phases retain original base and zero delta; actual clipped target inverse enters learned history'"),
 ("frozen_sources_sha256=sha(FROZEN_RECEIPT)","frozen_sources_sha256=evaluation_binding['binding_sha256']"),
 ("prior_action='rawBFM actor*5 initial250; preclipcombinedBFM+linearresidual250..1268; rawBFM actor*5 terminal; unclippedhistory'",
  "prior_action='original rawBFM startup and terminal; inverse actual clipped native target during learned phase; no +/-5 history clamp'"),
 ("initial_BFM_controls=250,learned_controls=[250,1269]", "initial_BFM_controls=runtime.phases.learned_start,learned_controls=[runtime.phases.learned_start,runtime.phases.terminal_start]"),
 ("controller_mode_map={'0':'initial_BFM','1':'learned_residual','2':'terminal_BFM'}", "controller_mode_map={'0':'initial_BFM','1':'direct_absolute_target','2':'terminal_BFM'},learned_BFM_calls=0,clock_foundation_connected=False"),
 ("pairs=[(key,np.asarray(trace[key])[:len(baseline[key])],baseline[key]) for key in fields]",
  "pairs=[(key,np.asarray(trace[key])[:len(baseline[key])],baseline[key][:,np.r_[0:52,75:1023]] if key=='features' else baseline[key]) for key in fields]"),
 ("pairs=[(key,proposed[key],labels[key][0]) for key in ('features','base_target','previous_action','history','state')]",
  "pairs=[(key,proposed[key],labels[key][0]) for key in ('previous_action','history','state')]\n        pairs.append(('features1000',proposed['features'],labels['features'][0,np.r_[0:52,75:1023]]))"),
 ("'features',proposed['features'],witness['features']),('onnx_delta',proposed['delta'],witness['onnx_delta'])",
  "'features',proposed['features'],witness['features']),('normalized_target',proposed['normalized_head'],witness['normalized_target']),('preclamp_raw',proposed['base_target']+proposed['delta'],witness['raw_proposal']),('target',proposed['target'],witness['target'])"),
 ("trace['raw_proposal']=[];trace['actual_normalized_action']=[]", "trace['raw_proposal']=[];trace['actual_normalized_action']=[];trace['normalized_head']=[]"),
 ('if control==250:verify_generated_prefix250(trace)', 'if control==runtime.phases.learned_start:verify_generated_prefix250(trace)'),
 ('proposed=runtime.propose(control,data.qpos,data.qvel,terminal=control>=switch,disable_head=control<250)',
  'proposed=runtime.propose(control,data.qpos,data.qvel)'),
 ('if control==250:verify_actual_query250(proposed)',
  'if control==runtime.phases.learned_start:verify_actual_query250(proposed)\n                runtime.commit(proposed)'),
 ("frame=min(control+11,len(motion['joint_pos'])-1)", "frame=min(control+runtime.phases.frame_offset,len(motion['joint_pos'])-1)"),
 ('controller_mode=2 if control>=switch else (0 if control<250 else 1)', 'controller_mode=runtime.phases.mode(control)'),
 ('actual_normalized_action_used_for_history=False,','actual_normalized_action_used_for_history=True,actual_action_feedback_scope=\'learned phase only\',inference_counts=dict(runtime.counts),forbidden_inference_calls=runtime.forbidden_calls,'),
 ('source_metrics(native,arrays,motion,original29,audit) if', 'source_metrics(native,arrays,motion,original29,audit,timeline) if'),
 ('result,expected=segment(output,0,1569,expected)',
  "initial_integration=get_state(native,data)\n    assert initial_integration.shape==(291,)\n    np.savez_compressed(BASE/'canonical_initial_snapshot.npz',integration=initial_integration,qpos=data.qpos,qvel=data.qvel,warning_counts=data.warning.number,warning_lastinfo=data.warning.lastinfo)\n    record_parity('canonical_initial_full291_parity.json',[('integration',initial_integration,baseline['initial_integration']),('qpos',data.qpos,baseline['qpos'][0]),('qvel',data.qvel,baseline['qvel'][0]),('warning_counts',data.warning.number,baseline['physics_warning_counts'][0]),('warning_lastinfo',data.warning.lastinfo,baseline['physics_warning_lastinfo'][0])])\n    result,expected=segment(output,0,runtime.phases.total,expected)"),
 ('segment(extension,1569,250,expected)', 'segment(extension,runtime.phases.total,250,expected)'),
 ('one_fixed_physical_response_fit=True,ordinary_final_global_step=75000',
  "direct_absolute_target_trial=True,ordinary_final_global_step=fit.get('ordinary_final_step')"),
 ("(BASE/'pilot_outcome.json').write_text", "final_binding=require_ready(BASE)\n    assert final_binding['binding_sha256']==evaluation_binding['binding_sha256']\n    assert runtime.forbidden_calls==0\n    (BASE/'pilot_outcome.json').write_text")]
for old,new in edits:
    assert text.count(old)==1,(old,text.count(old))
    text=text.replace(old,new)
ast.parse(text)
(SOURCE/'evaluate_direct_target_student.py').write_text(text)
original=ast.parse((SOURCE/'original_evaluator.py').read_text());derived=ast.parse(text)
checks={}
for name in ('get_state','assess'):
    left=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name==name)
    right=next(n for n in derived.body if isinstance(n,ast.FunctionDef) and n.name==name)
    checks[name]=ast.dump(left,include_attributes=False)==ast.dump(right,include_attributes=False)
    assert checks[name]
with (BASE/'evaluator_derivation_v2.json').open('x') as stream:json.dump(dict(
    original_sha256=sha(SOURCE/'original_evaluator.py'),derived_sha256=sha(SOURCE/'evaluate_direct_target_student.py'),
    edits=edits,unchanged_AST=checks,unused_zero_head_physics_function_removed=True,model_calls=0,native_steps=0),stream,indent=2)
print('Direct evaluator derived; strict assess/get_state unchanged.')
