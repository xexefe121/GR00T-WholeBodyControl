"""Explicit saved FP64 release, runtime and actual output identities; no task imports."""
import hashlib,json,sys
from pathlib import Path
def local(v):
 s=str(v).replace('\\','/')
 return Path('/mnt/'+s[0].lower()+s[2:] if sys.platform!='win32' and len(s)>2 and s[1]==':' else s)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def release_paths(new):
 run=new/'direct_target_fp64_export_evaluation_v2';src=run/'source_draft_v1'
 p={'binding':run/'evaluation_binding.json','launch':run/'evaluation_process/launch_receipt.json','posthash':run/'evaluation_process/postrun_hashes.json',
 'actual_request':run/'nominal/request.json','process_exit':run/'evaluation_process/exit.json','process_raw':run/'evaluation_process/raw_exit.json',
 'release_source':Path(__file__),'semantic_original':Path(__file__).with_name('audit_original.py'),'source_derivation':Path(__file__).with_name('source_derivation.json'),
 'export_review':new/'direct_target_fp64_export_final_review_v1/review.json','root_export_audit':new/'direct_target_fp64_root_audit_v1/results_v1/report.json',
 'root_training_audit':new/'direct_target_continuation_failure_audit_v1/results_v1/report.json','source_review':new/'direct_target_fp64_export_evaluation_source_review_v1/review.json',
 'canonical_review':new/'direct_target_fp64_canonical_prelaunch_review_v1/review.json'}
 p['process_start']=run/'evaluation_process/start.json';p['process_child']=run/'evaluation_process/child.json'
 binding=read(p['binding'])
 for role in ('head','checkpoint','fit_report','source_head','normalization','training_manifest','training_request','export_report','export_request','export_manifest'):
  p['subject_'+role]=local(binding[role]['path'])
 p['export_owner']=local(binding['export_owner_completion']['path'])
 for file in src.rglob('*.py'):p['actual_source_'+file.relative_to(src).as_posix()]=file
 return p
def check_release(p):
 b=read(p['binding']);launch=read(p['launch']);owner=read(p['owner']);post=read(p['posthash']);actual=read(p['actual_request']);report=read(p['report']);physics=read(p['physics'])
 assert sha(p['binding'])=='7375bca43395397e7f730246a1ab707013133b20ff2e8297fc6be21c4f70f284'
 assert sha(p['launch'])=='b966cbb2de2203a8776a1928be6162212ba955099988dfc3efe571010196ee8a'
 assert launch['binding_sha256']==sha(p['binding']) and post==launch['input_hashes'] and len(post)==5227
 for item in b['input_files']:assert post[item['path']]==item['sha256']
 assert owner['owner_completion_accounting_passed'] is True and owner['pins_exact']==5227 and owner['mode']=='evaluation'
 assert owner['process_absence']['wrapper_absent'] is True and owner['process_absence']['child_absent'] is True
 start=read(p['process_start']);child=read(p['process_child']);ex=read(p['process_exit']);raw=read(p['process_raw'])
 assert owner['process_absence']['wrapper_pid']==start['wrapper_pid']==child['wrapper_pid']
 assert owner['process_absence']['child_pid']==child['child_pid']
 assert ex['exit_code']==ex['raw_child_exit_code']==owner['diagnostic_exit_code']==2
 assert owner['diagnostic_passed'] is False and raw['known'] is True and raw['raw_python_exit_code']==owner['raw_python_exit_code']==0
 assert ex['error'] is None and ex['all_postrun_hashes_exact'] is True
 assert owner['output_hashes'][str(p['trace']).replace('/mnt/e/','E:/')]==sha(p['trace'])
 assert actual['head_sha256']==b['head']['sha256']=='147a710ac8d6fd6de592c93f3ca14af4f7fcf156b970bbc3ab7d5d86f3586501'
 assert actual['evaluation_binding_sha256']==sha(p['binding']) and report['request_sha256']==sha(p['actual_request'])
 assert report['requested_controls']==1569 and report['attempted_controls']==292 and report['physics_steps']==2916 and report['full_segment_completed'] is False
 assert report['trace_sha256']==sha(p['trace']) and read(p['hold'])['attempted_controls']==0
 assert physics['recorded_trace_reproduced_through_last_sample'] is True and physics['compared_physics_steps']==2916
 assert all(physics['original_trace_comparison'].values()) and physics['private_replay_steps_beyond_recorded_prefix']==0
 assert physics['intended_segment_controls']==1569 and physics['requested_segment_completed'] is False and physics['feasible'] is False
 assert physics['input_hashes'][str(p['trace'])]==sha(p['trace'])
 assert physics['first_failure']['control_index']==291 and physics['first_failure']['substep']==6
 assert actual['physical_hz']==500 and actual['control_hz']==50 and actual['physical_statewrites_after_initialization']==0
 assert actual['learned_BFM_calls']==0 and actual['clock_foundation_connected'] is False and actual['hardware_authorized'] is False
 expected={role:b[role]['sha256'] for role in ('head','checkpoint','fit_report','source_head','normalization','export_report','export_request','export_manifest')}
 for role,digest in expected.items():assert sha(p['subject_'+role])==digest
 final=read(p['export_review']);root=read(p['root_export_audit']);export_owner=read(p['export_owner'])
 assert final['export_review_pass'] is True and root['passed'] is True and root['evidence_audit_passed'] is True and root['export_qualified'] is True
 assert final['direct_subject_sha256']['root_export_audit']==sha(p['root_export_audit']) and final['direct_subject_sha256']['export_owner_completion']==sha(p['export_owner'])
 for role,digest in expected.items():assert final['direct_subject_sha256'][role]==root['direct_subject_sha256'][role]==export_owner['direct_subject_sha256'][role]==digest
 fit=read(p['subject_fit_report']);assert fit['completed'] is False and fit['numerical_gate_passed'] is False and fit['export_parity_passed'] is False and fit['optimization_completed'] is True
 source_review=read(p['source_review']);frozen=read(p['frozen'])
 for key,path in p.items():
  if key.startswith('actual_source_'):
   rel=key[len('actual_source_'):];assert sha(path)==frozen['source_sha256'][rel]
 witness=read(p['witness_report']);assert witness['head_sha256']==expected['head'] and witness['physics_steps']==witness['BFM_inference_calls']==0
 assert witness['runtime_binary_sha256'] in post.values() and witness['onnxruntime_version']=='1.23.2'
 assert witness['pinned_WSL_runtime'] is True and witness['intra_op_threads']==witness['inter_op_threads']==1
 assert actual['ordinary_final_step']==55000
 return dict(passed=True,direct_subject_sha256=expected,root_export_audit_sha256=sha(p['root_export_audit']),root_physics_report_sha256=sha(p['physics']),
  runtime_launch_pins=5227,all_binding_pins_equal_completed_owner_posthash_ledger=True,actual_source_files=31,original_FP32_failed_flags_preserved=True,
  new_FP64_export_previously_qualified=True,original_lifecycle_completed=False,model_calls=0,native_steps=0)
