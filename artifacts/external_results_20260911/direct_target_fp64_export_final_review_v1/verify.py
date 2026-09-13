"""Consume independent saved export audit and actual owner completion; no inference."""
import argparse,hashlib,json
from pathlib import Path
N=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911');B=N/'direct_target_fp64_export_v1';FIT=N/'direct_target_continuation_v1'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--root-audit',type=Path,required=True);parser.add_argument('--root-audit-sha',required=True);parser.add_argument('--owner',type=Path,required=True);parser.add_argument('--owner-sha',required=True);a=parser.parse_args()
 subjects={};pins={}
 def bind(name,path,expected=None):
  path=Path(path);digest=sha(path)
  if expected is not None:assert digest==expected,(name,digest,expected)
  subjects[name]={'path':str(path),'sha256':digest};pins[str(path)]=digest
  return read(path) if path.suffix=='.json' else None
 report=bind('export_report',B/'export/report.json','395cc5122e3562e3629085110e8d9538c68a78875235832e557b8bebbe640da0')
 bind('head',B/'export/student_head_fp64.onnx','147a710ac8d6fd6de592c93f3ca14af4f7fcf156b970bbc3ab7d5d86f3586501')
 bind('checkpoint',FIT/'fit/student_head.pt','9ceef5099ebd154e08a1c2c3784c4e06021d464e607474548763665b24f8f9e7')
 fit=bind('fit_report',FIT/'fit/report.json','dc8d834e8b7ef193cb6212248e4c35ca2cd7dab3cfb839a826a3e7adf7d220e6')
 bind('source_head',FIT/'fit/student_head.onnx','f9b352ce4cedbbdd70c59f09b8a28080f7a696bfff464621719cc93e6c899797')
 bind('normalization',FIT/'fit/normalization.npz','1baf7a2918fe2322ad0ef9583a29902eb7c3d8f8d37000c6bb9fd502d6746079')
 request=bind('export_request',B/'export_request.json','f6d4bd17cf2231f010126052b9fa02428fd3e40225707e8df6354038b0e5bc68')
 frozen=bind('export_manifest',B/'export_frozen_inputs.json','64ce9b8d9fa4cb8e654aef84ddd19dd9089d088fe146a28a5167d4da7ab0e346')
 output=bind('output_manifest',B/'export/manifest.json','4efc740654d6641c2929ef164dcd99a2dd0186549a5deb8c864b1fd1413f7af9')
 root_training=bind('root_training_audit',N/'direct_target_continuation_failure_audit_v1/results_v1/report.json','0accc9eea0a90158dbcfe38bd85aa04ae90ea88ef224da7ea4f86837672d516c')
 training=bind('training_review',N/'direct_target_continuation_training_only_review_v1/review.json','eb693269af05d9c761e0de6fa7c96f92c4143d8a8cf9c3429dd4b4887bc35649')
 root=bind('root_export_audit',a.root_audit,a.root_audit_sha);owner=bind('export_owner_completion',a.owner,a.owner_sha)
 bind('root_export_source_review',N/'direct_target_fp64_root_audit_source_review_v1/review.json','0fafac1233229fe7e2e89a171c5a501fcccd3e9807ca06c8061cb35682a25071')
 ex=bind('process_exit',B/'export_process/exit.json');absence=bind('process_absence',B/'process_absence.json')
 clearance=bind('export_clearance',B/'export_clearance.json');prelaunch=bind('prelaunch_review',N/'direct_target_fp64_export_prelaunch_review_v1/review.json','030df83c854d0efc341c47783026708a72462672dfaaa37e5326db0503c14b9e')
 for key in ('passed','evidence_audit_passed','export_qualified'):assert root[key] is True
 assert root['canonical_evaluation_cleared'] is False and root['same_trained_weights'] is True and root['new_FP64_execution_semantics'] is True
 expected_keys=('head','checkpoint','fit_report','source_head','normalization','export_report','export_request','export_manifest')
 for key in expected_keys:
  assert root['direct_subject_sha256'][key]==owner['direct_subject_sha256'][key]==subjects[key]['sha256'],key
 assert root['direct_subject_sha256']['root_training_audit']==subjects['root_training_audit']['sha256']
 assert root['direct_subject_sha256']['output_manifest']==subjects['output_manifest']['sha256']
 assert root['graph']['graph_nodes']==20 and root['graph']['literal_arrays']==10
 assert root['graph']['same_source_weights'] is True and root['graph']['same_source_normalization'] is True
 assert root['numerical']['parity_passed'] is True and root['numerical']['drift_arrays_verified']==27 and root['numerical']['all_call_partitions_exact'] is True
 assert owner['owner_verification_passed'] is True and owner['raw_exit_known'] is True
 assert owner['raw_python_exit_code']==owner['exit_code']==ex['raw_python_exit_code']==ex['exit_code']==0
 assert owner['all_postrun_pins_exact'] is True and owner['processes_absent'] is True and ex['exit_known'] is True and ex['error'] is None
 assert owner['exit_sha256']==subjects['process_exit']['sha256'] and owner['process_absence_sha256']==subjects['process_absence']['sha256']
 assert absence['any_present'] is False and absence['observed']==[] and set(absence['expected_pids'])=={ex['wrapper_pid'],ex['child_pid']}
 assert ex['all_postrun_pins_exact'] is True and owner['frozen_receipt_sha256']==subjects['export_manifest']['sha256']
 assert owner['clearance_sha256']==subjects['export_clearance']['sha256']==ex['clearance_sha256']
 assert clearance['review_sha256']==subjects['prelaunch_review']['sha256'] and clearance['request_sha256']==subjects['export_request']['sha256']
 for key in ('completed','validation_completed','numerical_gate_passed','export_parity_passed','all_frozen_inputs_unchanged'):assert report[key] is True
 assert report['max_preclip_error_rad']==root['numerical']['maximum_preclamp_rad']==owner['parity']['maximum_preclamp_rad']==0.0
 assert report['parity_tolerance_rad']==owner['parity']['tolerance_rad']==1e-5 and owner['parity']['nonfinite_comparisons']==[]
 assert len(owner['parity']['comparisons'])==9 and all(v==0.0 for v in owner['parity']['comparisons'].values())
 assert report['counters']==root['numerical']['counters']==owner['counters']
 assert set(report['counters'])=={'CPU64','GPU64','ORT64'}
 for counts in report['counters'].values():
  assert all(counts[k]==601 for k in ('calls_attempted','calls_returned','calls_synchronized','calls_verified'))
  assert all(counts[k]==153580 for k in ('rows_attempted','rows_returned','rows_verified'))
 assert fit['optimization_completed'] is True and fit['completed'] is False and fit['numerical_gate_passed'] is False and fit['export_parity_passed'] is False
 assert training['training_review_pass'] is True and training['dataset_review_pass'] is True and training['export_review_pass'] is False
 for key in ('task_model_calls','ORT_calls','optimizer_updates','native_steps'):assert root[key]==0
 for key in ('optimizer_updates','BFM_calls','native_steps','trace_forward_calls'):assert report[key]==0
 for item in output['files'].values():assert sha(B/'export'/item['path'])==item['sha256']
 for name,digest in frozen['source_sha256'].items():assert sha(Path(frozen['source_directory'])/name)==digest
 for name,digest in pins.items():assert sha(name)==digest
 r={'kind':'qualified_same55000_weights_new_FP64_export_final_review','passed':True,'export_review_pass':True,
 'training_review_pass':True,'dataset_review_pass':True,'original_FP32_export_qualified':False,'new_FP64_export_qualified':True,
 'canonical_evaluation_cleared':False,'actual_witness_launch_cleared':False,'actual_controller_launch_cleared':False,
 'subjects':subjects,'direct_subject_sha256':{k:v['sha256'] for k,v in subjects.items()},'input_sha256':pins,
 'ordinary_final_step':55000,'optimizer_updates_for_export':0,'maximum_preclamp_error_rad':0.0,'unchanged_tolerance_rad':1e-5,
 'cross_backend_comparisons':9,'fixed_corpus_rows':153580,'Torch_calls':1202,'Torch_rows':307160,'ORT_calls':601,
 'root_saved_audit_checks':root['checks'],'exact_graph_nodes':20,'exact_promoted_arrays':10,'full_drift_arrays':27,
 'scope':['Independent root audit verifies all saved parity/27 drift arrays/clipping metrics/full objectives,1803 exact graph call partitions and exact original parameters in the20-node promoted graph.',
 'Original failed FP32 export and exit1 remain preserved. Same trained weights use new FP64 internal arithmetic with public f32 inputs/outputs, separately qualified at unchanged1e-5 rad.',
 'Actual owner exit0, retained process identity, completed prefix counters and all output hashes verified. Reviewer makes no task model/native calls.',
 'Zero measured preclamp cross-backend target difference on selected saved inputs does not prove arbitrary-input bit identity, connected stability, timing or hardware readiness.',
 'Root-selected one WSL batch1 witness and later canonical simulation require their actual source/binding/launcher reviews.'],
 'reviewer_model_calls':0,'reviewer_native_steps':0,'reviewer_optimizer_updates':0,'findings':[]}
 out=Path(__file__).parent/'review.json'
 with out.open('x') as file:json.dump(r,file,indent=2);file.write('\n')
 print(json.dumps({'export_review_pass':True,'review_sha256':sha(out)}))
if __name__=='__main__':main()
