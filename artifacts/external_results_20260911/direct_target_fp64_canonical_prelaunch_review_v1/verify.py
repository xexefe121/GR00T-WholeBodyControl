"""Review one future bound canonical run and its actual completed witness; no inference."""
import argparse,hashlib,json,sys
from pathlib import Path
N=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911');B=N/'direct_target_fp64_export_evaluation_v2';S=B/'source_draft_v1';P=B/'evaluation_process';W=B/'witness_process'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--binding-sha',required=True);parser.add_argument('--launch-sha',required=True);parser.add_argument('--witness-owner-sha',required=True);a=parser.parse_args()
 bp=B/'evaluation_binding.json';rp=P/'launch_receipt.json';op=B/'witness_completion_verification.json'
 assert sha(bp)==a.binding_sha and sha(rp)==a.launch_sha and sha(op)==a.witness_owner_sha
 b=read(bp);r=read(rp);owner=read(op);wr=read(B/'head_witness/report.json')
 assert r['binding_sha256']==a.binding_sha and r['kind']=='one_selected_direct_target_evaluation'
 assert r['requested_main_controls']==1569 and r['conditional_hold_controls']==250 and r['expected_separate_head_calls']==0
 assert r['automatic_retry'] is False and r['hardware_authorized'] is False
 assert b['physics_authorized'] is True and b['requested_main_controls']==1569 and b['conditional_hold_controls']==250
 assert b['head']['sha256']=='147a710ac8d6fd6de592c93f3ca14af4f7fcf156b970bbc3ab7d5d86f3586501'
 assert b['reviews']['export']['sha256']=='81e7e86cee62c29b926a1f33ea096c1d5ea677fc83494c19cf9f2cbb039ffb0e'
 assert b['reviews']['source']['sha256']=='5b414d0260f21984f98e134134e7ab9e01a4465028929f607715973820f8354d'
 sys.path.insert(0,str(S));import evaluation_gate
 ready=evaluation_gate.require_ready(B)
 assert owner['owner_completion_accounting_passed'] is True and owner['diagnostic_passed'] is True and owner['mode']=='witness'
 assert owner['raw_python_exit_code']==owner['diagnostic_exit_code']==0 and owner['pins_exact']==5216
 assert wr['binding_sha256']=='48ab01a8238008ab806f00a3c2328399960040452762cc16965d1baae8a67319'
 assert wr['expected_head_calls']==wr['attempted_head_calls']==wr['returned_head_calls']==1
 assert wr['BFM_inference_calls']==wr['physics_steps']==0 and wr['head_sha256']==b['head']['sha256']
 for path,digest in owner['output_hashes'].items():assert sha(path)==digest
 start=read(W/'start.json');child=read(W/'child.json');ex=read(W/'exit.json');raw=read(W/'raw_exit.json');verdict=read(W/'diagnostic_verdict.json')
 assert owner['process_absence']['wrapper_pid']==start['wrapper_pid']==child['wrapper_pid']
 assert owner['process_absence']['child_pid']==child['child_pid'] and owner['process_absence']['wrapper_absent'] is True and owner['process_absence']['child_absent'] is True
 assert ex['error'] is None and ex['raw_child_exit_code']==ex['exit_code']==raw['raw_python_exit_code']==verdict['diagnostic_exit_code']==0
 assert raw['known'] is True and ex['all_postrun_hashes_exact'] is True
 old_receipt=read(W/'launch_receipt.json');assert read(W/'postrun_hashes.json')==old_receipt['input_hashes']
 for entry in b['input_files']:assert r['input_hashes'][entry['path']]==entry['sha256']
 for path,digest in r['input_hashes'].items():assert sha(path)==digest
 args=r['exact_wsl_arguments'];assert args[:6]==['-d','Ubuntu-22.04','--cd','/','--','bash']
 assert args[-1]=='/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/direct_target_fp64_export_evaluation_v2/source_draft_v1/evaluate_direct_target_student.py'
 assert args[-2]=='/mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python'
 for name in ('nominal','post_lifecycle_hold_5s','pilot_outcome.json'):assert not (B/name).exists()
 for name in ('start.json','child.json','started.lock','stdout.log','stderr.log','exit.json','raw_exit.json','launch_clearance.json'):assert not (P/name).exists()
 paths={'binding':bp,'launch_receipt':rp,'run':P/'run.ps1','durable':P/'run_durable.ps1','witness_owner':op,'witness_report':B/'head_witness/report.json','witness':B/'head_witness/witness.npz','export_review':N/'direct_target_fp64_export_final_review_v1/review.json'}
 subjects={name:{'path':str(path),'sha256':sha(path)} for name,path in paths.items()}
 result={'kind':'one_actual_same55000_FP64_canonical_prelaunch_review','passed':True,'prelaunch_review_pass':True,
  'binding_sha256':a.binding_sha,'launch_receipt_sha256':a.launch_sha,'subjects':subjects,
  'verified_launch_pin_count':len(r['input_hashes']),'witness_completed':True,'witness_head_calls':1,'witness_BFM_calls':0,'witness_native_steps':0,
  'requested_main_controls':1569,'conditional_continuous_hold_controls':250,'selected_canonical_launch_cleared':True,'behavioral_qualification':False,
  'checks':['Actual one-call WSL witness, raw/diagnostic/child/wrapper exit0, all5216 witness postpins and exact owner PID linkage verified.',
   'Actual final FP64 head/weights/norm, positive training-only/export/root evidence and frozen source roles match; original failed FP32 export remains preserved.',
   'Pure actual require_ready validates exact query2501000features, native output widening/clamp, first exported target and current WSL binary identity before learned control.',
   'Original strict2ms plant and1569-control lifecycle plus conditional continuous250 hold remain unchanged; startup BFM250 and first actual learned activation gates apply.',
   'All concrete canonical launch pins match; actual WSL/source arguments and hidden durable single-run wrapper preserve raw and diagnostic failure accounting. No canonical outputs/lock/process exist at review.',
   'This clears only the selected one simulation run; root independently audits resulting actual native samples and source/quiet qualification.'],
  'reviewer_model_calls':0,'reviewer_native_steps':0,'reviewer_optimizer_updates':0,'findings':[]}
 out=Path(__file__).parent/'review.json'
 with out.open('x') as file:json.dump(result,file,indent=2);file.write('\n')
 print(json.dumps({'passed':True,'review_sha256':sha(out),'pins':len(r['input_hashes'])}))
if __name__=='__main__':main()
