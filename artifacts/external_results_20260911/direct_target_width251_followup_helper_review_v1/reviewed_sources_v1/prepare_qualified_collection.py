"""Prepare a collection-only request after completed original independent audits.

JSON and hash reads only. Does not read task arrays or execute collection.
"""
import argparse,hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;NEW=HERE.parent
ROOT=Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof')
COL=NEW/'direct_target_width251_collection_v1';SOURCE=COL/'source_prepared_v1'
sys.path.insert(0,str(SOURCE))
from qualification_gate import validate_reports,QUALIFIED_ROLES,REPORT_ROLES
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);ap.add_argument('--owner-sha',required=True)
 a=ap.parse_args();owner=a.base/'owner_completion.json'
 assert sha(owner)==a.owner_sha
 recovery=read(a.base/'execution_request.json')
 source_review=NEW/'direct_target_width251_collection_root_review_v1/review.json'
 assert sha(source_review)=='8b1aeea19361feb5bde70625b003b0f7050b1b858655786b7b5c776a13c9b285'
 norm=NEW/'direct_target_causal_response_balanced_student_v2/fit/shared/normalization.npz'
 assert sha(norm)=='e914676e8f506ceb9ee0811fb058d75bb43285dac22b6b478c828e34c4acfdb9'
 bundle=ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1'
 paths=dict(recovery_request=a.base/'execution_request.json',owner=owner,
  main_trace=a.base/'nominal/trace.npz',main_report=a.base/'nominal/report.json',
  hold_trace=a.base/'post_lifecycle_hold_5s/trace.npz',hold_report=a.base/'post_lifecycle_hold_5s/report.json',
  main_physics=NEW/'direct_target_width251_expert_main_physics_v1/report.json',
  main_intent=NEW/'direct_target_width251_expert_main_intent_v1/report.json',
  hold_physics=NEW/'direct_target_width251_expert_hold_physics_v1/report.json',
  hold_intent=NEW/'direct_target_width251_expert_hold_intent_v1/report.json',
  boundary_snapshot=a.base/'inputs/precontrol251.npz',
  boundary_review=Path(recovery['subjects']['boundary_review']['path']),selection_receipt=a.base/'inputs/selection_receipt.json',
  plan_records=a.base/'nominal/plans.json',
  motion=NEW.parent/'sonic23_teleop_six_hour_20260910/mjbatch_intent_floor_inputs_v1/walk003/reference.npz',
  original29=bundle/'walk003/original29.npz',contract=bundle/'contract.json',normalization=norm,
  core=a.base/'source_snapshot_v1/gear_sonic/utils/g1_true23_mjbatch_ilqr_core.py',source_review=source_review)
 assert set(paths)==set(QUALIFIED_ROLES)
 subjects={k:dict(path=p.as_posix(),sha256=sha(p)) for k,p in paths.items()}
 plans=[]
 for control in range(251,1269,5):
  p=a.base/'nominal/plans'/('plan_%05d.npz'%control)
  plans.append(dict(control=control,path=p.as_posix(),sha256=sha(p)))
 assert len(plans)==204
 source_map={p.name:sha(p) for p in SOURCE.glob('*.py')}
 assert source_map==read(source_review)['source_sha256']
 qualification=dict(root_authorized_extraction=True,model_fitting_authorized=False,control_start=251,
  control_stop_exclusive=1269,rows=1018,subjects=subjects,plans=plans,
  fresh_student_state_queries=1,connected_expert_rows_after_first=1017,
  source_sha256=sha(__file__),actual_collection_executed=False)
 reports={role:read(paths[role]) for role in (*REPORT_ROLES,'owner','main_report','hold_report','recovery_request',
  'source_review','boundary_review','selection_receipt')}
 reports['qualification']=qualification
 validate_reports(subjects,reports)
 # Make the request only after every original complete-scope gate has passed.
 packet=COL/'actual_v1';assert not packet.exists(),'Preserve prior actual collection packet'
 packet.mkdir()
 q=packet/'qualification.json'
 with q.open('x',encoding='utf-8') as f:json.dump(qualification,f,indent=2);f.write('\n')
 subjects=dict(subjects,qualification=dict(path=q.as_posix(),sha256=sha(q)))
 request=dict(root_selected_collection=True,model_fitting_authorized=False,subjects=subjects,plans=plans,
  source_sha256=source_map,output=(packet/'results_v1').as_posix(),root_preparation_sha256=sha(__file__))
 p=packet/'request.json'
 with p.open('x',encoding='utf-8') as f:json.dump(request,f,indent=2);f.write('\n')
 print(json.dumps(dict(prepared=True,qualification_sha256=sha(q),request_sha256=sha(p),request=p.as_posix(),
  sources=len(source_map),rows=1018,plans=204,actual_collection_executed=False)))
if __name__=='__main__':main()
