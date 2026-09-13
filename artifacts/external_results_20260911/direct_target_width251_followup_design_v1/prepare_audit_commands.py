"""Bind completed recovery outputs to existing independent audit commands.

This preparation reads JSON and hashes files only. It does not load archives,
create a native model, replay a controller or execute any audit.
"""
import argparse,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;NEW=HERE.parent
ROOT=Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof')
ART=ROOT/'artifacts/teleop_resume_20260911'
BUNDLE=ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def key(p):return str(p).replace('\\','/').casefold()
def linux(p):
 s=str(p).replace('\\','/')
 return '/mnt/'+s[0].lower()+s[2:] if len(s)>2 and s[1]==':' else s
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);ap.add_argument('--owner-sha',required=True);ap.add_argument('--output',type=Path,required=True)
 a=ap.parse_args();owner_path=a.base/'owner_completion.json';assert sha(owner_path)==a.owner_sha
 owner=read(owner_path)
 assert owner['completion_accounting_passed'] and owner['raw_exit_known'] and owner['processes_absent'] and owner['all_postrun_pins_exact']
 assert owner['accounting_uncertainty']==[]
 owned={key(p):h for p,h in owner['output_sha256'].items()}
 nominal=a.base/'nominal/trace.npz';nominal_report=a.base/'nominal/report.json'
 for p in (nominal,nominal_report):assert owned[key(p)]==sha(p)
 assert read(nominal_report)['requested_controls']==1569
 assert not a.output.exists(),'Preserve prior audit preparation'
 source=a.base/'source_snapshot_v1';oracle=source/'gear_sonic/utils/g1_true23_feasibility_referee.py'
 physics=ART/'audit_restored_native_segment.py';intent=ART/'inspect_recorded_intent.py'
 fixture=NEW/'walk003_canonical_initial_fixture_v1/initial_integration_state.npz'
 reference=NEW.parent/'sonic23_teleop_six_hour_20260910/mjbatch_intent_floor_inputs_v1/walk003/reference.npz'
 fixture_maker=NEW/'bfm250_expert_hold_fixture_v1/source.py'
 subjects={p.as_posix():sha(p) for p in (owner_path,nominal,nominal_report,oracle,physics,intent,fixture,reference,fixture_maker,
  BUNDLE/'manifest.json',BUNDLE/'contract.json',BUNDLE/'native_prepared.xml',BUNDLE/'prepared_model_arrays.npz',
  ROOT/'gear_sonic/utils/g1_true23_hand_frame_tasks.py',source/'gear_sonic/utils/g1_true23_mjbatch_mpc.py',Path(__file__))}
 def command(script,**kwargs):
  args=[linux(script)]
  for k,v in kwargs.items():args.extend(['--'+k.replace('_','-'),linux(v) if isinstance(v,Path) else str(v)])
  return args
 main_physics=NEW/'direct_target_width251_expert_main_physics_v1';main_intent=NEW/'direct_target_width251_expert_main_intent_v1'
 commands=[dict(stage='main_physics',arguments=command(physics,frozen_repo=source,oracle=oracle,bundle=BUNDLE,fixture=fixture,
  trace=nominal,output=main_physics,clip='walk003',requested_controls=1569),native_step_max=15690),
  dict(stage='main_intent',arguments=command(intent,bundle=BUNDLE,trace=nominal,reference=reference,
  physical_audit=main_physics/'report.json',output=main_intent,clip='walk003',requested_controls=1569),native_step_max=0)]
 hold=a.base/'post_lifecycle_hold_5s/trace.npz';hold_report=a.base/'post_lifecycle_hold_5s/report.json'
 if key(hold) in owned:
  assert key(hold_report) in owned,'Recorded conditional hold requires its owner-bound report'
  assert read(nominal_report)['full_segment_completed'] is True,'Conditional hold requires the full original main'
  for p in (hold,hold_report):assert owned[key(p)]==sha(p);subjects[p.as_posix()]=sha(p)
  assert read(hold_report)['requested_controls']==250
  hold_fixture=NEW/'direct_target_width251_expert_hold_fixture_v1';hold_physics=NEW/'direct_target_width251_expert_hold_physics_v1'
  commands.extend([dict(stage='hold_fixture',arguments=command(fixture_maker,frozen_repo=source,bundle=BUNDLE,archive=hold,
   matching_trace=nominal,output=hold_fixture,clip='walk003'),native_step_max=0),
   dict(stage='hold_physics',arguments=command(physics,frozen_repo=source,oracle=oracle,bundle=BUNDLE,
    fixture=hold_fixture/'initial_integration_state.npz',trace=hold,output=hold_physics,clip='walk003',requested_controls=250),native_step_max=2500),
   dict(stage='hold_intent',arguments=command(intent,bundle=BUNDLE,trace=hold,reference=reference,
    physical_audit=hold_physics/'report.json',output=NEW/'direct_target_width251_expert_hold_intent_v1',clip='walk003',
    requested_controls=250,global_start=1569,preceding_trace=nominal),native_step_max=0)])
 else:
  assert owner['requested_recovery_completed'] is False,'Completed recovery must own both hold outputs'
  if key(hold_report) in owned:
   assert owned[key(hold_report)]==sha(hold_report);subjects[hold_report.as_posix()]=sha(hold_report)
   skipped=read(hold_report)
   assert skipped['requested_controls']==250 and skipped['full_segment_completed'] is False
   assert skipped.get('attempted_controls')==0 and skipped.get('not_run_reason'),'Unrecorded hold is not a skipped hold'
 record=dict(preparation_pass=True,input_sha256=subjects,commands=commands,actual_audits_executed=False,
  python='/mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python',
  environment=dict(PYTHONPATH=linux(ROOT),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'),
  original_requested_scope_unchanged=True,partial_main_replay_may_finish_last_recorded_control=True,
  beyond_recorded_steps_must_be_read_from_audit_report=True,root_qualification_pending=True,model_training_authorized=False)
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('x',encoding='utf-8') as f:json.dump(record,f,indent=2);f.write('\n')
 print(json.dumps(dict(prepared=True,sha256=sha(a.output),stages=[v['stage'] for v in commands])))
if __name__=='__main__':main()
