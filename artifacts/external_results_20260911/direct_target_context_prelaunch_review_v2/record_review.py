"""Narrow pre-Python launcher repair review; no launch or task calculations."""
from pathlib import Path
import hashlib,json,ast
ROOT=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911');BASE=ROOT/'direct_target_causal_context_study_v1';DEST=Path(__file__).parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
known={'training_request.json':'284e1e6a69617b6f3f39ab877b92de4e1d426f96bf6abd983fe8d6a163a70c6c',
 'training_frozen_inputs.json':'4e292ff3005a330318f99c08aa47241d7b18a752a807befa72e41daae2a30bad',
 'run_fit_durable_v2.ps1':'6f114a44e4958598271379ce57be5f553e28ff2c078bfef3da8d3c7759627c5d'}
for name,digest in known.items():assert sha(BASE/name)==digest,name
prior_path=ROOT/'direct_target_context_prelaunch_review_v1/review.json';assert sha(prior_path)=='934a0211d9fc55162c867e3b9fcb49b5bdf61998478d1010256a99b6f75ed319';prior=read(prior_path)
old_frozen=BASE/'failed_launch_attempt_v1/training_frozen_inputs.json'
assert sha(old_frozen)=='f56d91b56c30a18fb0cdc3f6ee657b8e281352fbb54e04064cd3c3ff1e03c515'
old=read(old_frozen);frozen=read(BASE/'training_frozen_inputs.json');request=read(BASE/'training_request.json')
assert frozen['source_sha256']==old['source_sha256']==prior['source_sha256'] and len(frozen['source_sha256'])==19
assert all(frozen['input_sha256'].get(p)==d for p,d in old['input_sha256'].items())
added={p:d for p,d in frozen['input_sha256'].items() if p not in old['input_sha256']}
assert len(added)==8 and len(frozen['input_sha256'])==266
for p,d in added.items():assert sha(p)==d,p
for name,digest in frozen['source_sha256'].items():assert sha(Path(frozen['source_directory'])/name)==digest,name
assert frozen['owner_checker']=='verify_completed_v3.py' and frozen['process_directory']=='fit_process_v2'
v1=(BASE/'run_fit_durable_v1.ps1').read_text();v2=(BASE/'run_fit_durable_v2.ps1').read_text()
expected=v1.replace("$processRoot=Join-Path $runRoot 'fit_process'","$processRoot=Join-Path $runRoot 'fit_process_v2'").replace('$request.coefficient -ne 1.8188207859141674)','$request.coefficient -ne 1.8188207859141674d)')
assert expected==v2
owner=(BASE/'verify_completed_v2.py').read_text();actual=(BASE/'verify_completed_v3.py').read_text()
assert owner.replace("process=BASE/'fit_process'","process=BASE/'fit_process_v2'").replace('owner_completion_verification_v2.json','owner_completion_verification_v3.json')==actual
ast.parse(actual)
test=read(BASE/'launcher_v2_test.json')
assert test['passed'] is True and test['PSVersion'].startswith('5.1.') and test['json_coefficient_type']=='System.Decimal'
assert all(test[k] is True for k in ('original_mixed_comparison_unequal','decimal_comparison_equal','actual_corrected_guard_passed','changed_coefficient_rejected'))
failure=read(BASE/'fit_process/exit.json')
assert failure['child_started'] is False and failure['child_pid'] is None and failure['raw_python_exit_code'] is None and failure['error']=='Unexpected request.'
assert not (BASE/'fit').exists() and not (BASE/'fit_process_v2/start.json').exists()
subjects={role:dict(path=(BASE/name).as_posix(),sha256=known[name]) for role,name in [('training_request','training_request.json'),('frozen_inputs','training_frozen_inputs.json'),('launcher','run_fit_durable_v2.ps1')]}
for role,p in [('prior_review',prior_path),('prior_frozen_receipt',old_frozen),('failed_zero_call_attempt',BASE/'fit_process/exit.json'),('actual_PS51_guard_test',BASE/'launcher_v2_test.json'),('completion_checker',BASE/'verify_completed_v3.py')]:subjects[role]=dict(path=p.as_posix(),sha256=sha(p))
result=dict(passed=True,prelaunch_review_pass=True,training_request_sha256=known['training_request.json'],frozen_receipt_sha256=known['training_frozen_inputs.json'],
 subjects=subjects,source_sha256=frozen['source_sha256'],additional_input_sha256=added,source_files_checked=19,new_input_pins_checked=8,inherited_unchanged_input_pins=258,
 conditions=['blinded','causal'],updates_per_condition=3000,ordinary_final_step=68000,selected_single_pair=True,
 prior_task_model_calls=0,prior_optimizer_updates=0,process_directory='fit_process_v2',
 scope=['Only the request coefficient comparison changes to the exactly matching PowerShell Decimal literal d suffix; neither tolerance nor Python coefficient changes.',
  'Actual PS5.1 full request guard test passes and rejects a distinct decimal coefficient. Process path changes to fresh fit_process_v2; original failure, receipt, clearance and logs preserved.',
  'Owner v3 differs only in actual process and output owner paths; same exact launch-map/count validation and18-role condition receipts.',
  'All19 trainer files unchanged and rehashed. All258 prior input hashes retained exactly in new receipt; eight additional metadata inputs hashed. Full original source/data review remains bound; no task calculations repeated.'],
 no_automatic_retry=True,controller_selected=False,model_calls=0,ORT_calls=0,gradient_calls=0,optimizer_updates=0,native_steps=0)
for name,digest in known.items():assert sha(BASE/name)==digest
with (DEST/'review.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(dict(path=(DEST/'review.json').as_posix(),sha256=sha(DEST/'review.json'),prelaunch_review_pass=True)))
