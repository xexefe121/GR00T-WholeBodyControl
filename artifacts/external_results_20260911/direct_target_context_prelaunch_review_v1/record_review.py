"""Bind final paired fit source/input/launcher review. No task calls or launch."""
from pathlib import Path
import json,hashlib,ast
ROOT=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911')
BASE=ROOT/'direct_target_causal_context_study_v1';DEST=Path(__file__).parent
KNOWN={'training_request.json':'284e1e6a69617b6f3f39ab877b92de4e1d426f96bf6abd983fe8d6a163a70c6c',
 'training_frozen_inputs.json':'f56d91b56c30a18fb0cdc3f6ee657b8e281352fbb54e04064cd3c3ff1e03c515',
 'run_fit_durable_v1.ps1':'444b56956e5963298af53eb0294c6373c3f5283c26dab0ce486d65d6aa7b0a19'}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
for name,digest in KNOWN.items():assert sha(BASE/name)==digest,name
request=read(BASE/'training_request.json');frozen=read(BASE/'training_frozen_inputs.json');draft=read(BASE/'training_clearance_draft.json')
assert frozen['training_request_sha256']==KNOWN['training_request.json'] and draft['frozen_receipt_sha256']==KNOWN['training_frozen_inputs.json']
source_review_path=ROOT/'direct_target_context_source_review_v1/review.json';source_review=read(source_review_path)
assert sha(source_review_path)=='a5fb96bdc1c192454966dfe6f75a08a1ef371cf9f65af59ef1e62d33e3b1c58e'
assert source_review['source_review_pass'] and source_review['context_data_review_pass'] and frozen['source_sha256']==source_review['source_sha256']
assert len(frozen['source_sha256'])==19 and len(frozen['input_sha256'])==258
checked={}
for path,digest in frozen['input_sha256'].items():assert sha(path)==digest,path;checked[Path(path).resolve().as_posix()]=digest
for name,digest in frozen['source_sha256'].items():
    p=Path(frozen['source_directory'])/name;assert sha(p)==digest;ast.parse(p.read_text(encoding='utf-8-sig'));checked[p.resolve().as_posix()]=digest
assert Path(frozen['source_directory']).resolve()==(BASE/'source_snapshot_v1').resolve()
for role,subject in request['subjects'].items():assert frozen['input_sha256'][subject['path']]==subject['sha256']==sha(subject['path']),role
proof=read(BASE/'context_preflight/report.json')
assert proof['passed'] and sha(BASE/'context_preflight/report.json')=='f5add6c31a3c3aec9b215d837d86e62a8a9bffcac4575d096d24012296c04583'
assert all(frozen['input_sha256'][p]==d for p,d in proof['input_sha256'].items())
assert request['root_selected'] is True and request['conditions']==['blinded','causal'] and request['updates_per_condition']==3000
assert request['ordinary_final_step']==68000 and request['coefficient']==1.8188207859141674 and request['coefficient_recalibration'] is False
assert request['learning_rate']==[1e-5,1e-6] and request['features']==1323 and request['context_std_floor']==.05
assert request['budgets']==dict(training_forward_rows=88116000,training_forward_calls=18000,training_updates=6000,
 diagnostic_Torch_rows=2940560,diagnostic_Torch_calls=11496,diagnostic_ORT_rows=735140,diagnostic_ORT_calls=2874,
 calibration_forward_calls=0,calibration_gradient_calls=0,native_calls=0,BFM_calls=0,manual_export_trace_calls=0)
assert request['subjects']['checkpoint']['sha256']=='8a1b67e09285a77910d62dd1b2214c8d3684004e55a82041544e750006b29a0a'
assert (BASE/'verify_completed_v2.py').as_posix() in frozen['input_sha256']
assert draft['approved'] is False and draft['request_sha256']==KNOWN['training_request.json'] and draft['launcher_sha256']==KNOWN['run_fit_durable_v1.ps1']
assert not (BASE/'fit').exists() and not (BASE/'fit_process/start.json').exists()
subjects={role:dict(path=(BASE/name).as_posix(),sha256=KNOWN[name]) for role,name in [('training_request','training_request.json'),('frozen_inputs','training_frozen_inputs.json'),('launcher','run_fit_durable_v1.ps1')]}
for role,p in [('source_review',source_review_path),('source_preparation',BASE/'source_preparation.json'),('context_proof',BASE/'context_preflight/report.json'),('completion_checker',BASE/'verify_completed_v2.py')]:subjects[role]=dict(path=p.as_posix(),sha256=sha(p))
for name,digest in KNOWN.items():assert sha(BASE/name)==digest,name
record=dict(passed=True,prelaunch_review_pass=True,source_review_pass=True,context_data_review_pass=True,
 training_request_sha256=KNOWN['training_request.json'],frozen_receipt_sha256=KNOWN['training_frozen_inputs.json'],
 subjects=subjects,source_sha256=frozen['source_sha256'],input_sha256=checked,source_files_checked=19,input_pins_checked=258,
 conditions=['blinded','causal'],updates_per_condition=3000,ordinary_final_step=68000,selected_single_pair=True,
 fixed_coefficient=1.8188207859141674,no_gradient_recalibration=True,budgets=request['budgets'],
 review_scope=['Exact final source/data/proof/runtime identities and fixed paired restoration, normalization, schedule, objective and budgets.',
  'Durable hidden single child with explicit actual clearance SHA, fresh attempt lock, complete pre/post hash maps, captured Process.Handle before WaitForExit, known raw exit and retained failures.',
  'Owner v2 exact launch-pin membership/count and duplicate-path rejection; separate18-role condition qualification receipts. Original v1 checker/receipt preserved.',
  'No model, optimizer or native work performed by this review. Root still owns actual clearance/dispatch. No witness, controller or hardware clearance.'],
 limitations=source_review['limitations'],automatic_retry=False,controller_selected=False,model_calls=0,ORT_calls=0,gradient_calls=0,optimizer_updates=0,native_steps=0)
with (DEST/'review.json').open('x',encoding='utf-8') as f:json.dump(record,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(dict(path=(DEST/'review.json').as_posix(),sha256=sha(DEST/'review.json'),prelaunch_review_pass=True,input_pins=258,sources=19)))
