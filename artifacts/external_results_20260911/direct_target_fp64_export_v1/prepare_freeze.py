"""Freeze the selected validation; synthetic tests and saved-array preflight only."""
import hashlib,json,shutil,subprocess,sys,time
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parent
PYTHON=Path('E:/codex_sonic_runtime/direct_target_gpu_20260911/venv/Scripts/python.exe')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def main():
    snapshot=BASE/'source_snapshot_v1';snapshot.mkdir(exist_ok=False)
    names=['promoted_model.py','test_promoted.py','run_fp64.py','test_runner.py','direct_contract.py','direct_data.py','direct_diagnostics.py']
    for name in names:shutil.copy2(BASE/'source_draft_v1'/name,snapshot/name)
    sources={name:sha(snapshot/name) for name in names};tests=BASE/'frozen_tests';tests.mkdir(exist_ok=False)
    command=[str(PYTHON),'-m','unittest','-v','test_promoted','test_runner'];started=time.perf_counter()
    result=subprocess.run(command,cwd=snapshot,capture_output=True,text=True)
    (tests/'stdout.log').write_text(result.stdout,encoding='utf-8');(tests/'stderr.log').write_text(result.stderr,encoding='utf-8')
    write(tests/'report.json',dict(passed=result.returncode==0,exit_code=result.returncode,command=command,source_sha256=sources,synthetic_only=True,
        task_model_calls=0,optimizer_updates=0,native_steps=0,elapsed_seconds=time.perf_counter()-started))
    if result.returncode:raise RuntimeError('Synthetic tests failed; preserve frozen attempt')
    prior=ROOT/'direct_target_continuation_v1';oldrequest=read(prior/'training_request.json');paths=oldrequest['paths']
    sys.path.insert(0,str(snapshot));from direct_data import collect_input_pins,load_data
    pins=collect_input_pins(paths);data=load_data(paths,pins)
    write(BASE/'saved_data_preflight.json',dict(passed=True,features=[list(data[k].shape) for k in ('features','velocity_features','physical_features')],task_model_calls=0,native_steps=0,optimizer_updates=0));del data
    role_paths=dict(checkpoint=prior/'fit/student_head.pt',fit_report=prior/'fit/report.json',source_head=prior/'fit/student_head.onnx',normalization=prior/'fit/normalization.npz',
        training_request=prior/'training_request.json',training_frozen_inputs=prior/'training_frozen_inputs.json',owner_failure=prior/'owner_failure_verification.json',
        training_audit=ROOT/'direct_target_continuation_failure_audit_v1/results_v1/report.json',training_review=ROOT/'direct_target_continuation_training_only_review_v1/review.json')
    subjects={key:dict(path=p.as_posix(),sha256=sha(p)) for key,p in role_paths.items()}
    old_outputs={backend:{corpus:(prior/'fit'/('final_'+backend+'_'+corpus+'.npy')).as_posix() for corpus in ('nominal','velocity','physical')} for backend in ('GPU','CPU','ORT')}
    for p in role_paths.values():pins[p.as_posix()]=sha(p)
    for corpora in old_outputs.values():
        for path in corpora.values():pins[path]=sha(path)
    runtime=oldrequest['runtime'];identity=read(runtime['identity_path'])
    for path,digest in identity['source_binary_sha256'].items():
        if sha(path)!=digest:raise ValueError('Runtime binary changed '+path)
        pins[Path(path).as_posix()]=digest
    extra=[Path(runtime['identity_path']),Path(runtime['verification_path']),BASE/'proposal.json',BASE/'PROPOSAL.md',BASE/'prepare_freeze.py',BASE/'run_export_durable_v1.ps1',BASE/'read_export_progress.ps1',BASE/'saved_data_preflight.json',tests/'report.json',tests/'stdout.log',tests/'stderr.log']
    extra += [p for p in (BASE/'support').iterdir() if p.is_file()]
    for p in extra:pins[p.as_posix()]=sha(p)
    request=dict(kind='one_same_weight_fp64_export_validation',root_selected=True,ordinary_final_step=55000,features=1000,head_output='normalized_target',
        execution_dtype='float64',public_input_dtype='float32',public_output_dtype='float32',ELU_implementation='Where(x>0,x,Exp(Min(x,0))-1)',
        parity_tolerance_rad=1e-5,backend_order=['CPU64','GPU64','ORT64'],corpus_rows=[9904,140622,3054],batch_size=256,per_backend_batches=[39,550,12],
        budgets=dict(Torch_calls=1202,Torch_rows=307160,ORT_calls=601,ORT_rows=153580,trace_forward_calls=0,optimizer_updates=0,BFM_calls=0,native_steps=0),
        subjects=subjects,old_outputs=old_outputs,paths=paths,runtime=runtime,full_drift_arrays=27,automatic_retry=False,checkpoint_selection=False,canonical_evaluation_authorized=False)
    write(BASE/'export_request.json',request)
    receipt=dict(kind='frozen_same_weight_fp64_validation',source_directory=snapshot.as_posix(),source_sha256=sources,input_sha256=dict(sorted(pins.items())),
        export_request_sha256=sha(BASE/'export_request.json'),source_count=len(sources),input_count=len(pins))
    write(BASE/'export_frozen_inputs.json',receipt)
    write(BASE/'export_clearance_draft.json',dict(approved=False,request_sha256=sha(BASE/'export_request.json'),frozen_receipt_sha256=sha(BASE/'export_frozen_inputs.json'),
        launcher_path=(BASE/'run_export_durable_v1.ps1').as_posix(),launcher_sha256=sha(BASE/'run_export_durable_v1.ps1'),review_pass_field='prelaunch_review_pass',
        Torch_rows=307160,ORT_calls=601,ordinary_final_step=55000,automatic_retry=False))
    print(json.dumps(dict(frozen=True,request_sha256=sha(BASE/'export_request.json'),receipt_sha256=sha(BASE/'export_frozen_inputs.json'),sources=len(sources),inputs=len(pins))))
if __name__=='__main__':main()
