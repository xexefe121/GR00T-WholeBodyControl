"""Freeze one selected fit after qualified data; saved-array checks/synthetic tests only."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

BASE=Path(__file__).resolve().parent
PYTHON=Path('E:/codex_sonic_runtime/direct_target_gpu_20260911/venv/Scripts/python.exe')
RUNTIME=PYTHON.parent.parent.parent
NAMES=['direct_contract.py','direct_model.py','direct_objective.py','direct_data.py','restoration_support.py',
    'full_state_contract.py','full_state_objective.py','full_state_data.py','full_state_diagnostics.py',
    'training_support.py','train_full_state.py','promoted_model.py',
    'test_full_state_helpers.py','test_training_preservation.py','test_promoted_graph.py']

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')

def main():
    selected=read(BASE/'qualification_roles.json')
    if selected['root_selected'] is not True:raise ValueError('Actual fit selection absent.')
    full_root=BASE.parent/'direct_target_full_state_secants_v1';generation=full_root/'generation'
    generation_report=read(generation/'report.json')
    if generation_report['complete'] is not True:raise ValueError('Generation incomplete.')
    for name in ('full_state_root_audit','full_state_data_review'):
        subject=selected['subjects'][name]
        if sha(subject['path'])!=subject['sha256']:raise ValueError('Qualification identity changed.')
        report=read(subject['path'])
        if report[subject['pass_field']] is not True:raise ValueError('Qualification not passed: '+name)
        if not subject['required_fields']:raise ValueError('Qualification needs concrete subject bindings.')
        for key,digest in subject['required_fields'].items():
            if report.get(key)!=digest:raise ValueError('Qualification direct binding differs: '+name+'/'+key)
    snapshot=BASE/'source_snapshot_v1';snapshot.mkdir(exist_ok=False)
    for name in NAMES:shutil.copy2(BASE/'source_draft_v1'/name,snapshot/name)
    sources={name:sha(snapshot/name) for name in NAMES}
    tests=BASE/'tests';tests.mkdir(exist_ok=False);started=time.perf_counter()
    command=[str(PYTHON),'-m','unittest','-v','test_full_state_helpers','test_training_preservation','test_promoted_graph']
    result=subprocess.run(command,cwd=snapshot,capture_output=True,text=True)
    (tests/'stdout.log').write_text(result.stdout,encoding='utf-8');(tests/'stderr.log').write_text(result.stderr,encoding='utf-8')
    write(tests/'report.json',dict(passed=result.returncode==0,exit_code=result.returncode,command=command,
        source_sha256=sources,elapsed_seconds=time.perf_counter()-started,synthetic_only=True,
        task_model_calls=0,CUDA_calls=0,ORT_calls=0,optimizer_updates=0,native_steps=0,
        stdout_sha256=sha(tests/'stdout.log'),stderr_sha256=sha(tests/'stderr.log')))
    if result.returncode:raise RuntimeError('Synthetic tests failed; preserve frozen attempt.')
    sys.path.insert(0,str(snapshot))
    from direct_data import default_paths,collect_input_pins
    from full_state_data import load_full_state_data
    paths=default_paths(BASE);pins=collect_input_pins(paths)
    full_paths=dict(report=(generation/'report.json').as_posix(),request=(full_root/'request.json').as_posix(),manifest=(generation/'manifest.json').as_posix())
    for name in generation_report['output_sha256']:
        path=generation/name
        if sha(path)!=generation_report['output_sha256'][name]:raise ValueError('Generation output changed.')
        pins[path.as_posix()]=sha(path)
    for path in list(map(Path,full_paths.values()))+[generation/'group_cell_summary.json',generation/'center_gate.json',generation/'overlap_gate.json',generation/'input_schema_gate.json']:
        pins[path.as_posix()]=sha(path)
    for path in (full_root/'source_snapshot_v1').glob('*.py'):pins[path.as_posix()]=sha(path)
    subjects=dict(selected['subjects'])
    old=BASE.parent/'direct_target_continuation_v1/fit'
    for name,path in [('checkpoint',old/'student_head.pt'),('normalization',old/'normalization.npz'),('source_fit_report',old/'report.json'),
        ('source_training_review',BASE.parent/'direct_target_continuation_training_only_review_v1/review.json')]:
        subjects[name]=dict(path=path.as_posix(),sha256=sha(path))
    predictions={key:(old/('final_GPU_'+key+'.npy')).as_posix() for key in ('nominal','velocity','physical')}
    for subject in subjects.values():pins[Path(subject['path']).as_posix()]=subject['sha256']
    for path in predictions.values():pins[Path(path).as_posix()]=sha(path)
    data=load_full_state_data(paths,full_paths,pins)
    write(BASE/'saved_data_preflight.json',dict(passed=True,source_sha256=sources,input_sha256=pins,
        nominal_shape=list(data['features'].shape),full_state_shape=list(data['full_state_features'].shape),physical_shape=list(data['physical_features'].shape),
        nominal_cells=list(map(len,data['cells'])),full_state_center_cells=list(map(len,data['full_state_cells'])),physical_cells=list(map(len,data['physical_cells'])),
        center_map_shape=list(data['center_map'].shape),group_axes=[len(value) for value in data['full_state_group_axes']],
        task_model_calls=0,BFM_calls=0,native_steps=0,optimizer_updates=0,normalization_computed=False))
    del data
    identity_path=RUNTIME/'runtime_identity.json';identity=read(identity_path)
    if identity['passed'] is not True:raise ValueError('Runtime identity failed.')
    for path,digest in identity['source_binary_sha256'].items():
        if sha(path)!=digest:raise ValueError('Runtime binary changed: '+path)
        pins[Path(path).as_posix()]=digest
    for path in [identity_path,RUNTIME/'verification.json',BASE/'DESIGN.md',BASE/'proposal.json',BASE/'OUTPUT_SCHEMA.md',BASE/'qualification_roles.json',
        BASE/'prepare_freeze.py',BASE/'run_fit_durable_v1.ps1',BASE/'read_fit_progress.ps1',BASE/'verify_completed.py',
        tests/'report.json',tests/'stdout.log',tests/'stderr.log',BASE/'saved_data_preflight.json']:
        pins[path.as_posix()]=sha(path)
    proposal=read(BASE/'proposal.json');verification_path=RUNTIME/'verification.json'
    request=dict(kind='one_full_state_finite_feedback_fit',root_selected=True,updates=10000,ordinary_final_step=65000,optimizer_step=10000,
        sampler_seed=20260911,features=1000,head_output='normalized_target',paths=paths,full_state_paths=full_paths,subjects=subjects,
        restoration_predictions=predictions,restoration=dict(actor='exact ordinary55000',optimizer='fresh empty AdamW moments',RNG='restore original55k after model construction',
            normalization='unchanged original15-cell archive',initial_saved_prediction_gate='nominal/physical exact partitions; full58 oldvelocity overlap drift only'),
        objective=proposal['objective'],sampler=proposal['sampler'],optimizer=proposal['optimizer'],export=proposal['export'],
        diagnostics=dict(corpora=['nominal','full_state','physical'],rows=[9904,354612,3054],batch=256,calls=[39,1386,12],
            labels=['initial_GPU32','final_GPU32','CPU64','GPU64','ORT64'],preclamp_parity_tolerance_rad=1e-5,FP32_ONNX_release=False),
        budgets=dict(calibration_forward_rows=14686,calibration_forward_calls=3,calibration_gradient_calls=3,
            training_forward_rows=146860000,training_forward_calls=30000,diagnostic_Torch_rows=1470280,diagnostic_Torch_calls=5748,
            diagnostic_ORT_rows=367570,diagnostic_ORT_calls=1437,native_calls=0,BFM_calls=0,manual_export_trace_calls=0),
        runtime=dict(python_path=PYTHON.as_posix(),torch_version='2.10.0+cu128',cuda_version='12.8',numpy_version='1.23.5',onnx_version='1.22.0',ort_version='1.23.2',
            verification_path=verification_path.as_posix(),verification_sha256=sha(verification_path),pass_field='passed',identity_path=identity_path.as_posix(),identity_sha256=sha(identity_path),
            deterministic_algorithms=True,TF32=False,AMP=False,CUBLAS_WORKSPACE_CONFIG=':4096:8'),
        automatic_retry=False,checkpoint_selection=False,canonical_evaluation_authorized=False,hardware_authorized=False)
    write(BASE/'training_request.json',request)
    receipt=dict(kind='frozen_full_state_finite_feedback_fit',source_directory=snapshot.as_posix(),source_sha256=sources,input_sha256=dict(sorted(pins.items())),
        training_request_sha256=sha(BASE/'training_request.json'),source_file_count=len(sources),input_count=len(pins),native_steps=0,task_model_calls=0,optimizer_updates=0)
    write(BASE/'training_frozen_inputs.json',receipt)
    write(BASE/'training_clearance_draft.json',dict(approved=False,request_sha256=sha(BASE/'training_request.json'),frozen_receipt_sha256=sha(BASE/'training_frozen_inputs.json'),
        launcher_path=(BASE/'run_fit_durable_v1.ps1').as_posix(),launcher_sha256=sha(BASE/'run_fit_durable_v1.ps1'),review_pass_field='prelaunch_review_pass',
        required_review_fields=['training_request_sha256','frozen_receipt_sha256'],additional_updates=10000,ordinary_final_step=65000,optimizer_step=10000,automatic_retry=False))
    print(json.dumps(dict(frozen=True,request_sha256=sha(BASE/'training_request.json'),frozen_receipt_sha256=sha(BASE/'training_frozen_inputs.json'),source_count=len(sources),input_count=len(pins))),flush=True)

if __name__=='__main__':main()
