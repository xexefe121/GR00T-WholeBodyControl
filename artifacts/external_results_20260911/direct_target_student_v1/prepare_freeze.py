"""Freeze the selected direct-target fit; only synthetic tests and saved arrays."""
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

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')

def main():
    snapshot=BASE/'source_snapshot_v1';snapshot.mkdir(exist_ok=False)
    names=['direct_contract.py','direct_model.py','direct_objective.py','direct_data.py','direct_diagnostics.py','train_direct.py','test_direct.py']
    for name in names:shutil.copy2(BASE/'source_draft_v1'/name,snapshot/name)
    sources={name:sha(snapshot/name) for name in names}
    tests=BASE/'tests';tests.mkdir(exist_ok=False)
    started=time.perf_counter()
    command=[str(PYTHON),'-m','unittest','-v','test_direct']
    result=subprocess.run(command,cwd=snapshot,capture_output=True,text=True)
    (tests/'stdout.log').write_text(result.stdout,encoding='utf-8');(tests/'stderr.log').write_text(result.stderr,encoding='utf-8')
    write(tests/'report.json',dict(passed=result.returncode==0,exit_code=result.returncode,command=command,working_directory=str(snapshot),
        source_sha256=sources,elapsed_seconds=time.perf_counter()-started,synthetic_only=True,task_model_calls=0,optimizer_updates=0,native_steps=0,
        stdout_sha256=sha(tests/'stdout.log'),stderr_sha256=sha(tests/'stderr.log')))
    if result.returncode:raise RuntimeError('Synthetic tests failed; preserve the frozen source and test evidence.')
    sys.path.insert(0,str(snapshot))
    from direct_data import default_paths,collect_input_pins,load_data
    paths=default_paths(BASE);pins=collect_input_pins(paths)
    data=load_data(paths,pins)
    write(BASE/'saved_data_preflight.json',dict(passed=True,source_sha256=sources,input_sha256=pins,nominal_shape=list(data['features'].shape),
        velocity_shape=list(data['velocity_features'].shape),physical_shape=list(data['physical_features'].shape),
        nominal_cells=list(map(len,data['cells'])),physical_cells=list(map(len,data['physical_cells'])),
        center_map_shape=list(data['center_map'].shape),sampler_shape=list(data['sampled_rows'].shape),
        task_model_calls=0,BFM_calls=0,native_steps=0,optimizer_updates=0,normalization_computed=False))
    del data
    identity_path=RUNTIME/'runtime_identity.json';identity=json.loads(identity_path.read_text(encoding='utf-8-sig'))
    if identity['passed'] is not True:raise ValueError('Runtime identity not passed.')
    for path,digest in identity['source_binary_sha256'].items():
        if sha(path)!=digest:raise ValueError('Changed runtime binary: '+path)
        pins[Path(path).as_posix()]=digest
    for path in [identity_path,RUNTIME/'verification.json',BASE/'DESIGN.md',BASE/'prepare_freeze.py',BASE/'run_fit_durable_v1.ps1',BASE/'read_fit_progress.ps1',
                 tests/'report.json',tests/'stdout.log',tests/'stderr.log',BASE/'saved_data_preflight.json']:
        pins[path.as_posix()]=sha(path)
    verification_path=RUNTIME/'verification.json'
    request=dict(kind='one_fresh_direct_absolute_target_fit',root_selected=True,updates=5000,ordinary_final_step=5000,seed=20260911,
        features=1000,retained_indices=[[0,52],[75,1023]],head_output='normalized_target',architecture=[1000,256,256,23],activation='ELU',
        initialization='fresh default Torch under seed20260911; no restored state or zeroed output layer',paths=paths,
        normalization=dict(population='all9904 nominal only',equal_cells=15,mean_variance='float64 global mean and between-cell population variance',
            saved_dtype='float32',standard_deviation_floor=0.05),
        target_contract=dict(labels='((absolute_target_f64-default_f64)/existing_span_f32.astype(float64)).astype(float32)',
            reconstruction='default_f64 + existing_span_f32.astype(float64) * head_f32.astype(float64)',
            final_clamp='unchanged native float64 bounds',output='normalized absolute target; no BFM or residual output'),
        objective=dict(nominal_weight=1,velocity_weight=1,physical_weight=1,nominal_cells=15,velocity_cells=9,physical_cells=9,
            nominal_counts=[[100,819,100],[100,819,100],[100,819,100],[100,5780,100],[100,667,100]],
            physical_requested_counts=[99,819,100]*3,velocity_pairs_per_step=576,velocity_signs=2,
            response_cast='cast each float32 endpoint and center prediction tofloat64 BEFORE subtraction',
            teacher_response='float64 absolute endpoint minus matching nominal absolute target, divided only by promoted existing span',
            sampler='exact saved original5000x576 rows and axes; zero new sample draws; differentiable nominal center reuse'),
        optimizer=dict(name='AdamW',lr_start=3e-4,lr_end=3e-5,schedule='inclusive cosine over5000',weight_decay=1e-5,
            grad_clip_norm=10,foreach=False,fused=False),
        diagnostics=dict(partition='each corpus separately in fixed order nominal,velocity,physical; batch256',
            corpora_rows=[9904,140622,3054],passes=['initial_GPU','final_GPU','final_CPU','final_ORT'],
            ORT_calls_per_corpus=[39,550,12],preclamp_parity_tolerance_rad=1e-5,checkpoint_selection=False),
        budgets=dict(training_head_rows=70550000,diagnostic_torch_rows=460740,ORT_calls=601,native_steps=0,BFM_calls=0),
        runtime=dict(python_path=PYTHON.as_posix(),torch_version='2.10.0+cu128',cuda_version='12.8',numpy_version='1.23.5',
            onnx_version='1.22.0',onnxruntime_version='1.23.2',verification_path=verification_path.as_posix(),verification_sha256=sha(verification_path),
            pass_field='passed',identity_path=identity_path.as_posix(),identity_sha256=sha(identity_path),device='NVIDIA GeForce RTX 3070 Laptop GPU',
            deterministic_algorithms=True,TF32=False,AMP=False,CUBLAS_WORKSPACE_CONFIG=':4096:8'),
        automatic_retry=False,hardware_authorized=False,canonical_evaluation_authorized=False)
    write(BASE/'training_request.json',request)
    receipt=dict(kind='frozen_direct_target_fit',source_directory=snapshot.as_posix(),source_sha256=sources,input_sha256=dict(sorted(pins.items())),
        training_request_sha256=sha(BASE/'training_request.json'),source_file_count=len(sources),input_count=len(pins),
        native_steps=0,task_model_calls=0,optimizer_updates=0)
    write(BASE/'training_frozen_inputs.json',receipt)
    write(BASE/'training_clearance_draft.json',dict(approved=False,request_sha256=sha(BASE/'training_request.json'),
        frozen_receipt_sha256=sha(BASE/'training_frozen_inputs.json'),launcher_path=(BASE/'run_fit_durable_v1.ps1').as_posix(),
        launcher_sha256=sha(BASE/'run_fit_durable_v1.ps1'),review_pass_field='prelaunch_review_pass',
        required_review_fields=['training_request_sha256','frozen_receipt_sha256'],additional_updates=5000,ordinary_final_step=5000,
        automatic_retry=False))
    print(json.dumps(dict(frozen=True,request_sha256=sha(BASE/'training_request.json'),frozen_receipt_sha256=sha(BASE/'training_frozen_inputs.json'),
        source_count=len(sources),input_count=len(pins))),flush=True)

if __name__=='__main__':main()
