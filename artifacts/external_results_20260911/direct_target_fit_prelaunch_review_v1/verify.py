"""Frozen fit source/input/command review. No task model or optimizer calls."""
import ast,hashlib,json,re
from pathlib import Path
BASE=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911');OWNER=BASE/'direct_target_student_v1';OUT=Path(__file__).parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
pins={}
def pin(p,d=None):
    p=Path(p);actual=sha(p)
    if d is not None:assert actual==d,p
    pins[p.as_posix()]=actual
rp=OWNER/'training_request.json';fp=OWNER/'training_frozen_inputs.json';lp=OWNER/'run_fit_durable_v1.ps1'
pin(rp,'c77506075081e608133eba7921e9ff6842464e069b31e01bafae12d9f0442a0e');pin(fp,'ff918ef575c276bc4ed0ee262217163f5c65146faee1829d791d8ad793a6b226')
r=read(rp);f=read(fp);source=Path(f['source_directory'])
assert f['training_request_sha256']==sha(rp) and f['source_file_count']==7 and f['input_count']==169
assert len(f['source_sha256'])==7 and len(f['input_sha256'])==169
for n,d in f['source_sha256'].items():
    pin(source/n,d);ast.parse((source/n).read_text());assert sha(OWNER/'source_draft_v1'/n)==d
for p,d in f['input_sha256'].items():pin(p,d)
assert r['kind']=='one_fresh_direct_absolute_target_fit' and r['root_selected'] is True
assert r['updates']==r['ordinary_final_step']==5000 and r['seed']==20260911
assert r['architecture']==[1000,256,256,23] and r['activation']=='ELU'
assert r['features']==1000 and r['head_output']=='normalized_target' and r['retained_indices']==[[0,52],[75,1023]]
assert r['budgets']==dict(training_head_rows=70550000,diagnostic_torch_rows=460740,ORT_calls=601,native_steps=0,BFM_calls=0)
assert r['diagnostics']['ORT_calls_per_corpus']==[39,550,12] and r['diagnostics']['preclamp_parity_tolerance_rad']==1e-5
assert r['diagnostics']['passes']==['initial_GPU','final_GPU','final_CPU','final_ORT']
assert all(r['objective'][k]==1 for k in ('nominal_weight','velocity_weight','physical_weight'))
assert r['objective']['nominal_counts']==[[100,819,100]]*3+[[100,5780,100],[100,667,100]]
assert r['objective']['physical_requested_counts']==[99,819,100]*3
assert r['objective']['velocity_pairs_per_step']==576 and r['objective']['velocity_signs']==2
assert r['normalization']['equal_cells']==15 and r['normalization']['standard_deviation_floor']==.05
assert r['optimizer']==dict(name='AdamW',lr_start=3e-4,lr_end=3e-5,schedule='inclusive cosine over5000',weight_decay=1e-5,grad_clip_norm=10,foreach=False,fused=False)
assert r['automatic_retry'] is False and r['hardware_authorized'] is False and r['canonical_evaluation_authorized'] is False
runtime=r['runtime'];pin(runtime['verification_path'],runtime['verification_sha256']);pin(runtime['identity_path'],runtime['identity_sha256'])
assert runtime['verification_sha256']=='671ca4c18c4da39a1538af7a1322180cc7ead9115b6bb93daf554a59a0689dcc'
assert runtime['identity_sha256']=='b173e01f26921f895f3570eace50d655cc2f767df0c9c98142b5fcff0c0e55f2'
assert read(runtime['verification_path'])[runtime['pass_field']] is True
identity=read(runtime['identity_path']);assert identity['passed'] is True and len(identity['source_binary_sha256'])==53
for p,d in identity['source_binary_sha256'].items():assert pins[Path(p).as_posix()]==d
assert runtime['python_path']=='E:/codex_sonic_runtime/direct_target_gpu_20260911/venv/Scripts/python.exe'
assert runtime['torch_version']=='2.10.0+cu128' and runtime['cuda_version']=='12.8'
assert runtime['CUBLAS_WORKSPACE_CONFIG']==':4096:8' and runtime['deterministic_algorithms'] is True and runtime['TF32'] is False and runtime['AMP'] is False
tests=read(OWNER/'tests/report.json');pre=read(OWNER/'saved_data_preflight.json')
assert tests['passed'] is True and tests['exit_code']==0 and tests['source_sha256']==f['source_sha256']
assert re.search(r'Ran 23 tests.*\n\s*\nOK\s*$',(OWNER/'tests/stderr.log').read_text(),re.S)
assert pre['passed'] is True and pre['source_sha256']==f['source_sha256']
assert pre['nominal_shape']==[9904,1000] and pre['velocity_shape']==[140622,1000] and pre['physical_shape']==[3054,1000]
assert pre['nominal_cells']==[100,819,100]*3+[100,5780,100,100,667,100] and pre['physical_cells']==[99,819,100]*3
assert pre['center_map_shape']==[3057] and pre['sampler_shape']==[5000,576]
for p,d in pre['input_sha256'].items():assert pins[Path(p).as_posix()]==d
launcher=lp.read_text();pin(lp)
for term in ('CreateNew','-WindowStyle Hidden',"'CUBLAS_WORKSPACE_CONFIG',':4096:8'",'70550000','460740','601','training_request_sha256','frozen_receipt_sha256','[IO.FileShare]::Delete'):
    assert term in launcher,term
assert launcher.index('$capturedHandle=$child.Handle')<launcher.index('$child.WaitForExit()')<launcher.index('$rawExit=$child.ExitCode')
assert "source_snapshot_v1" in launcher and 'direct_target_gpu_20260911' in launcher and "train_direct.py" in launcher
for name in ('fit','training_clearance.json','fit_process/running.lock','fit_process/start.json','fit_process/child.json','fit_process/exit.json','fit_process/stdout.log','fit_process/stderr.log'):assert not (OWNER/name).exists(),name
for p in (BASE/'direct_target_evaluation_source_review_v1/review.json',BASE/'direct_target_root_audit_source_review_v1/review.json'):pin(p)
pin(Path(__file__))
def subject(p):return dict(path=Path(p).as_posix(),sha256=sha(p))
review=dict(kind='one_selected_direct_absolute_target_fit_prelaunch_review',verdict='CLEAR',prelaunch_review_pass=True,source_review_pass=True,
    training_request_sha256=sha(rp),frozen_receipt_sha256=sha(fp),
    subjects=dict(training_request=subject(rp),frozen_inputs=subject(fp),launcher=subject(lp),runtime_identity=subject(runtime['identity_path']),runtime_verification=subject(runtime['verification_path']),
                  saved_data_preflight=subject(OWNER/'saved_data_preflight.json'),tests=subject(OWNER/'tests/report.json')),
    source_sha256=f['source_sha256'],input_sha256=pins,verified_frozen_input_count=169,verified_source_count=7,verified_runtime_pins=53,
    synthetic_tests_bound=23,qualified_nominal_rows=9904,qualified_velocity_rows=140622,qualified_physical_rows=3054,
    selected_scope=dict(fits=1,fresh_seed=20260911,updates=5000,ordinary_final_step=5000,architecture=[1000,256,256,23],budgets=r['budgets'],canonical_evaluation_authorized=False),
    evidence=['Equal15-cell nominal normalization includes global between-cell variance; only nominal rows set moments.','Output is normalized absolute target; existing span32 and each output32 promote to64 before runtime math and response subtraction.','Explicit dataset/control center/successor mappings and existing5000x576 schedule preserve nine response cells; all coefficients1.','Fresh default initialization with deterministicCUDA/noTF32/noAMP; no weights/optimizer restored or output-zeroed.','Ordinaryfinal5000 checkpoint and optimizer/RNG saved before selected final export diagnostics; no checkpoint selection.','All three finalGPU/CPU/ORT corpus partitions use39+550+12 calls and preclamp1e-5rad parity. Nonfinite comparisons fail.','Exception paths retain current inputs/returns, actual optimizer counters and committed prefixes; no automatic retry.','Hidden isolated Python launch requires exact clearance/review/request/receipt, CreateNew attempt guard, captured process handle, known exit and pre/post source/input pins.'],
    reviewer_model_calls=0,reviewer_native_steps=0,reviewer_optimizer_updates=0,hardware_authorized=False,
    limitations=['Training/export success is not connected simulation qualification.','Future model witness and canonical evaluation require their actual final artifact reviews and selected launches.'])
with (OUT/'review.json').open('x',encoding='utf-8') as stream:json.dump(review,stream,indent=2);stream.write('\n')
print(json.dumps(dict(verdict='CLEAR',review_sha256=sha(OUT/'review.json'),pins=len(pins))))
