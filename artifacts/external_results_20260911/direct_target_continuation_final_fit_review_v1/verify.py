"""Review completed immutable direct fit evidence; zero task graphs or physics."""
import hashlib
import json
from pathlib import Path
import numpy as np

NEW=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911')
BASE=NEW/'direct_target_continuation_v1'
OUT=Path(__file__).resolve().parent
tracked={}
import argparse
parser=argparse.ArgumentParser()
for key in ('fit_report_sha','checkpoint_sha','head_sha','root_audit','root_audit_sha','owner_sha'):parser.add_argument('--'+key.replace('_','-'),required=True)
args=parser.parse_args()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def bind(path,expected=None):
    path=Path(path).resolve();digest=sha(path)
    if expected is not None:assert digest==expected,(str(path),digest,expected)
    key=path.as_posix()
    if key in tracked:assert tracked[key]==digest,key
    tracked[key]=digest
    return path
def read(path,expected=None):return json.loads(bind(path,expected).read_text(encoding='utf-8-sig'))
def subject(path):
    path=bind(path);return {'path':path.as_posix(),'sha256':tracked[path.as_posix()]}
def exact(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()

assert not (OUT/'review.json').exists()
request=read(BASE/'training_request.json','e3dbc8c41433699326cd611ea97767974b6eb9ce5de2c7b1f678f83594054625')
frozen=read(BASE/'training_frozen_inputs.json','1b9499a88b26b9feb5e1259d63284010e7521150c591a9815063cacc0251fa9c')
launch_review=read(NEW/'direct_target_continuation_prelaunch_review_v1/review.json','c1f0ab3a53503985bde58f7ab566a5ccb656af41d1d9e9c8c73f8f61320a120f')
assert launch_review['prelaunch_review_pass'] is True
fit=BASE/'fit'
report=read(fit/'report.json',args.fit_report_sha)
checkpoint=bind(fit/'student_head.pt',args.checkpoint_sha)
head=bind(fit/'student_head.onnx',args.head_sha)
audit_path=Path(args.root_audit)
audit=read(audit_path,args.root_audit_sha)
audit_review=read(NEW/'direct_target_continuation_root_audit_source_review_v1/review.json','ccc24870b1c5bc422b439c9c32e9a95181f5800ab838d042e66eb938b7826581')
bind(NEW/'direct_target_continuation_root_audit_v1/audit_saved_fit.py','d4055bfea1fd7d7a0813954cc8f6135f43466ed021160e781a6ef5d6b8c829a5')
assert audit['passed'] is True and audit['checks']>=50000
assert all(audit[k]==0 for k in ('task_model_calls','ORT_calls','optimizer_updates','native_steps'))
for name in ('completed','optimization_completed','final_export_diagnostics_completed','numerical_gate_passed','export_parity_passed','all_frozen_inputs_unchanged'):
    assert report[name] is True,name
assert report['ordinary_final_step']==55000 and report['additional_updates']==50000
assert report['fresh_initialization'] is False and report['restored_start_step']==5000
restoration=read(fit/'restoration.json',report['restoration_sha256'])
assert restoration['all_restoration_checks_passed'] is True and restoration['initial_GPU_predictions_exact'] is True and restoration['initial_metrics_exact'] is True
assert report['features']==1000 and report['head_output']=='normalized_target' and report['seed']==20260911
assert report['BFM_calls']==report['native_steps']==0 and report['checkpoint_selection'] is False
assert report['hardware_authorized'] is False
assert report['checkpoint_sha256']==sha(checkpoint) and report['onnx_sha256']==sha(head)
expected_counters={'training_head_rows_attempted':705500000,'training_head_rows_returned':705500000,'diagnostic_torch_rows_attempted':460740,'diagnostic_torch_rows_returned':460740,'ORT_calls_attempted':601,'ORT_calls_returned':601}
assert report['counters']==expected_counters
owner=read(BASE/'owner_completion_verification.json',args.owner_sha)
assert owner['owner_verification_passed'] is True and owner['report_sha256']==sha(fit/'report.json')
assert owner['checkpoint_sha256']==sha(checkpoint) and owner['onnx_sha256']==sha(head)
for relative,digest in owner['output_sha256'].items():bind(BASE/relative,digest)
exit_receipt=read(BASE/'fit_process/exit.json',owner['exit_sha256'])
assert exit_receipt['raw_python_exit_code']==exit_receipt['exit_code']==0
assert exit_receipt['exit_known'] is True and exit_receipt['error'] is None and exit_receipt['all_postrun_pins_exact'] is True
absence=read(BASE/'process_absence.json')
assert absence['processes_absent'] is True and absence['running']==[]
assert set(absence['expected_pids'])=={exit_receipt['wrapper_pid'],exit_receipt['child_pid']}
post=read(BASE/'fit_process/postrun_pins.json')
assert post['all_exact'] is True and post['count']==len(post['files'])==202
for path,item in post['files'].items():
    assert item['matched'] is True and item['expected']==item['actual'],path
    bind(path,item['expected'])
for path,digest in audit['input_sha256'].items():bind(path,digest)
for key in ('training_request','frozen_inputs','launcher'):
    sub=launch_review['subjects'][key];bind(sub['path'],sub['sha256'])
assert not (fit/'failure.json').exists()
audit_subjects={Path(path).resolve().as_posix():digest for path,digest in audit['input_sha256'].items()}
for path in (fit/'report.json',checkpoint,head,BASE/'training_request.json',BASE/'training_frozen_inputs.json'):
    assert audit_subjects[path.resolve().as_posix()]==sha(path),path

# Recompute saved export numerical comparisons and verify all finite outputs.
norm=np.load(bind(fit/'normalization.npz'),allow_pickle=False)
span=norm['joint_span'].astype(np.float64);default=norm['default_q']
assert span.shape==(23,) and default.shape==(23,) and span.dtype==default.dtype==np.float64
parity=read(fit/'export_parity.json');maximum=0.
for corpus,count in [('nominal',9904),('velocity',140622),('physical',3054)]:
    arrays={}
    for label in ('initial_GPU','final_GPU','final_CPU','final_ORT'):
        array=np.load(bind(fit/f'{label}_{corpus}.npy'),allow_pickle=False,mmap_mode='r')
        assert array.shape==(count,23) and array.dtype==np.float32 and np.isfinite(array).all()
        arrays[label]=default+span*array.astype(np.float64)
    for a,b,name in [('final_GPU','final_CPU','GPU_CPU'),('final_GPU','final_ORT','GPU_ORT'),('final_CPU','final_ORT','CPU_ORT')]:
        error=float(np.max(np.abs(arrays[a]-arrays[b])))
        assert error==parity['comparisons'][corpus+'_'+name]
        maximum=max(maximum,error)
assert maximum==parity['maximum_preclamp_rad']==audit['maximum_preclamp_export_error_rad']
assert maximum<=parity['tolerance_rad']==1e-5 and parity['passed'] is True and parity['nonfinite_comparisons']==[]
assert report['export_parity']==parity

dataset_names=('centers','velocity_features','velocity_target','physical_manifest','pico','walk002')
subjects={name:subject(request['paths'][name]) for name in dataset_names}
subjects.update(training_request=subject(BASE/'training_request.json'),frozen_inputs=subject(BASE/'training_frozen_inputs.json'),
    training_manifest=subject(BASE/'training_frozen_inputs.json'),restoration=subject(fit/'restoration.json'),
    fit_report=subject(fit/'report.json'),checkpoint=subject(checkpoint),head=subject(head),root_audit=subject(audit_path),
    root_audit_source_review=subject(NEW/'direct_target_continuation_root_audit_source_review_v1/review.json'),
    prelaunch_review=subject(NEW/'direct_target_continuation_prelaunch_review_v1/review.json'),
    owner_completion=subject(BASE/'owner_completion_verification.json'),exit=subject(BASE/'fit_process/exit.json'),
    process_absence=subject(BASE/'process_absence.json'),postrun_pins=subject(BASE/'fit_process/postrun_pins.json'))
metric_names=('nominal_objective','full_velocity_objective','physical_objective')
metrics={phase:{key:report[phase+'_metrics'][key] for key in metric_names} for phase in ('initial','final')}
for phase in ('initial','final'):
    cell_metrics=report[phase+'_metrics']['nominal_cells']
    assert len(cell_metrics)==15
    metrics[phase]['nominal_equal_cell_preclip_RMSE_rad']=float(np.sqrt(np.mean([cell['preclip_RMSE_rad']**2 for cell in cell_metrics])))
    metrics[phase]['nominal_clipped_rows']=sum(cell['clipped_rows'] for cell in cell_metrics)
changes={key:100*(metrics['final'][key]/metrics['initial'][key]-1) for key in metric_names[:3]}
# Preserve all objective tradeoffs; no monotonicity or best-checkpoint selection.
bind(__file__)
review=dict(kind='ordinary_final_direct_absolute_target_fit_dataset_export_review',verdict='CLEAR',
    dataset_review_pass=True,fit_review_pass=True,export_review_pass=True,source_review_pass=True,
    subjects=subjects,training_dataset_sha256=[subjects['training_manifest']['sha256']],
    training_request_sha256=sha(BASE/'training_request.json'),frozen_receipt_sha256=sha(BASE/'training_frozen_inputs.json'),
    ordinary_final_step=55000,additional_updates=50000,restored_start_step=5000,features=1000,head_output='normalized_target',architecture=[1000,256,256,23],
    root_audit_checks=audit['checks'],reviewed_input_files=len(tracked),input_sha256=tracked,
    counters=expected_counters,maximum_preclamp_export_error_rad=maximum,metrics=metrics,objective_change_percent=changes,
    selection='The root-selected ordinary final55000 after exact5000 restoration is retained; no checkpoint selection or additional fit.',
    limitations=['All objective tradeoffs are retained. No monotonic-loss requirement was selected; this continues the exact ordinary5000 checkpoint.',
        'Saved result checks and export tolerance do not establish feedback stability, source tracking, real-time behavior or hardware readiness.',
        'CPU/GPU/ORT comparisons satisfy the selected 1e-5 rad tolerance; no bitexact cross-provider claim.',
        'One separately selected WSL batch-one witness and canonical evaluation require concrete source/binding/launcher review before execution.'],
    reviewer_task_model_calls=0,reviewer_ORT_calls=0,reviewer_optimizer_updates=0,reviewer_native_steps=0,
    behavioral_qualification=False,hardware_authorized=False)
(OUT/'review.json').write_text(json.dumps(review,indent=2)+'\n')
print(json.dumps({'verdict':'CLEAR','review_sha256':sha(OUT/'review.json'),'reviewed_inputs':len(tracked),'maximum_preclamp_rad':maximum,'objective_change_percent':changes}))
