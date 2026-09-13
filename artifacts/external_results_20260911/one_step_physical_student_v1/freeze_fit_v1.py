"""Prepare actual fit inputs only after selected data/source gates; never launch."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import traceback
import numpy as np

BASE=Path(__file__).resolve().parent
NEW=BASE.parent
PRIOR=NEW/'velocity_chord_student_v1'
SOURCE=BASE/'source_draft_v2'
SNAPSHOT=BASE/'source_snapshot_v1'
PYTHON=Path('C:/Users/camer/AppData/Local/Programs/Python/Python310/python.exe')
PT_SHA='17b8e240e3ca711e993a5c84c0221326d365f7ed892c8166bb594d804726bc3e'
ONNX_SHA='219b86cc4decd51671cb7aa036a944741cebed684accd91f7a03de37a7694bb5'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def pathstr(path):return Path(path).resolve().as_posix()
def direct_hash(value,digest):
    if isinstance(value,dict):return any(direct_hash(v,digest) for v in value.values())
    if isinstance(value,list):return any(direct_hash(v,digest) for v in value)
    return value==digest

def write_new(path,value):
    path=Path(path)
    with path.open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def digest_arg(value):
    if not re.fullmatch('[0-9a-f]{64}',value):raise argparse.ArgumentTypeError('Actual lowercase SHA256 required.')
    return value

def budget(valid):
    assert type(valid) is int and 0<valid<=3054
    return dict(valid_rows=valid,updates=5000,training_head_rows=5000*(4209+valid),
        legacy_head_onnx_calls=1126,physical_head_onnx_calls=2*((valid+255)//256),
        head_onnx_calls=1126+2*((valid+255)//256),diagnostic_torch_rows=2*(143686+valid),
        analytical_head_evaluations=14,BFM_calls=0,native_steps=0)

def validate_conflicts(report,valid):
    assert report['passed'] is True and report['rows_checked']==3054
    assert report['valid_labels']==valid and report['failed_branches']==3054-valid
    result=report['compatibility_with_unchanged_3057_nominal_centers']
    assert result['total_rows']==3057+valid
    assert result['exact_target_conflicts']==result['float32_residual_conflicts']==0
    assert result['rows_removed']==result['targets_averaged']==0
    assert result['unique_feature_inputs']+result['duplicate_inputs']==3057+valid
    assert result['signed_zero_canonicalized'] is True
    return result

def review_role(spec,subjects,pins):
    path=Path(spec['path']).resolve()
    assert sha(path)==spec['sha256']
    review=read(path);assert review[spec['pass_field']] is True
    pins[pathstr(path)]=sha(path)
    for subject,digest in subjects.items():
        assert sha(subject)==digest,subject
        assert direct_hash(review,digest),('Review lacks literal subject',str(path),subject)
        pins[pathstr(subject)]=digest
    return dict(path=pathstr(path),sha256=sha(path),pass_field=spec['pass_field'],
        subjects={pathstr(p):h for p,h in subjects.items()})

def selected(path,digest):
    assert sha(path)==digest
    data=read(path)
    assert data['kind']=='one_physical_fit_selection'
    assert data['model_fitting_authorized'] is True
    assert data['additional_updates']==5000 and data['ordinary_final_step']==75000
    assert type(data['exact_conflict_groups']) is int and data['exact_conflict_groups']==0
    budget(data['expected_valid_rows'])
    return data

def assert_absent(names):
    for name in names:assert not (BASE/name).exists(),('Preserve existing attempt',name)

def freeze(args):
    assert sys.platform=='win32' and sys.version_info[:3]==(3,10,11) and np.__version__=='1.23.5'
    assert_absent(('source_snapshot_v1','training_request.json','training_frozen_inputs.json','training_clearance.json',
        'fit_launch_plan.json','fit_launch_receipt.json','freeze_report.json','fit','fit_process',
        'fit_preflight_failure.json','freeze_failure.json','finalize_failure.json'))
    selection_path=args.selection.resolve()
    selection=selected(selection_path,args.selection_sha256)
    # Every supplied evidence file is pinned by the actual selection, never inferred.
    pins={pathstr(selection_path):args.selection_sha256}
    for name,digest in selection['input_sha256'].items():
        assert re.fullmatch('[0-9a-f]{64}',digest) and sha(name)==digest,name
        pins[pathstr(name)]=digest
    def selected_path(key):
        p=Path(selection['paths'][key]).resolve()
        assert pathstr(p) in pins and pins[pathstr(p)]==sha(p),key
        return p
    manifest_path=selected_path('collection_manifest');producer_path=selected_path('collection_report')
    collection_request=selected_path('collection_request')
    physics=selected_path('root_physics_report');labels=selected_path('root_labels_report')
    conflicts=selected_path('conflict_report')
    assert read(physics)['passed'] is True and read(labels)['passed'] is True
    conflict_report=read(conflicts)
    compatibility=validate_conflicts(conflict_report,selection['expected_valid_rows'])
    manifest=read(manifest_path);producer=read(producer_path)
    assert manifest['complete'] is True and producer['completed'] is True and producer['passed'] is True
    assert producer['rows']==producer['nominal_verified']==producer['policy_branches']==manifest['rows']==3054
    assert producer['manifest_sha256']==sha(manifest_path)
    assert producer['request_sha256']==manifest['request_sha256']==sha(collection_request)
    assert producer['query250_actual251_calibration_passed'] is True and producer['all_frozen_inputs_unchanged'] is True
    values={}
    for key,spec in manifest['arrays'].items():
        path=(manifest_path.parent/spec['path']).resolve()
        assert path.parent==manifest_path.parent.resolve(),key
        assert sha(path)==spec['sha256'],key
        pins[pathstr(path)]=spec['sha256']
        a=np.load(path,mmap_mode='r',allow_pickle=False)
        assert list(a.shape)==spec['shape'] and str(a.dtype)==spec['dtype'] and len(a)==3054,key
        if key in ('dataset','start_control','successor_control','center_index','successor_center_index',
            'source_frame','successor_frame','nominal_verified','label_valid','nominal_status','policy_status',
            'nominal_valid_steps','nominal_attempted_steps','nominal_returned_steps',
            'policy_valid_steps','policy_attempted_steps','policy_returned_steps'):values[key]=a
    d=np.repeat(np.arange(3,dtype=np.int64),1018);c=np.tile(np.arange(250,1268,dtype=np.int64),3);i=d*1019+c-250
    for key,a in dict(dataset=d,start_control=c,successor_control=c+1,center_index=i,successor_center_index=i+1,
                     source_frame=c+11,successor_frame=c+12).items():
        assert values[key].dtype==a.dtype and values[key].tobytes()==a.tobytes(),key
    valid=values['label_valid'];assert valid.dtype==np.bool_ and valid.shape==(3054,)
    assert values['nominal_verified'].dtype==np.bool_ and values['nominal_verified'].all()
    assert (values['nominal_status']==1).all() and np.array_equal(valid,values['policy_status']==1)
    assert (values['policy_status'][~valid]==2).all()
    for k in ('valid_steps','attempted_steps','returned_steps'):
        assert (values['nominal_'+k]==10).all()
        assert (values['policy_'+k][valid]==10).all()
        assert ((values['policy_'+k][~valid]>=1)&(values['policy_'+k][~valid]<=10)).all()
    V=int(valid.sum())
    assert V==selection['expected_valid_rows']==producer['labels_valid']==manifest['labels_valid']
    assert producer['strict_failed']==manifest['strict_failed']==3054-V
    assert producer['native_attempted']==producer['native_returned']==30540+int(values['policy_returned_steps'].sum())<=61080
    expected_calls={k:dict(attempted=n,returned=n) for k,n in dict(backward=V,actor=V,head=3054+V).items()}
    assert producer['graph_calls']==manifest['graph_calls']==expected_calls
    coverage=[]
    for code in range(3):
        for phase,lo,hi,n in (('acquisition',251,350,99),('source',350,1169,819),('return',1169,1269,100)):
            mask=(d==code)&(c+1>=lo)&(c+1<hi);assert int(mask.sum())==n
            count=int((mask&valid).sum())
            coverage.append(dict(dataset=code,phase=phase,requested=n,valid=count,strict_failed=n-count,
                                 effective_mass=count/(9*n),empty=count==0))
    data_subjects={pathstr(p):sha(p) for p in (manifest_path,producer_path,collection_request,physics,labels,conflicts)}
    # Prefix-restoration and root report requests must be explicit selection inputs;
    # combined review cannot claim a subject merely through an unbound request link.
    for path in selection.get('additional_data_subjects',[]):
        p=Path(path).resolve();assert pathstr(p) in pins
        data_subjects[pathstr(p)]=sha(p)
    data_role=review_role(selection['reviews']['branch_data'],data_subjects,pins)
    prior_report=read(PRIOR/'fit/report.json')
    assert prior_report['ordinary_final_step']==70000 and prior_report['numerical_gate_passed'] is True
    assert sha(PRIOR/'fit/student_head.pt')==prior_report['checkpoint_sha256']==PT_SHA
    assert sha(PRIOR/'fit/student_head.onnx')==prior_report['onnx_sha256']==ONNX_SHA
    prior_subjects={pathstr(PRIOR/'fit'/name):sha(PRIOR/'fit'/name) for name in ('student_head.pt','student_head.onnx','report.json')}
    fit_role=review_role(selection['reviews']['prior_fit'],prior_subjects,pins)
    export_role=review_role(selection['reviews']['prior_export'],prior_subjects,pins)
    prep=read(BASE/'preparation_report_v2.json')
    assert prep['passed'] is True and prep['tests']==32 and prep['unchanged_original_helpers']==32
    assert len(prep['source_sha256'])==36
    original_subjects={}
    for name,digest in prep['source_sha256'].items():
        path=SOURCE/name;assert sha(path)==digest,name
        ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
        original_subjects[pathstr(path)]=digest
    tools=[Path(__file__),BASE/'run_fit_durable_v1.ps1',BASE/'read_fit_progress_v1.ps1']
    original_subjects.update({pathstr(p):sha(p) for p in tools})
    source_role=review_role(selection['reviews']['source'],original_subjects,pins)
    previous=read(PRIOR/'training_frozen_inputs.json')
    for name,digest in previous['input_sha256'].items():
        assert sha(name)==digest,name
        pins[pathstr(name)]=digest
    for name,digest in previous['source_sha256'].items():
        path=Path(previous['source_directory'])/name
        assert sha(path)==digest,name;pins[pathstr(path)]=digest
    fixed=[PRIOR/'training_frozen_inputs.json',PRIOR/'training_runtime_identity.json',
        PRIOR/'generation/report.json',PRIOR/'generation/centers.npz',PRIOR/'generation/features.npy',
        PRIOR/'generation/base_target.npy',PRIOR/'generation/teacher_target.npy',PRIOR/'fit/final_predictions.npz',
        NEW/'fast_controller_phase_fit_v1/fit/teacher_fit.npz',NEW/'fast_controller_phase_fit_v1/nominal/trace.npz',
        NEW/'bfm_entry250_labels_v1/compatibility/fixed_pairs.json',BASE/'preparation_report_v2.json',
        BASE/'preparation_tests_v2_final.xml',BASE/'preservation_v2_derivation.json']
    for path in fixed:pins[pathstr(path)]=sha(path)
    runtime=read(PRIOR/'training_runtime_identity.json')
    assert runtime['python']=='3.10.11' and runtime['torch']=='2.10.0+cpu' and runtime['onnxruntime']=='1.23.2'
    for name,digest in runtime['binary_sha256'].items():
        assert sha(name)==digest,name;pins[pathstr(name)]=digest
    assert pathstr(PYTHON) in pins
    # All checks above precede any concrete output; failures never run the trainer.
    SNAPSHOT.mkdir()
    for name,digest in prep['source_sha256'].items():
        target=SNAPSHOT/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(SOURCE/name,target);assert sha(target)==digest
        pins[pathstr(target)]=digest
    # Literal reviewed digests match the byte-identical frozen copies.
    source_role['subjects']={pathstr(SNAPSHOT/name):digest for name,digest in prep['source_sha256'].items()}
    for path in tools:source_role['subjects'][pathstr(path)]=sha(path)
    request=dict(kind='one_physical_response_continuation_from70000',additional_updates=5000,ordinary_final_step=75000,
        coefficients=dict(nominal=1.,velocity=1.,physical=1.),valid_rows=V,budgets=budget(V),
        requested_rows=3054,nominal_rows=3057,requested_cell_denominators=[99,819,100]*3,coverage=coverage,
        exact_conflict_groups=0,compatibility=compatibility,selection_path=pathstr(selection_path),selection_sha256=args.selection_sha256,
        physical_paths=dict(manifest=pathstr(manifest_path),report=pathstr(producer_path),collection_request=pathstr(collection_request)),
        reviews=dict(prior_fit=fit_role,prior_export=export_role,branch_data=data_role,source=source_role),
        input_sha256=pins)
    for name,digest in pins.items():assert sha(name)==digest,name
    write_new(BASE/'training_request.json',request)
    receipt=dict(kind='one_fixed_physical_response_fit70001_75000',source_directory=pathstr(SNAPSHOT),
        source_sha256=prep['source_sha256'],input_sha256=pins,training_request_sha256=sha(BASE/'training_request.json'),
        additional_updates=5000,ordinary_final_step=75000,budgets=budget(V),
        selection_path=pathstr(selection_path),selection_sha256=args.selection_sha256)
    write_new(BASE/'training_frozen_inputs.json',receipt)
    launch_pins=dict(pins)
    for path in (BASE/'training_request.json',BASE/'training_frozen_inputs.json'):
        launch_pins[pathstr(path)]=sha(path)
    command=dict(python=pathstr(PYTHON),driver=pathstr(SNAPSHOT/'fit_physical_continuation.py'),
        working_directory=pathstr(SNAPSHOT),arguments=['-u',pathstr(SNAPSHOT/'fit_physical_continuation.py')],
        environment=dict(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=pathstr(SNAPSHOT)))
    plan=dict(kind='one_selected_physical_fit_launch_plan',command=command,budgets=budget(V),input_sha256=launch_pins,
        training_request_sha256=sha(BASE/'training_request.json'),frozen_receipt_sha256=sha(BASE/'training_frozen_inputs.json'),
        selection_path=pathstr(selection_path),selection_sha256=args.selection_sha256,
        root_data_audits=dict(physics=pathstr(physics),labels=pathstr(labels),conflicts=pathstr(conflicts)),
        requested_updates=5000,ordinary_final_step=75000,automatic_resume=False)
    write_new(BASE/'fit_launch_plan.json',plan)
    write_new(BASE/'freeze_report.json',dict(completed=True,source_count=36,input_count=len(pins),valid_rows=V,coverage=coverage,
        training_request_sha256=sha(BASE/'training_request.json'),frozen_receipt_sha256=sha(BASE/'training_frozen_inputs.json'),
        launch_plan_sha256=sha(BASE/'fit_launch_plan.json'),launch_clearance_created=False,
        model_evaluations=0,optimizer_updates=0,ORT_calls=0,physics_steps=0))
    print(json.dumps(read(BASE/'freeze_report.json')))

def finalize(args):
    assert_absent(('training_clearance.json','fit_launch_receipt.json','fit','fit_process',
                   'fit_preflight_failure.json','freeze_failure.json','finalize_failure.json'))
    plan=read(BASE/'fit_launch_plan.json');request=read(BASE/'training_request.json')
    selected(Path(plan['selection_path']),plan['selection_sha256'])
    for path,digest in plan['input_sha256'].items():assert sha(path)==digest,path
    review_path=args.final_review.resolve();assert sha(review_path)==args.final_review_sha256
    final_review=read(review_path);assert final_review['passed'] is True
    subjects=[BASE/'fit_launch_plan.json',BASE/'training_frozen_inputs.json',BASE/'training_request.json',
        Path(plan['selection_path']),Path(request['reviews']['branch_data']['path'])]
    for p in subjects:assert direct_hash(final_review,sha(p)),('Final review lacks direct subject',str(p))
    clearance=dict(approved=True,model_fitting_authorized=True,additional_updates=5000,ordinary_final_step=75000,
        frozen_receipt_sha256=sha(BASE/'training_frozen_inputs.json'),training_request_sha256=sha(BASE/'training_request.json'),
        source_review_path=request['reviews']['source']['path'],source_review_sha256=request['reviews']['source']['sha256'],
        final_launch_review_path=pathstr(review_path),final_launch_review_sha256=args.final_review_sha256,
        selection_path=plan['selection_path'],selection_sha256=plan['selection_sha256'],
        launcher_sha256=sha(BASE/'run_fit_durable_v1.ps1'),automatic_resume=False)
    write_new(BASE/'training_clearance.json',clearance)
    pins=dict(plan['input_sha256'])
    for p in (BASE/'fit_launch_plan.json',BASE/'freeze_report.json',BASE/'training_clearance.json',review_path):pins[pathstr(p)]=sha(p)
    launch=dict(kind='one_selected_physical_fit_launch',command=plan['command'],budgets=plan['budgets'],input_sha256=pins,
        requested_updates=5000,ordinary_final_step=75000,automatic_resume=False,
        clearance_sha256=sha(BASE/'training_clearance.json'),final_review_path=pathstr(review_path),
        final_review_sha256=args.final_review_sha256,launch_plan_sha256=sha(BASE/'fit_launch_plan.json'))
    write_new(BASE/'fit_launch_receipt.json',launch)
    print(json.dumps(dict(finalized=True,launch_receipt_sha256=sha(BASE/'fit_launch_receipt.json'),
                         process_launched=False,model_evaluations=0,optimizer_updates=0)))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='mode',required=True)
    f=commands.add_parser('freeze');f.add_argument('--selection',type=Path,required=True)
    f.add_argument('--selection-sha256',type=digest_arg,required=True)
    final=commands.add_parser('finalize');final.add_argument('--final-review',type=Path,required=True)
    final.add_argument('--final-review-sha256',type=digest_arg,required=True)
    args=parser.parse_args()
    try:freeze(args) if args.mode=='freeze' else finalize(args)
    except BaseException as exc:
        path=BASE/(args.mode+'_failure.json')
        if not path.exists():write_new(path,dict(completed=False,exception=type(exc).__name__,message=str(exc),
            traceback=traceback.format_exc(),no_launch=True,optimizer_updates=0,model_evaluations=0,physics_steps=0))
        raise
