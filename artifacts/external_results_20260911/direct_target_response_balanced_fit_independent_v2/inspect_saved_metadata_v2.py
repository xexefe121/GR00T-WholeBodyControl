"""Saved JSON schema relationships only; no arrays/checkpoints or metric reconstruction."""
from pathlib import Path
import ast,hashlib,json
BASE=Path(__file__).resolve().parent;FIT=BASE.parent/'direct_target_causal_response_balanced_student_v2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
tracked={};checks=[]
def read(p):
    p=Path(p);tracked[p.as_posix()]=sha(p);return json.loads(p.read_text(encoding='utf-8-sig'))
def check(name,ok):
    checks.append(dict(name=name,passed=bool(ok)))
    if not ok:raise AssertionError(name)
source=BASE/'source_prepared_v1/audit_saved_warm.py';tracked[source.as_posix()]=sha(source)
t=ast.parse(source.read_text());request=read(FIT/'training_request.json');frozen=read(FIT/'training_frozen_inputs.json')
folder=FIT/'fit';shared=folder/'shared';report=read(folder/'report.json');restoration=read(folder/'restoration.json')
env=dict(condition='causal',request=request,WEIGHT=1.8188207859141674)
def loops_with_marker(marker):
    for n in ast.walk(t):
        if not isinstance(n,ast.For) or not isinstance(n.iter,ast.Call) or not isinstance(n.iter.func,ast.Attribute) or n.iter.func.attr!='items':continue
        if any(isinstance(v,ast.Constant) and v.value==marker for statement in n.body for v in ast.walk(statement)):
            yield n
for marker in ('_report.','report_extra.'):
    loop=next(loops_with_marker(marker));mapping=eval(compile(ast.Expression(loop.iter.func.value),'<metadata mapping>','eval'),env)
    for field,wanted in mapping.items():check('report:'+field,report[field]==wanted)
env.update(maximum=restoration['initial_drift']['max_preclip_error_rad'],restored=restoration['initial_drift']['corpora'])
restoration_expression=next(n.args[-1] for n in ast.walk(t) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
    and n.func.id=='compare' and n.args and isinstance(n.args[0],ast.BinOp)
    and isinstance(n.args[0].right,ast.Constant) and n.args[0].right.value=='_restoration')
expected=eval(compile(ast.Expression(restoration_expression),'<saved restoration metadata>','eval'),env)
check('restoration_exact_metadata_schema',expected==restoration)
check('report_initial_restoration',report['initial_restoration']==restoration['initial_drift'])
check('report_subject_map',report['direct_subject_sha256']=={k:s['sha256'] for k,s in request['subjects'].items()})
check('report_loss_columns',report['training_loss_columns']==['nominal','balanced_full_state','physical','balanced_total','learning_rate','preclip_gradient_norm'])
rule='mean_six_teacher_group_energies_over_group_energy'
check('report_rule',report['response_weight_rule']==rule)
for backend in ('initial_GPU32','final_GPU32','CPU64','GPU64','ORT64'):
    metrics=read(folder/(backend+'_metrics.json'))
    check('metric_literal_metadata:'+backend,metrics['response_weight_rule']==rule and metrics['optimization_objective']=='balanced_weighted_objective')
    check('report_contains_identical_metric_record:'+backend,report['metrics'][backend]==metrics)
    check('metric_group_schema:'+backend,len(metrics['balanced_full_state_cells'])==54 and len(metrics['balanced_full_state_group_comparison'])==6)
optimization=read(folder/'optimization_completed.json')
check('ordinary_endpoint_metadata',optimization==dict(optimization_completed=True,condition='causal',ordinary_final_step=71000,optimizer_step=6000,checkpoint_sha256=report['checkpoint_sha256'],export_validation_pending=True))
clearance=read(FIT/'training_clearance.json');review=read(clearance['review_path'])
for field,key in [('training_request_sha256','request_sha256'),('frozen_receipt_sha256','frozen_receipt_sha256'),('launcher_sha256','launcher_sha256')]:check('concrete:'+field,review[field]==clearance[key])
check('concrete_source_schema',review['source_sha256']==frozen['source_sha256'])
start=read(FIT/'fit_process_v1/start.json');child=read(FIT/'fit_process_v1/child.json');end=read(FIT/'fit_process_v1/exit.json');owner=read(FIT/'owner_completion_verification.json')
check('producer_process_metadata',start['wrapper_pid']==child['wrapper_pid']==end['wrapper_pid'] and child['child_pid']==end['child_pid'] and child['captured_handle_nonzero'] is True)
check('owner_process_metadata',owner['expected_pids']==[end['wrapper_pid'],end['child_pid']] and owner['raw_python_exit_code']==owner['exit_code']==end['exit_code']==0)
check('owner_single_roles',len(owner['direct_subject_sha256'])==19 and owner['condition']=='causal')
runtime=read(shared/'runtime.json')
check('runtime_schema',all(k in runtime for k in ('deterministic_algorithms','matmul_TF32','cudnn_TF32','AMP','CUBLAS_WORKSPACE_CONFIG')))
promotion=read(folder/'promoted_parameters.json');check('promotion_metadata_schema',set(promotion)=={'parameters','checkpoint_sha256'} and len(promotion['parameters'])==8)
parity=read(folder/'export_parity.json');check('parity_metadata_schema',set(('comparisons','maximum_preclamp_rad','tolerance_rad','nonfinite_comparisons','passed'))<=set(parity) and len(parity['comparisons'])==9)
for path,digest in tracked.items():check('unchanged:'+path,sha(path)==digest)
result=dict(metadata_inspection_passed=True,checks=len(checks),input_sha256=tracked,task_array_loads=0,task_checkpoint_loads=0,
    model_calls=0,ORT_calls=0,gradient_calls=0,optimizer_updates=0,native_steps=0,actual_numerical_audit_repeated=False,
    limitation='Checks saved JSON identities, producer schemas and isolated expected metadata AST. Does not reconstruct metrics or qualify fit evidence.')
with (BASE/'saved_metadata_inspection.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(dict(checks=len(checks),sha256=sha(BASE/'saved_metadata_inspection.json'))))
