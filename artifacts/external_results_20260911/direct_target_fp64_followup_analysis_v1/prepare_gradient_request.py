"""Freeze a DRAFT request only, with no task model imports or execution."""
from pathlib import Path
import ast
import hashlib
import json

BASE = Path(__file__).resolve().parent
NEW = BASE.parent
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def main():
    original=NEW/'direct_target_continuation_v1'
    receipt_path=original/'training_frozen_inputs.json'
    receipt=read(receipt_path)
    pins=dict(receipt['input_sha256'])
    source_dir=Path(receipt['source_directory'])
    for name,digest in receipt['source_sha256'].items():
        pins[(source_dir/name).as_posix()]=digest
    roles=dict(checkpoint=original/'fit/student_head.pt',normalization=original/'fit/normalization.npz',
        fit_report=original/'fit/report.json',training_request=original/'training_request.json',
        original_frozen_receipt=receipt_path,
        saved_analysis=BASE/'saved_scale_report.json',
        training_evidence=NEW/'direct_target_continuation_training_only_review_v1/review.json',
        promoted_export=NEW/'direct_target_fp64_export_v1/export/report.json',
        actual_failure=NEW/'direct_target_fp64_export_evaluation_v2/nominal/report.json')
    subjects={key:dict(path=path.as_posix(),sha256=sha(path)) for key,path in roles.items()}
    for value in subjects.values():pins[value['path']]=value['sha256']
    for path,digest in pins.items():
        if sha(path)!=digest:raise ValueError('Changed existing input: '+path)
    source=BASE/'gradient_diagnostic_draft.py'
    tree=ast.parse(source.read_text(encoding='utf-8'))
    # Static sanity only; this does not execute or prove runtime behavior.
    forbidden={'step','backward','mj_step','InferenceSession','export_onnx'}
    calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
    if forbidden.intersection(calls):raise ValueError('Unexpected API in diagnostic.')
    request=dict(kind='one_fixed_current55000_gradient_diagnostic',prepared_only=True,ordinary_final_step=55000,
        schedule_row=0,source_sha256=sha(source),original_source_directory=source_dir.as_posix(),
        subjects=subjects,input_sha256=pins,
        runtime=read(original/'training_request.json')['runtime'],
        budgets=dict(forward_calls=3,forward_rows=14110,gradient_calls=3,optimizer_updates=0,native_calls=0,ORT_calls=0),
        fixed_weight_compositions=[1,10000],new_parameter_updates=0,
        scope='parameter gradients of original float32 training objective; one fixed batch, not exported FP64 task inference',
        existing_FP32_export_failed_preserved=True,automatic_retry=False)
    destination=BASE/'gradient_request_draft.json'
    with destination.open('x',encoding='utf-8') as f:json.dump(request,f,indent=2);f.write('\n')
    print(json.dumps(dict(prepared_only=True,source_sha256=sha(source),request_sha256=sha(destination),input_pins=len(pins),model_calls=0)))

if __name__=='__main__':main()
