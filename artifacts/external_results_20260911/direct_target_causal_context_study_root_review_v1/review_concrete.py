"""Root concrete selected-pair review; hashes and receipts only, no task execution."""
import argparse
import hashlib
import json
from pathlib import Path

BASE=Path(__file__).resolve().parent
TASK=BASE.parent/'direct_target_causal_context_study_v1'
def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def main():
    p=argparse.ArgumentParser()
    for name in ('request-sha','frozen-sha','launcher-sha','launcher'):p.add_argument('--'+name,required=True)
    args=p.parse_args()
    request_path=TASK/'training_request.json';frozen_path=TASK/'training_frozen_inputs.json';launcher=Path(args.launcher)
    assert sha(request_path)==args.request_sha and sha(frozen_path)==args.frozen_sha and sha(launcher)==args.launcher_sha
    request=read(request_path);frozen=read(frozen_path);proposal=read(TASK/'training_request_proposal.json')
    prep_path=TASK/'source_preparation.json';prep=read(prep_path)
    assert sha(prep_path)=='210b8cfeee5f180493cc8e38a3644e019aad90967fcb7f4ece11b9b59616e638'
    independent_path=BASE.parent/'direct_target_context_source_review_v1/review.json'
    independent=read(independent_path)
    assert sha(independent_path)=='a5fb96bdc1c192454966dfe6f75a08a1ef371cf9f65af59ef1e62d33e3b1c58e'
    assert independent['source_review_pass'] is True and independent['context_data_review_pass'] is True
    assert request['root_selected'] is True and proposal['root_selected'] is False
    for key,value in proposal.items():
        if key not in ('root_selected','subjects','pending_before_actual_fit'):assert request[key]==value,key
    assert request['pending_before_actual_fit']==['root concrete request/launcher clearance']
    for key,value in proposal['subjects'].items():assert request['subjects'][key]==value,key
    assert request['conditions']==['blinded','causal'] and request['updates_per_condition']==3000
    assert request['coefficient']==1.8188207859141674 and request['coefficient_recalibration'] is False
    assert request['learning_rate']==[1e-5,1e-6] and request['ordinary_final_step']==68000
    assert request['budgets']['training_forward_rows']==88116000 and request['budgets']['training_updates']==6000
    assert request['budgets']['calibration_forward_calls']==request['budgets']['calibration_gradient_calls']==0
    assert frozen['training_request_sha256']==args.request_sha
    assert frozen['source_sha256']==prep['source_sha256']==independent['source_sha256']
    source=Path(frozen['source_directory'])
    assert {path.name:sha(path) for path in source.glob('*.py')}==frozen['source_sha256']
    checked={}
    for path,digest in frozen['input_sha256'].items():
        assert sha(path)==digest,path
        checked[path]=digest
    proof_path=TASK/'context_preflight/report.json';proof=read(proof_path)
    assert sha(proof_path)=='f5add6c31a3c3aec9b215d837d86e62a8a9bffcac4575d096d24012296c04583'
    assert proof['passed'] is True and proof['proof']['nominal_chronological_history_and_prior_pairs']==9899
    assert proof['proof']['physical_actual_prior_exact'] is True
    for name,digest in proof['output_sha256'].items():assert sha(proof_path.parent/name)==digest,name
    assert not (TASK/'fit').exists() and not (TASK/'training_clearance.json').exists()
    result=dict(prelaunch_review_pass=True,root_selected_single_pair=True,training_request_sha256=args.request_sha,
        frozen_receipt_sha256=args.frozen_sha,launcher_path=str(launcher),launcher_sha256=args.launcher_sha,
        subjects=dict(training_request=dict(path=str(request_path),sha256=args.request_sha),
                      frozen_inputs=dict(path=str(frozen_path),sha256=args.frozen_sha),
                      launcher=dict(path=str(launcher),sha256=args.launcher_sha)),
        source_sha256=frozen['source_sha256'],input_sha256=checked,
        independent_source_review=dict(path=str(independent_path),sha256=sha(independent_path)),
        source_preparation_sha256=sha(prep_path),completed_context_proof_sha256=sha(proof_path),
        root_synthetic_tests=dict(passed=15,path=str(TASK/'root_context_synthetic_tests_v1.xml'),
                                 sha256=sha(TASK/'root_context_synthetic_tests_v1.xml')),
        reviewed=['Root read all new context/data/model/diagnostic/export/fit sources and preserved-helper identities.',
                  'Fixed matched two-condition protocol, zero calibration, same65000 weights/RNG, fresh lower-LR AdamW.',
                  'Applied physical prior, incoming nominal history, held center context and every original1000 prefix.',
                  'Failure-prefix and actual-call accounting, ordinary final checkpoints, complete fixed diagnostics.',
                  'Concrete durable launcher read and PowerShell syntax checked separately before this review.'],
        updates_per_condition=3000,conditions=['blinded','causal'],training_forward_rows=88116000,
        model_calls_executed_by_review=0,native_steps=0,optimizer_updates=0,controller_selected=False)
    output=BASE/'review.json'
    with output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(path=str(output),sha256=sha(output),input_pins=len(checked),passed=True)))

if __name__=='__main__':main()
