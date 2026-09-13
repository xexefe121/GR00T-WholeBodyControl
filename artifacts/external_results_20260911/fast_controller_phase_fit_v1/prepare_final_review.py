"""Bind completed ordinary final artifacts for read-only review, no inference."""
from pathlib import Path
import hashlib
import json
BASE=Path(__file__).resolve().parent
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    dest=BASE/'final_export_clearance_DRAFT.json';assert not dest.exists()
    status=read(BASE/'fit_process_status.json')
    assert status['state']=='EXITED' and status['process_exit_code']==0
    report=read(BASE/'fit/report.json');validation=read(BASE/'final_export_validation.json')
    assert validation['pass_'] and validation['ordinary_final_global_step']==65000
    assert report['steps']==65000 and report['additional_updates']==5000
    assert report['full_objective_improved'] and report['export_parity_passed'] and report['rollout_numerical_prerequisites_pass']
    frozen=read(BASE/'frozen_inputs_v2.json')
    for name,expected in frozen['source_sha256'].items():assert sha(BASE/'source_snapshot_v1'/name)==expected
    for name,expected in frozen['input_sha256'].items():assert sha(Path(name))==expected
    bindings={}
    def bind(path,expected=None):
        path=Path(path);actual=sha(path)
        if expected is not None:assert expected==actual
        bindings[str(path)]=actual
    for name,expected in validation['files_sha256'].items():bind(name,expected)
    for path in [BASE/name for name in ('run_canonical_durable.ps1','final_export_validation.json',
        'training_clearance.json','fit_launch_receipt.json','fit_process_status.json','frozen_inputs_v2.json',
        'source_snapshot_v1/evaluate_phase_student.py','source_snapshot_v1/student_linear_runtime.py',
        'source_snapshot_v1/fit_phase_fullbatch_once.py')]+[Path(__file__)]:bind(path)
    result=dict(kind='ordinary_final65000_canonical_export_clearance_draft',canonical_rollout_authorized=False,
        source_review_pending=True,ordinary_final_global_step=65000,additional_updates=5000,
        full_objective_improved=True,export_parity_passed=True,frozen_sources_sha256=sha(BASE/'frozen_inputs_v2.json'),
        bound_files=[dict(path=path,sha256=digest) for path,digest in bindings.items()],
        intended_runtime='Fresh originalBFM controls0..249, learnedresidual250..1268, frozen terminalBFMyaw4 from1269;1569 lifecycle plus continuous250 hold.',
        training_scope='Original/query1/qualifiedBFM250 acquisition-source-return labels only; no initial standing labels.',
        transition_gate='Generate own baseline250; compare exact prefix plus actual query250 features/base/prior/history/state/integration before first learned physics.',
        canonical_trials=1,new_expert_queries=0,additional_model_fit_after65000=False,
        checkpoint_selection=False,expert_target_replay=False,hardware_authorized=False)
    dest.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(draft_sha256=sha(dest),bound_files=len(bindings),head_sha256=report['checkpoints']['student_head.onnx'])))

if __name__=='__main__':main()
