from pathlib import Path
import ast
import hashlib
import json
import shutil

BASE=Path(__file__).resolve().parent;NEW=BASE.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')

def main():
    assert not (BASE/'generation_frozen_inputs.json').exists()
    draft=BASE/'source_draft_v1';dest=BASE/'source_snapshot_v1'
    dest.mkdir(exist_ok=False)
    for p in sorted(draft.rglob('*.py')):
        ast.parse(p.read_text());to=dest/p.relative_to(draft);to.parent.mkdir(exist_ok=True,parents=True);shutil.copyfile(p,to)
    phase=read(NEW/'fast_controller_phase_fit_v1/frozen_inputs_v2.json')
    for name,digest in phase['source_sha256'].items():assert sha(dest/name)==digest,name
    inputs={}
    def add(p):inputs[str(Path(p)).replace('\\','/')]=sha(p)
    for name,digest in phase['input_sha256'].items():
        assert sha(name)==digest,name
        inputs[name.replace('\\','/')]=digest
    for name,digest in read(NEW/'three_expert_feedback_audit_v1/input_hashes.json').items():
        assert sha(name)==digest,name
        inputs[name.replace('\\','/')]=digest
    for rel in ['learner_stability_design_v1/proposal.json','learner_stability_design_v1/PROPOSAL.md',
        'learner_stability_design_v1/scales.json','velocity_chord_design_review_v1/review.json',
        'fast_controller_phase_fit_v1/fit/student_head.pt','fast_controller_phase_fit_v1/fit/student_head.onnx',
        'fast_controller_phase_fit_v1/frozen_inputs_v2.json','three_expert_feedback_audit_v1/report.json',
        'three_expert_feedback_audit_v1/per_control.json','three_expert_feedback_audit_v1/input_hashes.json',
        'bfm_entry250_labels_v1/compatibility/report.json','bfm250_expert_labels_independent_v1/report.json']:
        add(NEW/rel)
    add(BASE/'saved_center_pure_test_v2.json');add(BASE/'run_generation_durable.ps1');add(Path(__file__))
    proposal=read(NEW/'learner_stability_design_v1/proposal.json')
    expected='2beea3bbd343393a494958a7086a5d3f0cde91e0345eb9e21512bfed33c38cf5'
    assert sha(NEW/'learner_stability_design_v1/proposal.json')==expected
    result=dict(kind='frozen_one_velocity_chord_generation',source_directory=str(dest),
        source_sha256={str(p.relative_to(dest)).replace('\\','/'):sha(p) for p in sorted(dest.rglob('*.py'))},
        input_sha256=inputs,selected_proposal_sha256=expected,original18_phase_sources_byteexact=True,
        centers=3057,probes=140622,sign_order=[-1,1],radius_native_cap_fraction=.01,
        actor_calls=143679,backward_calls=3057,total_BFM_ONNX_calls=146736,
        all_center_parity_required_before_first_probe=True,full_matrix_teacher_operation_order=True,
        optimizer_updates=0,physics_steps=0,review_required_before_generation=True,hardware_authorized=False)
    write(BASE/'generation_frozen_inputs.json',result)
    write(BASE/'generation_clearance_DRAFT.json',dict(approved=False,generation_only=True,
        selected_proposal_sha256=expected,frozen_receipt_sha256=sha(BASE/'generation_frozen_inputs.json'),
        launcher_sha256=sha(BASE/'run_generation_durable.ps1'),review_path=None,review_sha256=None))
    print(json.dumps(dict(sources=len(result['source_sha256']),inputs=len(inputs),receipt_sha256=sha(BASE/'generation_frozen_inputs.json'))))

if __name__=='__main__':main()
