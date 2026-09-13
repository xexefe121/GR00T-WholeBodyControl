"""Freeze the root-selected single continuation without model calls or updates."""
import ast, hashlib, json, shutil
from pathlib import Path

BASE=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911/velocity_chord_student_v1')
NEW=BASE.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')

snap=BASE/'source_snapshot_v3'
assert not snap.exists() and not (BASE/'training_frozen_inputs.json').exists()
generation=read(BASE/'generation_frozen_inputs_v2.json')
snap.mkdir()
for name,digest in generation['source_sha256'].items():
    original=BASE/'source_snapshot_v2'/name
    assert sha(original)==digest,name
    target=snap/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original,target)
for original in sorted((BASE/'training_draft_v1').glob('*.py')):
    target=snap/original.name;assert not target.exists();shutil.copyfile(original,target)
sources={str(p.relative_to(snap)).replace('\\','/'):sha(p) for p in sorted(snap.rglob('*.py'))}
for p in snap.rglob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
inputs=dict(generation['input_sha256'])
data_review=NEW/'velocity_chord_completed_data_review_v1/review.json'
review=read(data_review);assert review['verdict']=='CLEAR'
for name,digest in review['evidence_hashes'].items():
    assert sha(name)==digest,name
    inputs[str(Path(name)).replace('\\','/')]=digest
for name,digest in review['output_hashes'].items():
    p=BASE/'generation'/name;assert sha(p)==digest,name;inputs[str(p).replace('\\','/')]=digest
extra=[data_review,BASE/'run_fit_durable.ps1',BASE/'training_runtime_identity.json',
    BASE/'chord_objective_pure_tests_v2.json',BASE/'diagnostic_preservation_pure_tests.json',
    BASE/'generation_owner_audit_request.json',BASE/'generation/center_parity.json',
    NEW/'velocity_chord_design_review_v1/review.json',NEW/'velocity_chord_audit_source_review_v1/review.json',
    NEW/'learner_stability_design_v1/proposal.json',NEW/'learner_stability_design_v1/PROPOSAL.md',
    NEW/'fast_controller_phase_fit_v1/fit/student_head.pt',NEW/'fast_controller_phase_fit_v1/fit/student_head.onnx',
    NEW/'fast_controller_phase_fit_v1/fit/teacher_fit.npz',NEW/'fast_controller_phase_fit_v1/fit/report.json',
    NEW/'fast_controller_phase_fit_v1/nominal/trace.npz',NEW/'bfm_entry250_labels_v1/compatibility/fixed_pairs.json',
    Path(__file__)]
for p in extra:inputs[str(p).replace('\\','/')]=sha(p)
for name,digest in read(BASE/'training_runtime_identity.json')['binary_sha256'].items():
    assert sha(name)==digest,name;inputs[str(Path(name)).replace('\\','/')]=digest
for name,digest in inputs.items():assert sha(name)==digest,name
write(BASE/'training_frozen_inputs.json',dict(kind='one_fixed_velocity_chord_continuation_65001_70000',
    source_directory=str(snap),source_sha256=sources,input_sha256=inputs,
    selected_proposal_sha256=sha(NEW/'learner_stability_design_v1/proposal.json'),
    dataset_review_sha256=sha(data_review),root_data_audit_sha256=sha(NEW/'velocity_chords_independent_v1/report.json'),
    nominal_rows=3057,pairs_per_update=576,additional_updates=5000,ordinary_final_step=70000,
    maximum_head_ONNX_calls=1126,BFM_calls=0,physics_steps=0,broader_data_included=False))
write(BASE/'training_clearance_DRAFT.json',dict(approved=False,training_only=True,additional_updates=5000,
    ordinary_final_step=70000,maximum_head_ONNX_calls=1126,
    frozen_receipt_sha256=sha(BASE/'training_frozen_inputs.json'),launcher_sha256=sha(BASE/'run_fit_durable.ps1'),
    selected_proposal_sha256=sha(NEW/'learner_stability_design_v1/proposal.json'),
    dataset_review_path=str(data_review),dataset_review_sha256=sha(data_review),
    root_data_audit_path=str(NEW/'velocity_chords_independent_v1/report.json'),
    root_data_audit_sha256=sha(NEW/'velocity_chords_independent_v1/report.json'),
    source_review_path=None,source_review_sha256=None))
print(json.dumps(dict(source_count=len(sources),input_count=len(inputs),
    receipt_sha256=sha(BASE/'training_frozen_inputs.json'),launcher_sha256=sha(BASE/'run_fit_durable.ps1'))))
