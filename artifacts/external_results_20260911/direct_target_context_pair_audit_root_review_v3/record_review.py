"""Review saved-auditor study-path and exact split-metadata alignment only."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'direct_target_context_pair_fit_independent_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
prior=HERE.parent/'direct_target_context_pair_audit_root_review_v2/review.json'
assert sha(prior)=='ee211499592f0e26c4189aee44cb14821d7b9075eacf69b1f51de5b09bcc2fc8'
prep=BASE/'source_preparation_v3.json'
assert sha(prep)=='396d0123aae97b85084e4653a46fe5f36397951cb1eb4c340b655d3732b47d95'
p=read(prep);assert p['source_preparation_passed'] is True
for name,digest in p['source_sha256'].items():assert sha(BASE/'source_draft_v3'/name)==digest
for name,digest in p['helper_sha256'].items():assert sha(BASE/name)==digest
for name in p['unchanged_v2_sources']:assert (BASE/'source_draft_v2'/name).read_bytes()==(BASE/'source_draft_v3'/name).read_bytes()
old=(BASE/'source_draft_v2/audit_saved_pair.py').read_text()
expected=old.replace("public_output_dtype='float32',parity_tolerance_rad=1e-5", "public_output_dtype='float32',training_first_layer_execution='split_contiguous_1000_plus_323',export_first_layer_execution='monolithic_float64_1323',parity_tolerance_rad=1e-5")
expected=expected.replace('changed_MatMul_dimension=True,byte_gate_required=False',"changed_MatMul_dimension=False,stored_first_layer_width=1323,first_layer_execution='split_contiguous_1000_plus_323',byte_gate_required=False")
assert expected==(BASE/'source_draft_v3/audit_saved_pair.py').read_text()
specs={
 'prepare_audit_request': [('direct_target_causal_context_study_v1','direct_target_causal_context_study_v2'),('source_draft_v2','source_draft_v3')],
 'freeze_launch': [('run_audit_durable_v2.ps1','run_audit_durable_v3.ps1'),('prepare_audit_request_v2.py','prepare_audit_request_v3.py'),('source_preparation_v2.json','source_preparation_v3.json'),('verify_completion_v2.py','verify_completion_v3.py')],
 'run_audit_durable': [('source_draft_v2','source_draft_v3')],
 'verify_completion': [('source_preparation_v2.json','source_preparation_v3.json')]}
for stem,replacements in specs.items():
    extension='.ps1' if stem=='run_audit_durable' else '.py'
    text=(BASE/(stem+'_v2'+extension)).read_text()
    for before,after in replacements:text=text.replace(before,after)
    assert text==(BASE/(stem+'_v3'+extension)).read_text(),stem
value=dict(read(prior))
value.update(source_sha256=p['source_sha256'],helper_sha256=p['helper_sha256'],
    source_preparation=dict(path=str(prep),sha256=sha(prep)),preceding_root_review=dict(path=str(prior),sha256=sha(prior)),
    reviewed=read(prior)['reviewed']+['Root read and mechanically checked complete v3 delta: experiment path, helper/source filenames, exact split restoration/report fields only.'],
    inherited_numerical_tests_unchanged=True,actual_pair_audit_selected=True,
    selected_scope='One saved-only audit after both split-study conditions and owner3 complete; no model/native/optimizer calls; fail on mismatch, no automatic retry.')
out=HERE/'review.json'
with out.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2);f.write('\n')
print(json.dumps({'source_review_pass':True,'path':str(out),'sha256':sha(out)}))
