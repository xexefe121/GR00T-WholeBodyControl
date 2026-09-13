"""Record completed independent source review; no task data/model execution."""
from pathlib import Path
import hashlib,json,ast,xml.etree.ElementTree as ET
ROOT=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911')
BASE=ROOT/'direct_target_causal_context_study_v1'
DEST=Path(__file__).parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
prep=read(BASE/'source_preparation.json');proof=read(BASE/'context_preflight/report.json')
assert prep['source_preparation_passed'] is True and prep['preparation_only'] is True and proof['passed'] is True
assert len(prep['source_sha256'])==19
checked={}
for name,digest in prep['source_sha256'].items():
    p=Path(prep['source_directory'])/name
    assert sha(p)==digest==sha(BASE/'source_draft_v1'/name)
    ast.parse(p.read_text(encoding='utf-8-sig'))
    checked[p.as_posix()]=digest
    assert proof['source_sha256'][name]==digest
for name in prep['unchanged_original_modules']:
    assert sha(ROOT/'direct_target_full_state_student_v1/source_snapshot_v1'/name)==prep['source_sha256'][name]
assert len(prep['unchanged_original_modules'])==10
subjects={}
for name,s in prep['subjects'].items():
    assert sha(s['path'])==s['sha256'];subjects[name]=s
for name in ('source_preparation.json','root_context_synthetic_tests_v1.xml'):
    p=BASE/name;subjects[name]=dict(path=p.as_posix(),sha256=sha(p))
for name in ('synthetic_tests_v3.xml','root_context_synthetic_tests_v1.xml'):
    root=ET.parse(BASE/name).getroot();suites=[root] if root.tag=='testsuite' else root.findall('testsuite')
    assert sum(int(s.get('tests','0')) for s in suites)==15
    assert all(int(s.get(k,'0'))==0 for s in suites for k in ('failures','errors','skipped'))
assert proof['proof']==prep['context_proof']
assert proof['proof']['nominal_chronological_history_and_prior_pairs']==9899
assert proof['proof']['physical_applied_prior_differs_raw_rows']==1463
assert proof['proof']['physical_applied_prior_differs_raw_components']==1862
assert proof['proof']['physical_native_clipped_rows']==1459
assert len(proof['input_sha256'])==229 and proof['all_inputs_unchanged'] is True
assert proof['input_sha256']==read(BASE/'preparation_inputs.json')['input_sha256']
assert proof['feature_proof']=={c+'_'+k:n for c in ('blinded','causal') for k,n in [('features',9904),('full_state_features',354612),('physical_features',3054)]}
for name,digest in proof['output_sha256'].items():
    p=BASE/'context_preflight'/name;assert sha(p)==digest;checked[p.as_posix()]=digest
proposal=read(BASE/'training_request_proposal.json')
assert proposal['root_selected'] is False and proposal['coefficient']==1.8188207859141674
assert proposal['conditions']==['blinded','causal'] and proposal['learning_rate']==[1e-5,1e-6]
assert proposal['updates_per_condition']==3000 and proposal['ordinary_final_step']==68000
main=(Path(prep['source_directory'])/'train_context_pair.py').read_text()
assert 'options.inter_op_num_threads=1' in main and 'options.inter_op_threads=' not in main
record=dict(passed=True,source_review_pass=True,context_data_review_pass=True,preparation_only=True,
    prelaunch_review_pass=False,task_training_authorized=False,controller_cleared=False,
    source_directory=prep['source_directory'],source_sha256=prep['source_sha256'],subjects=subjects,
    source_preparation_sha256=sha(BASE/'source_preparation.json'),context_proof_sha256=sha(BASE/'context_preflight/report.json'),
    direct_subject_sha256={k:s['sha256'] for k,s in subjects.items()},checked_file_sha256=checked,
    unchanged_original_modules=prep['unchanged_original_modules'],
    review_scope=['All19 Python sources;10 original utilities preserved exactly. Fixed2x3000 lower-LR paired protocol, exact65000 actor/RNG and original1000 normalization, fresh AdamW, zero-column expansion.',
      'All9904 nominal prior/history rows mapped;9899 contiguous history and actual-target inverse transitions verified by bound pure proof. Five first-row histories remain immutable original-source context.',
      'All3054 physical contexts use inverse actually applied target and exactly-once advanced history;1463 rows differ from old raw prior. Full58 endpoints retain center context.',
      'Unchanged15/54/9 cell losses, coefficient1.8188207859141674, saved first3000 schedule, no recalibration; complete per-condition diagnostics, actual optimizer/call counters and failure-prefix preservation.',
      'Same1323 architecture for both conditions, normalized-zero blinded context and zero unused-column guard; changed MatMul reduction drift disclosed and gated at original1e-5rad.',
      'Manual exact-parameter FP64 internal/publicFP32 export, fixed budget and no endpoint selection. Final actual request/launcher review remains required.'],
    fixed_findings=['Tuple-index test corrected.','Nominal chronological history/prior guard added and passed on all9899 transitions.','Invalid ORT inter_op_threads replaced by inter_op_num_threads with actual SessionOptions attribute regression.','Output schema now matches causal-only initial comparison, separate paired failure, aggregate pair deltas and retained condition cell metrics.'],
    tests=dict(owner_synthetic_passed=15,root_independent_synthetic_passed=15,tests_repeated_by_this_review=0),
    limitations=['No new full input-tree rehash or repeated saved-data proof in this source review; original229-pin unchanged proof and direct output hashes bound.',
      'Zero-column expansion is mathematical function preservation, not a byte-equality guarantee across changed GPU reduction dimensions.',
      'This matched study tests recorded context utility under its fixed protocol; it cannot uniquely establish hidden-state causation, sufficient capacity, or closed-loop stability.'],
    model_calls=0,ORT_calls=0,gradient_calls=0,optimizer_updates=0,native_steps=0)
with (DEST/'review.json').open('x',encoding='utf-8') as f:json.dump(record,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(dict(review_path=(DEST/'review.json').as_posix(),sha256=sha(DEST/'review.json'),source_review_pass=True,context_data_review_pass=True)))
