"""Bind completed pure semantic/map results and preserve their limits."""
import hashlib,json
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;NEW=BASE.parent
pins={}
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def native(p):
    s=str(p).replace('\\','/')
    if s.startswith('/mnt/') and len(s)>7:s=s[5].upper()+':'+s[6:]
    return Path(s)
def bind(p,expected=None):
    p=native(p).resolve();digest=sha(p)
    if expected is not None:assert digest==expected,(p,digest,expected)
    pins[p.as_posix()]=digest;return p
def read(p,expected=None):return json.loads(bind(p,expected).read_text(encoding='utf-8-sig'))
def subject(p):p=bind(p);return dict(path=p.as_posix(),sha256=pins[p.as_posix()])
semantic=read(NEW/'direct_target_saved_outcome_review_v1/report.json','a8e3b595db06bd4638fd3dbf623bfdb13cbd898bb8f7b425e026895451da0f02')
assert semantic['passed'] is True and semantic['moving_controls']==66 and semantic['exact_checks']==5691
assert semantic['first_feature_departure']==251 and semantic['first_learned_target_clip']==270 and semantic['learned_clipped_commands']==46
assert semantic['inference_calls']==semantic['BFM_calls']==semantic['physics_steps']==semantic['optimizer_updates']==0
srequest=read(NEW/'direct_target_saved_outcome_review_v1/request.json',semantic['request_sha256'])
bind(NEW/'direct_target_saved_outcome_review_v1/audit_saved.py',semantic['source_sha256'])
for path,digest in srequest['input_sha256'].items():bind(path,digest)
maps=read(BASE/'report.json','c8c278b57ef5dcb8735c13af484faf7bfd398ac0f4bdebfab653fdba4add6726')
request=read(BASE/'request.json',maps['request_sha256'])
bind(BASE/'diagnose_fixed_map.py',maps['source_sha256']);check=read(BASE/'source_check.json')
assert check['passed'] is True and check['derived_source_sha256']==maps['source_sha256']
for path,digest in request['input_sha256'].items():bind(path,digest)
assert maps['passed'] is True and maps['nominal_map_checks_exact']==maps['actual_state_map_evaluations']==66
assert [row['control'] for row in maps['rows']]==list(range(250,316))
assert maps['model_inference_calls']==maps['physics_steps']==maps['optimizer_updates']==maps['new_queries']==0
assert maps['all_inputs_unchanged'] is True and semantic['all_inputs_unchanged'] is True
with np.load(bind(BASE/'arrays.npz',maps['arrays_sha256']),allow_pickle=False) as a:
    assert a['map_target'].shape==(66,23) and a['actual_tangent'].shape==(66,58)
    assert np.isfinite(a['map_target']).all() and np.isfinite(a['actual_tangent']).all()
    assert np.array_equal(a['control'],np.arange(250,316))
subset=[row for row in maps['rows'] if row['control'] in (250,251,260,270,290,315)]
bind(__file__)
review=dict(verdict='CLEAR',saved_outcome_review_pass=True,passed=True,
    subjects={k:subject(v) for k,v in [('semantic_report',NEW/'direct_target_saved_outcome_review_v1/report.json'),('semantic_request',NEW/'direct_target_saved_outcome_review_v1/request.json'),('map_report',BASE/'report.json'),('map_request',BASE/'request.json'),('map_arrays',BASE/'arrays.npz'),('actual_trace',request['paths']['actual']),('root_physics',request['paths']['root_physics'])]},
    semantic_exact_checks=5691,actual_controls=316,moving_controls=66,nominal_map_checks_exact=66,actual_state_map_evaluations=66,
    first_feature_departure=251,first_learned_clip=270,learned_clipped_commands=46,feedback_map_clip_rows=65,native_map_clip_rows=10,
    selected_same_clock_rows=subset,input_sha256=pins,
    findings=['Exact control250 input still has0.098651rad target error; state departure begins251 before first clipping270.',
        'Direct head consumes no prior-action/history/BFM features; those retained records serve terminal handoff only.',
        'Map target error grows to0.694186rad RMS at315, with joint pose drift0.291491rad RMS.',
        'At315 failed left ankle pitch student target is upper bound+0.5236; fixed committed map requests−0.197648. This directional disagreement is saved arithmetic, not a safe-action certificate.',
        'All65 off-nominal committed-map rows have at least one feedback clip. Their highly clipped local responses are not fresh replanned expert truth.'],
    continuation_assessment='The selected unchanged-objective50k continuation isolates optimization budget; large nominal error and still-falling training loss support that test, but worsening response loss and clipped teacher maps remain limitations.',
    source_reporting_adjustment='Added left ankle pitch index4 qpos11/qvel10 reporting after the initial derivation; source_check binds final e62bc854 and proves original tangent/K/two-clamp expressions unchanged.',
    no_new_graph_or_native_calls=True,model_calls=0,native_steps=0,optimizer_updates=0,new_queries=0,
    behavioral_qualification=False,hardware_authorized=False)
with (BASE/'completion_review.json').open('x') as f:json.dump(review,f,indent=2);f.write('\n')
print(json.dumps({'verdict':'CLEAR','review_sha256':sha(BASE/'completion_review.json'),'pins':len(pins)}))
