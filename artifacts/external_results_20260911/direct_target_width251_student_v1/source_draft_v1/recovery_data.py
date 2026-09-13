"""Load qualified saved recovery rows only after the enclosing concrete fit gate."""
from pathlib import Path
import numpy as np
from direct_data import read,sha
from input_schema import load_numeric
from context_contract import exact
from response_data import load_response_data
from recovery_objective import phase_cells
from recovery_protocol import protocol

RECOVERY_ROLES=('recovery_rows','collection_report','collection_request','collection_qualification','collection_source_review','consistency_report','warm_restore_review')

def literal_member(mapping,subject):
    def key(path):
        v=str(path).replace('\\','/')
        if v.startswith('/mnt/') and v[6:7]=='/':v=v[5]+':'+v[6:]
        return v.casefold()
    normalized={}
    for path,digest in mapping.items():
        k=key(path)
        if k in normalized and normalized[k]!=digest:raise ValueError('Conflicting subject path aliases')
        normalized[k]=digest
    if normalized.get(key(subject['path']))!=subject['sha256']:raise ValueError('Missing actual consumed recovery subject')

def check_recovery_subjects(request,pins):
    subjects=request['subjects']
    for role in RECOVERY_ROLES:
        item=subjects[role];literal_member(pins,item)
        if sha(item['path'])!=item['sha256']:raise ValueError('Changed qualified recovery input')
    collection=read(subjects['collection_report']['path']);selected=read(subjects['collection_request']['path'])
    if (collection['passed'] is not True or collection['collection_completed'] is not True or collection['model_fitting_authorized'] is not False
        or (collection['rows'],collection['control_start'],collection['control_stop_exclusive'],collection['fresh_student_state_queries'])!=(1018,251,1269,1)):
        raise ValueError('Complete qualified1018 connected recovery collection required')
    if collection['request_sha256']!=subjects['collection_request']['sha256'] or collection['outputs']['expert_rows.npz']!=subjects['recovery_rows']['sha256']:
        raise ValueError('Actual collection output/request identity')
    if Path(subjects['recovery_rows']['path']).resolve()!=(Path(subjects['collection_report']['path']).parent/'expert_rows.npz').resolve():raise ValueError('Actual collection rows path')
    if collection['outputs']['normalization.npz']!=subjects['normalization']['sha256']:raise ValueError('No normalization refit allowed')
    if selected['root_selected_collection'] is not True or selected['model_fitting_authorized'] is not False:raise ValueError('Original collection-only selection')
    for role,original in [('collection_qualification','qualification'),('collection_source_review','source_review')]:
        actual=selected['subjects'][original]
        literal_member({actual['path']:actual['sha256']},subjects[role])
        literal_member(collection['input_sha256'],subjects[role])
    qualification=read(subjects['collection_qualification']['path']);source=read(subjects['collection_source_review']['path'])
    if qualification['root_authorized_extraction'] is not True or qualification['model_fitting_authorized'] is not False:raise ValueError('Original full recovery collection qualification')
    if (qualification['control_start'],qualification['control_stop_exclusive'],qualification['rows'])!=(251,1269,1018):raise ValueError('Exact original qualified recovery scope')
    if source['source_review_pass'] is not True or source['source_sha256']!=selected['source_sha256'] or source['source_sha256']!=collection['source_sha256']:raise ValueError('Exact qualified collector source')
    consistency=read(subjects['consistency_report']['path'])
    if consistency['passed'] is not True or consistency['evidence_diagnosis_completed'] is not True or (consistency['old_rows'],consistency['new_rows'],consistency['rows'])!=(12958,1018,13976):
        raise ValueError('Completed exact saved consistency evidence required')
    for role in ('recovery_rows','collection_report','collection_request','collection_qualification','collection_source_review','normalization'):
        literal_member(consistency['input_sha256'],subjects[role])
    warm=read(subjects['warm_restore_review']['path'])
    if warm['source_review_pass'] is not True:raise ValueError('Warm512 source review required')
    for name in ('warm512_restore.py','width512.py','restoration_support.py','balance_contract.py'):
        if warm['source_sha256'][name]!=sha(Path(__file__).parent/name):raise ValueError('Exact reviewed warm512 source differs')
    return dict(collection_report_sha256=subjects['collection_report']['sha256'],consistency_report_sha256=subjects['consistency_report']['sha256'],
        conflicts_not_automatically_repaired=True,consistency_reviewed_by_selected_fit=True,fresh_student_state_queries=1,connected_expert_rows=1018)

def attach_recovery(data,request,pins):
    evidence=check_recovery_subjects(request,pins)
    specs={'causal_features':((1018,1323),'float32'),'features':((1018,1000),'float32'),'context':((1018,323),'float32'),
        'incoming_prior':((1018,23),'float32'),'incoming_history':((1018,300),'float32'),'expert_target':((1018,23),'float64'),
        'control':((1018,),'int64'),'source_frame':((1018,),'int64'),'phase':((1018,),'int8'),
        'first_student_state_query':((1018,),'bool'),'plan_control':((1018,),'int64'),'plan_local':((1018,),'int64'),
        'feedback_clipped':((1018,23),'bool'),'native_clipped':((1018,23),'bool')}
    rows,_=load_numeric(request['subjects']['recovery_rows']['path'],specs)
    cells=phase_cells(rows['phase'],rows['control']);control=rows['control']
    exact(rows['source_frame'],control+11,'new global source frames')
    exact(rows['causal_features'][:,:1000],rows['features'],'new current features')
    exact(rows['causal_features'][:,1000:],rows['context'],'new causal suffix')
    exact(rows['context'][:,:23],rows['incoming_prior'],'new actual prior')
    exact(rows['context'][:,23:],rows['incoming_history'],'new incoming history')
    exact(rows['first_student_state_query'],np.r_[True,np.zeros(1017,bool)],'one fresh student state')
    exact(rows['plan_control'],251+5*((control-251)//5),'committed plan control')
    exact(rows['plan_local'],(control-251)%5,'committed local step')
    target=rows['expert_target'];limits=data['limits']
    if np.any(target<limits[:,0]) or np.any(target>limits[:,1]):raise ValueError('New qualified target outside native limits')
    data.update(recovery_features=rows['causal_features'],recovery_target=target,recovery_cells=cells,recovery_metadata=rows,recovery_evidence=evidence)
    return data

def load_data(request,pins):
    # Original old-data/context/full10000 verification remains byte-identical.
    data,normalization,centers,axes=load_response_data(request,pins)
    p=protocol(request)
    if centers.shape!=(10000,864) or axes.shape!=centers.shape:raise ValueError('Existing complete10000 schedule required')
    data['schedule_lineage']['selected_prefix_updates']=p.updates
    data['schedule_lineage']['wrap_or_regeneration']=False
    return attach_recovery(data,request,pins),normalization,centers[:p.updates],axes[:p.updates]
