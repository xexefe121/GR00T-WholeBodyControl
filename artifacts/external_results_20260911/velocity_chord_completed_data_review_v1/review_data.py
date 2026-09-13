"""Pure signed-zero-canonicalized uniqueness and completed-evidence binding review."""
from pathlib import Path
import hashlib,json,traceback
import numpy as np
BASE=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911')
OUT=Path(__file__).parent
EXP=BASE/'velocity_chord_student_v1'
GEN=EXP/'generation'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,data):p.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
paths={'generation_report':GEN/'report.json','generation_progress':GEN/'progress.json',
       'generation_status':EXP/'generation_process_status.json','compatibility':EXP/'generation_saved_compatibility.json',
       'compatibility_source':EXP/'audit_generated_chords.py','root_report':BASE/'velocity_chords_independent_v1/report.json',
       'root_request':BASE/'velocity_chords_independent_v1/request.json','root_source':BASE/'velocity_chords_independent_v1/source.py',
       'frozen_inputs':EXP/'generation_frozen_inputs_v2.json','clearance':EXP/'generation_clearance.json'}
pins={str(p):sha(p) for p in paths.values()}
request=dict(kind='completed_velocity_chord_data_review',source_sha256=sha(__file__),inputs=pins,
             additional_model_calls=0,physics_steps=0,optimizer_updates=0,labels_mutated=False,
             check='Canonicalize signed zero in temporary row copies and check all143679 numeric feature rows for duplicates/conflicting labels.')
assert not (OUT/'request.json').exists()
write(OUT/'request.json',request)
checked=0
try:
    producer=read(paths['generation_report']);root=read(paths['root_report']);compat=read(paths['compatibility'])
    assert sha(paths['generation_report'])=='3c611a67c26876c5794dce340dd919fec4e5a43c0e160d16481b9678593613a9'
    assert sha(paths['root_report'])=='2cc2346257ccedbc09ac406f5e4fcd8dc1b68851198c76646de28aebc4d0fc70'
    assert sha(paths['compatibility'])=='574a9ceb10a23859012c5184aee22c2599dedb245f23ef273ec1003ec64b0879'
    assert root['passed'] and root['centers']==3057 and root['probes']==140622
    assert root['source_sha256']==sha(paths['root_source'])=='ac2d28117a6a44881bf6271107711a523b8c989cfecad1a583262329c09f11a3'
    assert root['request_sha256']==sha(paths['root_request'])
    assert root['generation_report_sha256']==compat['generation_report_sha256']==sha(paths['generation_report'])
    assert root['all_nominal_labels_and_committed_plan_bindings_byteexact'] and root['all_state_feature_target_and_branch_arrays_byteexact']
    assert root['fixed_BFM_repeats_byteexact'] and root['audit_inference_calls']=={'actor':423,'backward':9}
    assert producer['complete'] and producer['inference_calls']=={'actor':143679,'backward':3057}
    assert producer['input_frozen_receipt_sha256']==sha(paths['frozen_inputs'])
    assert producer['clearance_sha256']==sha(paths['clearance'])
    assert read(paths['generation_progress'])['stage']=='COMPLETE'
    status=read(paths['generation_status'])
    assert status['raw_python_exit_code']==status['process_exit_code']==0 and status['launcher_error'] is None
    assert compat['audit_source_sha256']==sha(paths['compatibility_source'])
    assert compat['no_rows_removed_or_reweighted']
    assert root['maximum_teacher_center_delta_rad']==compat['max_abs_teacher_center_difference_rad']==0.14534659138196787
    for key,value in root['clipping_counts'].items():assert producer[key]==value
    for name,digest in producer['output_sha256'].items():assert sha(GEN/name)==digest,name
    with np.load(GEN/'centers.npz',allow_pickle=False) as z:
        cx=z['features'].copy();ct=z['expert_target'].copy();cb=z['base_target'].copy()
    px=np.load(GEN/'features.npy',mmap_mode='r').reshape(-1,1069)
    pt=np.load(GEN/'teacher_target.npy',mmap_mode='r').reshape(-1,23)
    pb=np.load(GEN/'base_target.npy',mmap_mode='r').reshape(-1,23)
    assert cx.shape==(3057,1069) and px.shape==(140622,1069) and cx.dtype==px.dtype==np.float32
    banks=((cx,ct,cb),(px,pt,pb));seen={};duplicates=[];changed_zero_rows=0
    for bank,(features,targets,bases) in enumerate(banks):
        for row,feature in enumerate(features):
            canonical=feature.copy();canonical[canonical==0]=np.float32(0)
            changed_zero_rows+=int(canonical.tobytes()!=feature.tobytes())
            digest=hashlib.sha256(canonical.tobytes()).digest()
            if digest in seen:
                old_bank,old_row=seen[digest];of,ot,ob=banks[old_bank]
                expected=of[old_row].copy();expected[expected==0]=np.float32(0)
                assert canonical.tobytes()==expected.tobytes(),'Hash collision'
                duplicates.append(dict(bank=bank,row=row,first_bank=old_bank,first_row=old_row,
                    target_conflict=bool(np.any(targets[row]!=ot[old_row])),
                    residual_conflict=bool(np.any((targets[row]-bases[row])!=(ot[old_row]-ob[old_row])))))
            else:seen[digest]=(bank,row)
            checked+=1
    assert checked==143679
    assert not duplicates,duplicates[:10]
    for p,digest in pins.items():assert sha(p)==digest,p
    for name,digest in producer['output_sha256'].items():assert sha(GEN/name)==digest,name
    assert sha(__file__)==request['source_sha256']
    result=dict(kind='completed_velocity_chord_data_review',verdict='CLEAR',approved=True,passed=True,
        scope='Dataset prerequisites for the ONE previously selected fixed5000 fit only; frozen trainer/source clearance remains required.',
        nominal_centers=3057,probes=140622,total_feature_rows=checked,unique_numeric_feature_rows=len(seen),
        rows_with_negative_zero_canonicalized=changed_zero_rows,exact_numeric_duplicate_rows=0,
        exact_target_conflicts=0,exact_residual_conflicts=0,
        full_teacher_state_feature_branch_independent_audit_passed=True,
        independent_actor_repeats=423,independent_backward_repeats=9,
        generation_actor_calls=143679,generation_backward_calls=3057,
        maximum_teacher_center_delta_rad=root['maximum_teacher_center_delta_rad'],clipping_counts=root['clipping_counts'],
        output_hashes=producer['output_sha256'],evidence_hashes=pins,all_evidence_and_outputs_unchanged=True,
        request_sha256=sha(OUT/'request.json'),source_sha256=sha(__file__),
        reviewer_additional_inference_calls=0,reviewer_physics_steps=0,reviewer_optimizer_updates=0,labels_mutated=False,
        limitations=['Perturbed labels describe the saved clipped committed map, not fresh replanning or qualified perturbed dynamics.',
                     'No nominal or difficult probe row is removed or reweighted. No broader PICO/walk002 data is added to the selected fit.'])
    write(OUT/'review.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('evidence_hashes','output_hashes')},indent=2))
except BaseException as error:
    write(OUT/'failure.json',dict(passed=False,checked_rows=checked,error=repr(error),traceback=traceback.format_exc()))
    raise
