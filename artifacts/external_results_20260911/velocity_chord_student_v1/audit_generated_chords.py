"""Saved-array compatibility only after the one generation completes; no inference."""
from pathlib import Path
import hashlib
import json
import numpy as np

BASE=Path(__file__).resolve().parent;GEN=BASE/'generation'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def summary(x):
    return dict(min=float(np.min(x)),median=float(np.median(x)),p95=float(np.percentile(x,95)),max=float(np.max(x)))

def main():
    output=BASE/'generation_saved_compatibility.json';assert not output.exists()
    report=read(GEN/'report.json');assert report['complete'] and report['probe_rows']==140622
    assert read(GEN/'progress.json')['stage']=='COMPLETE'
    for name,digest in report['output_sha256'].items():assert sha(GEN/name)==digest,name
    with np.load(GEN/'centers.npz',allow_pickle=False) as z:center={k:z[k].copy() for k in z.files}
    x=np.load(GEN/'features.npy',mmap_mode='r');target=np.load(GEN/'teacher_target.npy',mmap_mode='r')
    base=np.load(GEN/'base_target.npy',mmap_mode='r');action=np.load(GEN/'base_action.npy',mmap_mode='r')
    state=np.load(GEN/'state.npy',mmap_mode='r');v=np.load(GEN/'perturbed_joint_velocity.npy',mmap_mode='r')
    assert x.shape==(3057,23,2,1069) and x.dtype==np.float32
    assert target.shape==base.shape==(3057,23,2,23) and target.dtype==base.dtype==np.float64
    for dataset in range(3):
        ids=center['dataset']==dataset
        np.testing.assert_array_equal(center['control'][ids],np.arange(250,1269))
        np.testing.assert_array_equal(center['source_frame'][ids],np.arange(261,1280))
    lo,hi=center['joint_limits'].T;caps=center['native_velocity'];span=center['joint_span']
    c=read(Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1/contract.json'))
    for j in range(23):
        mask=np.ones(1069,bool);mask[23+j]=False;mask[1023:1046]=False
        # Select the axis first, then the feature mask, preserving row/sign order.
        actual=x[:,j,:,:][:,:,mask]
        expected=np.broadcast_to(center['features'][:,None,mask],actual.shape)
        assert np.array_equal(actual.view(np.uint32),expected.copy().view(np.uint32))
        for s,sign in enumerate((-1.,1.)):
            expected_v=center['qvel'][:,6+j]+sign*.01*caps[j]
            assert np.array_equal(v[:,j,s].view(np.uint64),expected_v.view(np.uint64))
            assert np.all(np.abs(expected_v)<caps[j])
            assert np.array_equal(x[:,j,s,23+j].view(np.uint32),expected_v.astype(np.float32).view(np.uint32))
    for a in (x,target,base,action,state,v):assert np.isfinite(a).all()
    assert np.all(target>=lo) and np.all(target<=hi)
    reconstructed=np.asarray(c['default_q'])+action*.25*np.asarray(c['training_effort'])/np.asarray(c['kp'])
    assert np.array_equal(reconstructed.view(np.uint64),base.view(np.uint64))
    expected_base_features=(base-np.asarray(c['default_q'])).astype(np.float32)
    assert np.array_equal(expected_base_features.view(np.uint32),x[:,:,:,1023:1046].copy().view(np.uint32))
    change=target-center['expert_target'][:,None,None,:]
    assert np.max(np.abs(change))<=.2+1e-12
    feature_sets=[center['features'],x.reshape(-1,1069)]
    target_sets=[center['expert_target'],target.reshape(-1,23)]
    base_sets=[center['base_target'],base.reshape(-1,23)]
    hashes={};duplicates=[]
    for set_index,features in enumerate(feature_sets):
        for row,feature in enumerate(features):
            digest=hashlib.sha256(feature.tobytes()).hexdigest()
            if digest in hashes:
                first_set,first_row=hashes[digest]
                assert feature.tobytes()==feature_sets[first_set][first_row].tobytes()
                tdelta=target_sets[set_index][row]-target_sets[first_set][first_row]
                rdelta=(target_sets[set_index][row]-base_sets[set_index][row])-(target_sets[first_set][first_row]-base_sets[first_set][first_row])
                duplicates.append(dict(first_set=first_set,first_row=first_row,set=set_index,row=row,
                    target_conflict=bool(np.any(tdelta!=0)),residual_conflict=bool(np.any(rdelta!=0)),
                    target_max_abs_delta=float(np.max(np.abs(tdelta))),residual_max_abs_delta=float(np.max(np.abs(rdelta)))))
            else:hashes[digest]=(set_index,row)
    cells=[]
    feedback=np.load(GEN/'teacher_feedback_clipped.npy',mmap_mode='r')
    native=np.load(GEN/'teacher_native_clipped.npy',mmap_mode='r')
    branch=np.load(GEN/'teacher_branch_changed.npy',mmap_mode='r')
    for d in range(3):
        for lower,upper in ((250,350),(350,1169),(1169,1269)):
            ids=np.flatnonzero((center['dataset']==d)&(center['control']>=lower)&(center['control']<upper))
            cells.append(dict(dataset=d,control_start=lower,control_stop=upper,centers=len(ids),probes=len(ids)*46,
                feedback_clipped_probes=int(np.any(feedback[ids],axis=-1).sum()),native_clipped_probes=int(np.any(native[ids],axis=-1).sum()),
                branch_changed_probes=int(np.any(branch[ids],axis=-1).sum()),
                teacher_change_rms_rad=summary(np.sqrt(np.mean(change[ids]**2,axis=-1))),
                base_change_rms_rad=summary(np.sqrt(np.mean((base[ids]-center['base_target'][ids,None,None,:])**2,axis=-1)))))
    result=dict(kind='saved_velocity_probe_compatibility',nominal_rows=3057,probe_rows=140622,total_rows=143679,
        all_output_hashes_exact=True,all_probe_feature_masks_byteexact=True,all_static_native_speed_bounds_pass=True,
        base_from_raw_action_byteexact=True,base_feature_encoding_byteexact=True,all_targets_finite_and_native_bounded=True,
        max_abs_teacher_center_difference_rad=float(np.max(np.abs(change))),
        exact_duplicate_rows=len(duplicates),exact_target_conflicts=sum(d['target_conflict'] for d in duplicates),
        exact_residual_conflicts=sum(d['residual_conflict'] for d in duplicates),duplicates=duplicates,cells=cells,
        no_rows_removed_or_reweighted=True,generation_report_sha256=sha(GEN/'report.json'),
        center_sha256=sha(GEN/'centers.npz'),audit_source_sha256=sha(__file__),model_inference_calls=0,physics_steps=0,optimizer_updates=0)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cells','duplicates')},indent=2))

if __name__=='__main__':main()
