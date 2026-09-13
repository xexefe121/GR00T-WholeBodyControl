"""Saved-array compatibility for three fixed1019-row phase-only datasets."""
import argparse
from pathlib import Path
import json
import numpy as np
from collect_bfm250_labels import BASE,TASK,RUN,read,write,archive,sha,check_qualification

NAMES=('old','query1','query250')
LABEL_PATHS=(TASK/'fast_controller_nominal_pilot_v1/labels/labels.npz',
             TASK/'fresh_expert_labels_resume_v1/labels/labels.npz',BASE/'labels/labels.npz')

def summary(x):
    x=np.asarray(x);x=x[np.isfinite(x)]
    return None if not len(x) else dict(minimum=float(np.min(x)),median=float(np.median(x)),
        p95=float(np.percentile(x,95)),maximum=float(np.max(x)))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--qualification',type=Path,required=True)
    args=parser.parse_args();decision,frozen=check_qualification(args.qualification)
    output=BASE/'compatibility';assert not output.exists(),'Existing compatibility run must remain preserved.'
    report=read(BASE/'labels/report.json')
    assert report['samples']==1019 and report['expert_controls']==[250,1268]
    assert report['collector_frozen_receipt_sha256']==sha(BASE/'collector_frozen_inputs.json')
    assert report['qualification_receipt_sha256']==sha(args.qualification)
    assert report['labels_sha256']==sha(LABEL_PATHS[2])
    all_data=[archive(p) for p in LABEL_PATHS]
    np.testing.assert_array_equal(all_data[0]['control'],np.arange(1269))
    np.testing.assert_array_equal(all_data[1]['control'],np.arange(1,1269))
    np.testing.assert_array_equal(all_data[2]['control'],np.arange(250,1269))
    selected=[];indices=[]
    for data in all_data:
        ids=np.flatnonzero((data['control']>=250)&(data['control']<1269));indices.append(ids)
        np.testing.assert_array_equal(data['control'][ids],np.arange(250,1269))
        np.testing.assert_array_equal(data['source_frame'][ids],np.arange(261,1280))
        selected.append({k:data[k][ids] for k in ('features','expert_target','residual_rad','base_target','control','source_frame')})
    norm=archive(BASE/'labels/existing_normalization.npz')
    original=archive(TASK/'fast_controller_aggregate_fit_v1/fit/teacher_fit.npz')
    for key in ('feature_mean','feature_std'):np.testing.assert_array_equal(norm[key],original[key])
    for data in all_data[1:]:
        np.testing.assert_array_equal(data['joint_span'],all_data[0]['joint_span'])
        np.testing.assert_array_equal(data['joint_limits'],all_data[0]['joint_limits'])
    X=np.concatenate([d['features'] for d in selected])
    target=np.concatenate([d['expert_target'] for d in selected])
    residual=np.concatenate([d['residual_rad'] for d in selected])
    assert X.shape==(3057,1069) and np.isfinite(X).all() and np.isfinite(target).all() and np.isfinite(residual).all()
    groups={}
    canonical=X.copy();canonical[canonical==0]=0.
    for row,values in enumerate(canonical):groups.setdefault(values.tobytes(),[]).append(row)
    duplicate=[]
    for ids in groups.values():
        if len(ids)<2:continue
        target_spread=np.ptp(target[ids],axis=0);residual_spread=np.ptp(residual[ids],axis=0)
        duplicate.append(dict(global_rows=ids,datasets=[NAMES[i//1019] for i in ids],controls=[250+i%1019 for i in ids],
            target_spread_by_joint_rad=target_spread.tolist(),residual_spread_by_joint_rad=residual_spread.tolist(),
            deterministic_target_conflict=bool(np.any(target_spread!=0)),
            deterministic_residual_conflict=bool(np.any(residual_spread!=0))))
    standardized=(X.astype(np.float64)-norm['feature_mean'].astype(np.float64))/norm['feature_std'].astype(np.float64)
    output.mkdir(exist_ok=False);arrays={};reports={};pairs=[];seen=set()
    for a,b in ((0,1),(0,2),(1,2)):
        A=standardized[a*1019:(a+1)*1019];B=standardized[b*1019:(b+1)*1019]
        distance=np.empty((1019,1019),np.float64)
        for row in range(1019):distance[row]=np.sqrt(np.mean((A[row]-B)**2,axis=1,dtype=np.float64))
        assert np.isfinite(distance).all()
        key=NAMES[a]+'__'+NAMES[b];arrays[key+'_distance']=distance
        for source,dest,matrix in ((a,b,distance),(b,a,distance.T)):
            nearest=np.argmin(matrix,axis=1);minimum=matrix[np.arange(1019),nearest]
            global_source=np.arange(1019)+source*1019;global_dest=nearest+dest*1019
            delta_target=target[global_dest]-target[global_source]
            delta_residual=residual[global_dest]-residual[global_source]
            rms=np.sqrt(np.mean(delta_target**2,axis=1));sensitivity=np.divide(rms,minimum,out=np.full(1019,np.nan),where=minimum>0)
            directed=NAMES[source]+'_to_'+NAMES[dest]
            arrays.update({directed+'_nearest_row':nearest,directed+'_feature_rms':minimum,
                directed+'_target_delta':delta_target,directed+'_residual_delta':delta_residual})
            closest=np.argsort(minimum,kind='stable')[:20]
            finite=np.flatnonzero(np.isfinite(sensitivity))
            sensitive=finite[np.argsort(sensitivity[finite],kind='stable')[-20:][::-1]]
            chosen=np.unique(np.r_[0,closest,sensitive])
            reports[directed]=dict(feature_distance=summary(minimum),target_rms_difference_rad=summary(rms),
                closest_twenty_source_rows=closest.tolist(),largest_finite_sensitivity_twenty_source_rows=sensitive.tolist())
            for row in chosen:
                i,j=int(global_source[row]),int(global_dest[row])
                if (i,j) in seen:continue
                seen.add((i,j));pairs.append(dict(selection=directed,a_global_row=i,b_global_row=j,
                    a_dataset=NAMES[source],b_dataset=NAMES[dest],a_control=250+int(row),b_control=250+int(nearest[row]),
                    normalized_feature_rms_distance=float(minimum[row])))
        # Always retain the same-clock query250 comparison as a fixed witness.
        i,j=a*1019,b*1019
        if (i,j) not in seen:
            seen.add((i,j));pairs.append(dict(selection='same_control250',a_global_row=i,b_global_row=j,
                a_dataset=NAMES[a],b_dataset=NAMES[b],a_control=250,b_control=250,
                normalized_feature_rms_distance=float(distance[0,0])))
    np.savez_compressed(output/'distances_and_pairs.npz',**arrays,
        source_label_row_indices=np.stack(indices),feature_mean=norm['feature_mean'],feature_std=norm['feature_std'],
        control=np.tile(np.arange(250,1269),3),dataset=np.repeat(np.arange(3),1019))
    write(output/'fixed_pairs.json',pairs);write(output/'exact_duplicate_groups.json',duplicate)
    conflicts=sum(d['deterministic_target_conflict'] or d['deterministic_residual_conflict'] for d in duplicate)
    result=dict(kind='saved_array_three_phase_dataset_compatibility',samples_each=1019,total_samples=3057,
        controls_each=[250,1268],phase_counts_each=[100,819,100],normalization_refitted=False,
        exact_duplicate_groups=len(duplicate),exact_target_conflict_groups=sum(d['deterministic_target_conflict'] for d in duplicate),
        exact_residual_conflict_groups=sum(d['deterministic_residual_conflict'] for d in duplicate),
        exact_conflict_groups=conflicts,compatibility_pass=conflicts==0,duplicates_averaged_or_removed=False,
        nearest_neighbor=reports,fixed_pairs=len(pairs),distance_dtype='float64 explicit standardized differences',
        distance_matrices_sha256=sha(output/'distances_and_pairs.npz'),fixed_pairs_sha256=sha(output/'fixed_pairs.json'),
        duplicate_groups_sha256=sha(output/'exact_duplicate_groups.json'),
        label_sha256={name:sha(path) for name,path in zip(NAMES,LABEL_PATHS)},
        collector_frozen_receipt_sha256=sha(BASE/'collector_frozen_inputs.json'),
        qualification_receipt_sha256=sha(args.qualification),model_fitting_authorized=False,
        inference_calls=0,optimizer_calls=0,physics_steps=0,hardware_authorized=False)
    write(output/'report.json',result);print(json.dumps(dict(compatibility_pass=conflicts==0,exact_conflict_groups=conflicts,total_samples=3057)))

if __name__=='__main__':main()
