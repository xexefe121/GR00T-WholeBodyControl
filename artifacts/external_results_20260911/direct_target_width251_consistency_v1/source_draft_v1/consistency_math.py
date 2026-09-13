"""Bounded saved-array grouping/proximity, never repairs or selects labels."""
import hashlib
import numpy as np
from normalization_math import normalize64

NOMINAL,PHYSICAL,NEW=9904,3054,1018
OLD=NOMINAL+PHYSICAL
TOTAL=OLD+NEW
BLOCK=256
MODES=('raw1323_bits','raw1323_zero_equivalent','current1000_bits','current1000_zero_equivalent',
       'normalized64_bits','normalized64_zero_equivalent')

def view_rows(features,mean,std,mode,indices):
    values=features[indices]
    if mode.startswith('normalized64'):values=normalize64(values,mean,std)
    elif mode.startswith('current1000'):values=values[:,:1000]
    elif not mode.startswith('raw1323'):raise ValueError('Unknown alias mode')
    if mode.endswith('zero_equivalent'):
        values=values.copy();values[values==0]=0.
    return np.ascontiguousarray(values)

def alias_groups(features,targets,normalized_targets,mean,std,mode,digest=hashlib.sha256,old_rows=OLD):
    if mode not in MODES:raise ValueError('Unknown alias mode')
    n=len(features)
    buckets={};groups=[];row_digest=np.empty(n,dtype='S64')
    # One full sequential feature pass per mode. Only colliding representative
    # rows are revisited; a digest match alone never establishes identity.
    for start in range(0,n,BLOCK):
        block=view_rows(features,mean,std,mode,slice(start,min(start+BLOCK,n)))
        for offset,row in enumerate(block):
            index=start+offset;key=digest(row.tobytes()).hexdigest();row_digest[index]=key.encode('ascii')
            matched=buckets.get(key)
            if matched is not None:
                representative=groups[matched][0]
                other=view_rows(features,mean,std,mode,slice(representative,representative+1))[0]
                if row.dtype!=other.dtype or row.shape!=other.shape or row.tobytes()!=other.tobytes():
                    raise ValueError('Digest collision between unequal input rows; no alias admitted')
            if matched is None:
                matched=len(groups);groups.append([]);buckets[key]=matched
            groups[matched].append(index)
    duplicate=[ids for ids in groups if len(ids)>1]
    group_id=np.full(n,-1,np.int32);anchor=np.full(n,-1,np.int64)
    delta=np.zeros((n,23),np.float64);normalized_delta=np.zeros((n,23),np.float64)
    members=[];offsets=[0];mins=[];maxs=[];norm_mins=[];norm_maxs=[];raw_variants=[];current_variants=[]
    for group,ids in enumerate(duplicate):
        indices=np.asarray(ids,np.int64);group_id[indices]=group;anchor[indices]=ids[0]
        delta[indices]=targets[indices]-targets[ids[0]]
        normalized_delta[indices]=normalized_targets[indices].astype(np.float64)-normalized_targets[ids[0]].astype(np.float64)
        members.extend(ids);offsets.append(len(members))
        values=targets[indices];norm=normalized_targets[indices]
        mins.append(values.min(axis=0));maxs.append(values.max(axis=0));norm_mins.append(norm.min(axis=0));norm_maxs.append(norm.max(axis=0))
        raw_variants.append(len({features[i].tobytes() for i in ids}))
        current_variants.append(len({features[i,:1000].tobytes() for i in ids}))
    matrix=lambda rows,dtype:np.asarray(rows,dtype=dtype).reshape(-1,23)
    result=dict(row_digest=row_digest,group_id=group_id,anchor_row=anchor,members=np.asarray(members,np.int64),
        offsets=np.asarray(offsets,np.int64),target_min=matrix(mins,np.float64),target_max=matrix(maxs,np.float64),
        target_delta_from_anchor=delta,target_RMSE_from_anchor=np.sqrt(np.mean(delta*delta,axis=1)),
        target_max_abs_from_anchor=np.max(np.abs(delta),axis=1),normalized_label_delta_from_anchor=normalized_delta,
        normalized_target_min=matrix(norm_mins,np.float32),normalized_target_max=matrix(norm_maxs,np.float32),
        distinct_raw1323_count=np.asarray(raw_variants,np.int64),distinct_current1000_count=np.asarray(current_variants,np.int64))
    conflict=(result['target_min']!=result['target_max']).any(axis=1)
    norm_conflict=(result['normalized_target_min']!=result['normalized_target_max']).any(axis=1)
    mixed=[any(i<old_rows for i in ids) and any(i>=old_rows for i in ids) for ids in duplicate]
    # Every actual member target remains in rows.npz; minima/maxima do not replace it.
    summary=dict(mode=mode,rows=n,duplicate_groups=len(duplicate),duplicate_members=len(members),
        raw_radian_disagreement_groups=int(conflict.sum()),normalized_label_disagreement_groups=int(norm_conflict.sum()),
        new_old_groups=int(sum(mixed)),new_raw_radian_disagreement_groups=int(sum(c and any(i>=old_rows for i in ids) for c,ids in zip(conflict,duplicate))),
        collapsed_distinct_raw1323_groups=int(np.sum(result['distinct_raw1323_count']>1)),
        maximum_component_target_span_rad=float(np.max(result['target_max']-result['target_min'])) if duplicate else 0.,
        canonicalized_signed_zero=mode.endswith('zero_equivalent'),digest_matches_verified_by_bytes=True)
    return result,summary

def fixed_queries(phases,controls,old_rows=OLD):
    new_phases=phases[old_rows:];new_controls=controls[old_rows:]
    if len(new_phases)!=NEW or new_controls.tolist()!=list(range(251,1269)):
        raise ValueError('Exactly1018 chronological new controls251..1268 required')
    if not np.array_equal(new_phases,np.repeat(np.arange(3),[99,819,100])):raise ValueError('Fixed new chronological phases')
    selected=np.concatenate([np.flatnonzero(new_phases==phase)[:24]+old_rows for phase in range(3)])
    if len(selected)!=72:raise ValueError('Exactly72 prespecified proximity queries')
    return selected

def proximity(features,targets,phases,queries,mean,std,old_rows=OLD,block_size=BLOCK):
    if not 1<=block_size<=BLOCK:raise ValueError('Bounded candidate block required')
    if any(int(i)<old_rows or int(i)>=len(features) for i in queries):raise ValueError('Only new rows are queries')
    rows=[]
    for query in queries:
        query=int(query);q=normalize64(features[query:query+1],mean,std)[0]
        # Four fixed views. Strict improvement preserves earliest original row on ties.
        best={name:(float('inf'),None) for name in ('full_global','current_global','full_same_phase','current_same_phase')}
        for start in range(0,old_rows,block_size):
            stop=min(start+block_size,old_rows)
            delta=normalize64(features[start:stop],mean,std)-q
            square=delta*delta
            current=np.sum(square[:,:1000],axis=1,dtype=np.float64)
            prior=np.sum(square[:,1000:1023],axis=1,dtype=np.float64)
            history=np.sum(square[:,1023:],axis=1,dtype=np.float64)
            full=current+prior+history
            if not np.isfinite(full).all():raise ValueError('Nonfinite normalized proximity distance')
            same=phases[start:stop]==phases[query]
            for name,distance,mask in [('full_global',full,None),('current_global',current,None),
                ('full_same_phase',full,same),('current_same_phase',current,same)]:
                candidates=np.arange(stop-start) if mask is None else np.flatnonzero(mask)
                if len(candidates)==0:continue
                offset=int(candidates[np.argmin(distance[candidates])]);value=float(distance[offset])
                if value<best[name][0]:best[name]=(value,start+offset)
        for name,(distance,index) in best.items():
            if index is None:raise ValueError('No eligible original candidate')
            delta=normalize64(features[index:index+1],mean,std)[0]-q
            jump=targets[query]-targets[index]
            rows.append(dict(query_row=query,candidate_row=index,view=name,
                normalized_full_RMS=float(np.sqrt(np.mean(delta*delta))),
                normalized_current_RMS=float(np.sqrt(np.mean(delta[:1000]**2))),
                normalized_prior_RMS=float(np.sqrt(np.mean(delta[1000:1023]**2))),
                normalized_history_RMS=float(np.sqrt(np.mean(delta[1023:]**2))),
                target_RMSE_rad=float(np.sqrt(np.mean(jump*jump))),target_max_abs_rad=float(np.max(np.abs(jump))),
                target_difference_rad=jump.tolist(),distance_squared=distance))
    return rows
