"""Independent saved full58 math; no trainer, model, optimizer or native imports."""
import math
import numpy as np
from audit_math import phase_indices,position_errors,targets

GROUP_SIZES=(3,3,23,3,3,23)
AXES=tuple(np.arange(sum(GROUP_SIZES[:g]),sum(GROUP_SIZES[:g+1]),dtype=np.int64) for g in range(6))
BACKENDS=('initial_GPU32','final_GPU32','CPU64','GPU64','ORT64')
CORPORA=('nominal','full_state','physical')
SIZES=(9904,354612,3054)

def schedule(updates=10000):
    if not 1<=updates<=10000:raise ValueError('Schedule size')
    generator=np.random.Generator(np.random.PCG64(20260911))
    before=generator.bit_generator.state
    centers=np.empty((updates,864),np.int32);axes=np.empty_like(centers,dtype=np.int8)
    # Literal contract: 16 center draws, then 16 axis draws, each cell/group.
    for update in range(updates):
        cursor=0
        for eligible in phase_indices()[:9]:
            for group_axes in AXES:
                a=generator.integers(0,len(eligible),16,dtype=np.int32)
                b=generator.integers(0,len(group_axes),16,dtype=np.int32)
                centers[update,cursor:cursor+16]=eligible[a]
                axes[update,cursor:cursor+16]=group_axes[b]
                cursor+=16
    return centers,axes,before,generator.bit_generator.state

def probe_indices(centers,axes):
    centers=np.asarray(centers);axes=np.asarray(axes)
    if centers.shape!=axes.shape or centers.ndim!=1:raise ValueError('Pair shapes')
    if not np.issubdtype(centers.dtype,np.integer) or not np.issubdtype(axes.dtype,np.integer):raise ValueError('Integer pair metadata')
    if np.any((centers<0)|(centers>=3057)) or np.any((axes<0)|(axes>=58)):raise ValueError('Pair range')
    starts=116*centers.astype(np.int64)+2*axes.astype(np.int64)
    return np.column_stack((starts,starts+1)).ravel()

def rate(index):
    if not 0<=index<10000:raise ValueError('Rate index')
    return 1e-5+.5*(1e-4-1e-5)*(1+math.cos(math.pi*index/9999))

def gradient_geometry(arrays):
    names=('nominal','full_state','physical')
    shapes=((256,1000),(256,),(256,256),(256,),(23,256),(23,))
    assert set(arrays)=={name+'_'+str(i) for name in names for i in range(6)}
    vectors={}
    for name in names:
        pieces=[]
        for i,shape in enumerate(shapes):
            value=arrays[name+'_'+str(i)]
            assert value.shape==shape and value.dtype==np.float32 and np.isfinite(value).all()
            pieces.append(value.astype(np.float64).ravel())
        vectors[name]=np.concatenate(pieces)
    norms={name:float(np.linalg.norm(vectors[name])) for name in names}
    assert np.isfinite(list(norms.values())).all() and norms['nominal']>0 and norms['full_state']>0
    weight=norms['nominal']/norms['full_state'];assert np.isfinite(weight) and weight>0
    gram=[[float(np.dot(vectors[a],vectors[b])) for b in names] for a in names]
    cosine=[[gram[i][j]/(norms[a]*norms[b]) if norms[a]*norms[b]>0 else None for j,b in enumerate(names)] for i,a in enumerate(names)]
    joint=vectors['nominal']+weight*vectors['full_state']+vectors['physical']
    return dict(coefficient=weight,names=list(names),norms=norms,gram_matrix=gram,cosine_matrix=cosine,
        first_order_descent_dots={name:float(np.dot(vectors[name],joint)) for name in names},total_gradient_norm=float(np.linalg.norm(joint)))

def initial_losses(predictions,data,centers,axes):
    n,f,p=[predictions[name] for name in CORPORA]
    assert n.shape==(9904,23) and f.shape==(1728,23) and p.shape==(3054,23)
    assert all(a.dtype==np.float32 and np.isfinite(a).all() for a in (n,f,p))
    span=data['span'].astype(np.float64)
    labels=((data['nominal_teacher']-data['default'])/span).astype(np.float32)
    nsq=np.square(n-labels).astype(np.float64)
    nc=np.array([nsq[group].mean() for group in phase_indices()])
    wanted=(data['full_state_teacher'].reshape(3057,58,2,23)[centers,axes]-data['nominal_teacher'][centers,None,:])/span
    actual=f.astype(np.float64).reshape(864,2,23)-n[centers].astype(np.float64)[:,None,:]
    fsq=np.square(actual-wanted)
    fc=np.array([fsq[i:i+16].mean() for i in range(0,864,16)])
    successor=data['physical_successor']
    pexpect=(data['physical_teacher']-data['nominal_teacher'][successor])/span
    psq=np.square(p.astype(np.float64)-n[successor].astype(np.float64)-pexpect)
    pc=np.array([psq[group].sum()/(len(group)*23) for group in phase_indices(((99,819,100),)*3)])
    return dict(nominal=float(nc.mean()),full_state=float(fc.mean()),physical=float(pc.mean())),dict(nominal=nc,full_state=fc,physical=pc)

def metrics(predictions,data):
    n,f,p=[predictions[name] for name in CORPORA]
    default,span,limits=data['default'],data['span'],data['limits'];wide=span.astype(np.float64)
    teacher=data['nominal_teacher'];labels=((teacher-default)/wide).astype(np.float32)
    nsq=np.square(n-labels).astype(np.float64)
    n_cells=[];f_cells=[];p_cells=[]
    def errors(values,truth):return position_errors(values,truth,default,span,limits)
    for i,group in enumerate(phase_indices()):
        n_cells.append(dict(dataset=i//3,phase=i%3,rows=len(group),normalized_MSE=float(nsq[group].mean()),
            first24=errors(n[group[:24]],teacher[group[:24]]),**errors(n[group],teacher[group])))
    fteacher=data['full_state_teacher'].reshape(3057,58,2,23)
    ferr=f.astype(np.float64).reshape(3057,58,2,23)-n[:3057].astype(np.float64)[:,None,None,:]-(fteacher-teacher[:3057,None,None,:])/wide
    for i,group in enumerate(phase_indices()[:9]):
        for g,axes in enumerate(AXES):
            pairs_c=np.repeat(group,len(axes));pairs_a=np.tile(axes,len(group));rows=probe_indices(pairs_c,pairs_a)
            selected=ferr[group][:,axes]
            f_cells.append(dict(dataset=i//3,phase=i%3,tangent_group=g,center_rows=len(group),axes=len(axes),endpoint_rows=len(rows),
                response_MSE=float(np.mean(selected**2)),first24_response_MSE=float(np.mean(ferr[group[:24]][:,axes]**2)),
                **errors(f[rows],data['full_state_teacher'][rows])))
    successor=data['physical_successor']
    perr=p.astype(np.float64)-n[successor].astype(np.float64)-(data['physical_teacher']-teacher[successor])/wide
    for i,group in enumerate(phase_indices(((99,819,100),)*3)):
        p_cells.append(dict(dataset=i//3,phase=i%3,requested=len(group),valid=len(group),
            response_MSE=float(np.square(perr[group]).sum()/(len(group)*23)),first24_response_MSE=float(np.square(perr[group[:24]]).mean()),
            **errors(p[group],data['physical_teacher'][group])))
    flag_report={}
    for name,flag in data['full_state_flags'].items():
        mask=np.any(flag,axis=-1)
        flag_report[name]=dict(endpoint_rows=int(mask.sum()),raw_row_weighted_response_MSE=float(np.mean(ferr[mask]**2)) if mask.any() else None)
    return dict(nominal_objective=float(np.mean([c['normalized_MSE'] for c in n_cells])),
        full_state_objective=float(np.mean([c['response_MSE'] for c in f_cells])),physical_objective=float(np.mean([c['response_MSE'] for c in p_cells])),
        nominal_cells=n_cells,full_state_cells=f_cells,physical_cells=p_cells,full_state_flag_diagnostics=flag_report,
        full_state_radius_division=False,no_connected_stability_claim=True)

def numerical_comparisons(outputs,data):
    result={}
    for corpus in CORPORA:
        raw={name:targets(outputs[name][corpus],data['default'],data['span'],data['limits'])[0] for name in ('CPU64','GPU64','ORT64')}
        for a,b in (('CPU64','GPU64'),('CPU64','ORT64'),('GPU64','ORT64')):
            result[corpus+'_'+a+'_'+b]=float(np.max(np.abs(raw[a]-raw[b])))
    assert np.isfinite(list(result.values())).all()
    return result

def drift_summary(old,new,data):
    before,applied_before,_=targets(old,data['default'],data['span'],data['limits'])
    after,applied_after,_=targets(new,data['default'],data['span'],data['limits'])
    delta=after-before;oldclip=before!=applied_before;newclip=after!=applied_after;changed=oldclip!=newclip
    return delta,dict(shape=list(delta.shape),max_abs_preclamp_rad=float(np.max(np.abs(delta))),RMS_preclamp_rad=float(np.sqrt(np.mean(delta**2))),
        old_clipped_rows=int(oldclip.any(axis=1).sum()),old_clipped_components=int(oldclip.sum()),new_clipped_rows=int(newclip.any(axis=1).sum()),
        new_clipped_components=int(newclip.sum()),clipping_changed_rows=int(changed.any(axis=1).sum()),clipping_changed_components=int(changed.sum()))
