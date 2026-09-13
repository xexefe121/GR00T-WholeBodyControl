"""Unchanged saved map math; all actual learned controls, no planning or native calls."""
from types import SimpleNamespace
from saved_common import *
GROUPS=(("root_position",0,3),("root_rotation",3,6),("joint_position",6,29),("root_linear_velocity",29,32),("root_angular_velocity",32,35),("joint_velocity",35,58))

def difference_function(core):
    tree=ast.parse(core.read_text())
    body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('quat_mul','quat_log')]
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Planner')
    body.append(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='difference'))
    namespace={'np':np}
    exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),str(core),'exec'),namespace)
    holder=SimpleNamespace(nq=30,nv=29,lin_q=np.r_[0:3,7:30],lin_v=np.r_[0:3,6:29],free=[(0,0)])
    return lambda x,y:namespace['difference'](holder,x[None],y[None])[0]

def committed_target(difference,plan_state,plan_target,gain,qpos,qvel,limits):
    # Preserve the reviewed saved_committed_map full product and both nested clips.
    actual=np.r_[qpos,qvel]
    tangent=difference(plan_state,actual)
    raw=gain@tangent
    correction=np.clip(raw,-.1,.1)
    preclip=plan_target+correction
    target=np.clip(preclip,limits[:,0],limits[:,1])
    return target,raw,correction,preclip

def grouped(gain,tangent):
    assert gain.shape==(23,58) and tangent.shape==(58,)
    parts=np.stack([gain[:,start:end]@tangent[start:end] for _,start,end in GROUPS])
    full=gain@tangent
    error=float(np.max(np.abs(parts.sum(axis=0)-full)))
    # Grouped products change accumulation order; they never produce the command.
    scale=max(1.,float(np.max(np.sum(np.abs(gain*tangent[None,:]),axis=1))))
    assert error<=64*np.finfo(np.float64).eps*scale
    other=gain[:,:35]@tangent[:35]
    velocity=gain[:,35:]@tangent[35:]
    return parts,other,velocity,error

def run_maps(actual,selected,centers,contract,core,original,output,check):
    # Verify unchanged full-product and clipping expression trees before any calculation.
    functions=lambda path:{n.name:ast.dump(n,include_attributes=False) for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef)}
    ours,old=functions(__file__),functions(original)
    for name in ('difference_function','committed_target','grouped'):check.require(ours[name]==old[name],'unchanged fixed-map function '+name)
    diff=difference_function(core);limits=centers['joint_limits'];caps=np.asarray(contract['native_velocity'])
    rows=[];arrays={};rms=lambda v:float(np.sqrt(np.mean(np.asarray(v,np.float64)**2)))
    for control in selected:
        control=int(control);matches=np.flatnonzero((centers['dataset']==2)&(centers['control']==control))
        check.require(len(matches)==1,'unique query250 center '+str(control));idx=int(matches[0])
        check.require(int(centers['plan_control'][idx])+int(centers['plan_local'][idx])==control,'matching committed clock '+str(control))
        plan,u,gain=[centers[name][idx] for name in ('planned_state','planned_target','gain')]
        q,dq=actual['qpos'][control],actual['qvel'][control]
        q0,dq0=centers['qpos'][idx],centers['qvel'][idx]
        nominal,nominal_raw,_,_=committed_target(diff,plan,u,gain,q0,dq0,limits)
        check.exact(nominal,centers['expert_target'][idx],'exact nominal map '+str(control))
        target,raw,correction,preclip=committed_target(diff,plan,u,gain,q,dq,limits)
        tangent=diff(plan,np.r_[q,dq]);nominal_tangent=diff(plan,np.r_[q0,dq0]);change=tangent-nominal_tangent
        departure=diff(np.r_[q0,dq0],np.r_[q,dq])
        parts,other,velocity,error=grouped(gain,change)
        absolute,abs_other,abs_velocity,abs_error=grouped(gain,tangent)
        scale=max(1.,float(np.max(np.abs(raw))),float(np.max(np.abs(nominal_raw))),float(np.max(np.abs(gain@change))))
        check.require(np.max(np.abs(gain@change-(raw-nominal_raw)))<=128*np.finfo(np.float64).eps*scale,'linear change roundoff '+str(control))
        row=dict(control=control,center_index=idx,plan_control=int(centers['plan_control'][idx]),plan_local=int(centers['plan_local'][idx]),
            head_vs_fixed_map_rmse_rad=rms(actual['target'][control]-target),head_vs_nominal_rmse_rad=rms(actual['target'][control]-nominal),
            map_vs_nominal_rmse_rad=rms(target-nominal),head_clipped=bool(np.any(actual['raw_proposal'][control]!=actual['target'][control])),
            fixed_feedback_clip_joints=[contract['joint_names'][j] for j in np.flatnonzero(raw!=correction)],
            fixed_native_clip_joints=[contract['joint_names'][j] for j in np.flatnonzero(preclip!=target)],
            joint_velocity_native_cap_normalized_rms=rms(departure[35:]/caps),joint_velocity_native_cap_normalized_max=float(np.max(np.abs(departure[35:]/caps))),
            physical_departure_group_rms={name:rms(departure[start:end]) for name,start,end in GROUPS},
            raw_change_group_rms_rad={name:rms(parts[j]) for j,(name,_,_) in enumerate(GROUPS)},
            raw_actual_group_rms_rad={name:rms(absolute[j]) for j,(name,_,_) in enumerate(GROUPS)},
            raw_change_other35_rms_rad=rms(other),raw_change_velocity23_rms_rad=rms(velocity),raw_change_other35_larger=bool(rms(other)>rms(velocity)),
            grouped_accumulation_max_error_rad=max(error,abs_error))
        rows.append(row)
        values=dict(control=control,center_index=idx,map_target=target,nominal_target=nominal,head_target=actual['target'][control],gain=gain,
            planned_state=plan,planned_target=u,actual_tangent=tangent,nominal_tangent=nominal_tangent,plan_chart_tangent_change=change,physical_departure=departure,
            feedback_raw=raw,nominal_feedback_raw=nominal_raw,feedback_correction=correction,preclip=preclip,raw_change_groups=parts,raw_actual_groups=absolute,
            raw_change_other35=other,raw_change_velocity23=velocity)
        for name,value in values.items():arrays.setdefault(name,[]).append(value)
    np.savez_compressed(output/'fixed_map_arrays.npz',**arrays)
    first_clip=next((row['control'] for row in rows if row['head_clipped']),None)
    groups={'all_actual_learned':rows,'departed':rows[1:],'departed_before_first_clip':[r for r in rows[1:] if first_clip is None or r['control']<first_clip]}
    summary={name:dict(rows=len(group),other35_larger_rows=sum(row['raw_change_other35_larger'] for row in group),
        raw_change_group_rms_rad={g:rms([row['raw_change_group_rms_rad'][g] for row in group]) if group else None for g,_,_ in GROUPS}) for name,group in groups.items()}
    report=dict(passed=True,nominal_map_checks_exact=len(rows),actual_map_evaluations=len(rows),rows=rows,summary=summary,
        arrays_sha256=sha(output/'fixed_map_arrays.npz'),model_calls=0,native_steps=0,replans=0,
        limitations=['Stale same-clock committed maps are not replanned expert truth or feasibility certificates.',
            'Group products decompose preclip linear feedback; their norms are not additive percentages. Both nonlinear clips remain unchanged.',
            'The new full-state dataset covers all58 axes locally; this audit does not establish behavior beyond those neighborhoods.'])
    write(output/'fixed_map_report.json',report)
    return dict(report_sha256=sha(output/'fixed_map_report.json'),nominal_map_checks_exact=len(rows),actual_map_evaluations=len(rows),summary=summary)
