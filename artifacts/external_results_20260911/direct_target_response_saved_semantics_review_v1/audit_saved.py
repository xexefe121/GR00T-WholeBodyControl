"""One saved semantic audit; independent native replay is consumed, never repeated."""
from saved_common import *
from release_checks import check_release
from fixed_maps import run_maps
from context_math import context, features, nominal_features, map_selection

def segment(a,r,start,count,previous,history,goals,terms_fn,contract,span,check,cap=None):
    n=len(a['target']);steps=len(a['physics_torque']);pre=len(a['control_integration_before'])
    tag=r['segment']+' '
    default,kp,effort=[np.asarray(contract[k]) for k in ('default_q','kp','training_effort')]
    limits=np.asarray(contract['joint_limits']);check.require(0<=n<=count and pre in (n,n+1),tag+'command/precontrol coverage')
    check.require(r['requested_controls']==count and r['attempted_controls']==n and r['attempted_precontrols']==pre and r['physics_steps']==steps,tag+'report actual counts')
    check.require(len(a['qpos'])==len(a['qvel'])==n+1,tag+'command state boundaries')
    check.exact(a['global_control'],np.arange(start,start+n,dtype=np.int64),tag+'global clock')
    check.exact(a['source_frame'],np.minimum(np.arange(start+11,start+n+11,dtype=np.int64),1579),tag+'source clock and terminal endpoint')
    modes=np.asarray([phase(control) for control in range(start,start+n)],np.int64)
    check.exact(a['controller_mode'],modes,tag+'phase clock')
    sub=a['physics_substeps'];check.require(sub.shape==(n,) and np.all((sub>=0)&(sub<=10)) and int(sub.sum())==steps,tag+'issued native substep count')
    check.require(n==0 or np.all(sub[:-1]==10),tag+'no skipped control substeps')
    clock=times(a['initial_integration'][0],steps)
    check.exact(a['physics_time'],clock,tag+'actual accumulated clock');check.exact(a['physics_expected_time'],clock,tag+'expected accumulated clock')
    check.exact(a['initial_integration'][1:31],a['qpos'][0],tag+'initial full291 q');check.exact(a['initial_integration'][31:60],a['qvel'][0],tag+'initial full291 dq')
    named=[];boundary=0
    for i,control in enumerate(range(start,start+n)):
        prefix=tag+str(control)+' ';q,dq=a['qpos'][i],a['qvel'][i];full=a['control_integration_before'][i]
        check.exact(full[0],clock[boundary],prefix+'precontrol time')
        check.exact(full[1:31],q,prefix+'full291 q');check.exact(full[31:60],dq,prefix+'full291 dq')
        check.exact(q,a['physics_qpos'][boundary],prefix+'native boundary q');check.exact(dq,a['physics_qvel'][boundary],prefix+'native boundary dq')
        sensed,terms=terms_fn(q[7:],dq[6:],q[3:7],dq[3:6],previous,default)
        check.exact(sensed,a['state'][i],prefix+'state52')
        current=np.zeros(1000,np.float32) if phase(control)==2 else goals(q,dq,control+11)
        wanted_features=features(current,previous,history,control)
        check.exact(wanted_features,a['features'][i],prefix+'causal features1323')
        check.exact(previous,a['previous_action'][i],prefix+'current prior');check.exact(previous,a['control_previous_action_before'][i],prefix+'precontrol prior')
        check.exact(flat(history),a['history'][i],prefix+'lag history');check.exact(flat(history),a['control_history_before'][i],prefix+'precontrol lag history')
        named.append({key:value.copy() for key,value in history.items()})
        if phase(control)==1:
            check.exact(a['base_target'][i],default,prefix+'direct default')
            delta=span*a['normalized_head'][i].astype(np.float64);raw=default+delta
            check.require(a['normalized_head'][i].dtype==np.float32 and np.isfinite(a['normalized_head'][i]).all(),prefix+'returned public head32')
            check.exact(delta,a['delta'][i],prefix+'widen before target multiplication')
            outgoing=actual_action(np.clip(raw,limits[:,0],limits[:,1]),default,kp,effort)
        else:
            check.exact(a['normalized_head'][i],np.zeros(23,np.float32),prefix+'inactive head')
            # Segment array delta becomes f64 if any learned row was issued; a hold-only array stays f32.
            check.exact(a['delta'][i],np.zeros(23,a['delta'].dtype),prefix+'original BFM zero delta')
            outgoing=a['action'][i]
            check.require(outgoing.dtype==np.float32,prefix+'BFM raw action32')
            check.exact(default+outgoing*.25*effort/kp,a['base_target'][i],prefix+'BFM raw action to original base')
            raw=a['base_target'][i]+np.zeros(23,np.float32)
        target=np.clip(raw,limits[:,0],limits[:,1]);actual=actual_action(target,default,kp,effort)
        for name,wanted in [('raw_proposal',raw),('target',target),('actual_normalized_action',actual),('action',outgoing)]:check.exact(a[name][i],wanted,prefix+name)
        advance(history,terms);previous=outgoing.copy();boundary+=int(sub[i])
        check.exact(a['qpos'][i+1],a['physics_qpos'][boundary],prefix+'ending q');check.exact(a['qvel'][i+1],a['physics_qvel'][boundary],prefix+'ending dq')
    check.exact(previous,a['final_previous_action'],tag+'final prior')
    check.require(int(a['final_recorded_controls'])==r['controller_recorded_controls']==start+n,tag+'committed controls')
    for name,value in history.items():check.exact(value,a['final_history_'+name],tag+'final history '+name)
    check.exact(a['final_integration'][0],clock[-1],tag+'final full291 time');check.exact(a['final_integration'][1:31],a['qpos'][-1],tag+'final full291 q');check.exact(a['final_integration'][31:60],a['qvel'][-1],tag+'final full291 dq')
    check.require(r['completed_full_controls']==int(np.sum(sub==10)),tag+'completed controls')
    check.require(r['full_segment_completed']==(n==count and pre==n and np.all(sub==10) and r['failure'] is None),tag+'full lifecycle definition')
    if cap is not None:
        rejected=pre==n+1;idx=n if rejected else n-1;control=start+idx
        check.exact(cap['integration_at_failure'],a['final_integration'],tag+'capsule final291')
        check.exact(cap['qpos_at_failure'],a['qpos'][-1],tag+'capsule q');check.exact(cap['qvel_at_failure'],a['qvel'][-1],tag+'capsule dq')
        check.exact(cap['previous_action_after'],previous,tag+'capsule current prior')
        check.exact(cap['previous_action_before'],a['control_previous_action_before'][idx],tag+'capsule original prior')
        check.exact(cap['warning_counts_at_failure'],a['physics_warning_counts'][-1],tag+'capsule warning counts')
        check.exact(cap['warning_lastinfo_at_failure'],a['physics_warning_lastinfo'][-1],tag+'capsule warning info')
        check.exact(cap['integration_before'],a['control_integration_before'][idx],tag+'capsule original precontrol291')
        if rejected and phase(control)==1:
            expected_x=features(goals(a['qpos'][-1],a['qvel'][-1],control+11),previous,history,control)
            if 'head_input_features' in cap:check.exact(cap['head_input_features'][0],expected_x,tag+'rejected causal head input')
            if 'features' in cap:check.exact(cap['features'],expected_x,tag+'rejected causal feature context')
        check.require(int(cap['global_control'])==control and int(cap['recorded_controls_after'])==start+n,tag+'capsule control identity')
        before=history if rejected else named[-1]
        preq=cap['integration_before'][1:31];prev=cap['integration_before'][31:60]
        _,preterms=terms_fn(preq[7:],prev[6:],preq[3:7],prev[3:6],cap['previous_action_before'],default)
        staged={name:value.copy() for name,value in before.items()};advance(staged,preterms)
        if 'history' in cap:check.exact(cap['history'],flat(before),tag+'capsule incoming history')
        for name,value in history.items():
            check.exact(cap['history_before_'+name],before[name],tag+'capsule before history '+name)
            check.exact(cap['history_after_'+name],value,tag+'capsule after history '+name)
            if 'staged_history_'+name in cap:check.exact(cap['staged_history_'+name],staged[name],tag+'capsule staged history '+name)
        if rejected:
            check.exact(a['control_previous_action_before'][idx],previous,tag+'rejected prior unchanged')
            check.exact(a['control_history_before'][idx],flat(history),tag+'rejected history unchanged')
            check.exact(a['control_integration_before'][idx],a['final_integration'],tag+'rejected physical state unchanged')
        elif phase(control)==1:
            check.exact(cap['head_input_features'][0],a['features'][idx],tag+'actual failure head input')
            check.exact(cap['head_output_0'][0],a['normalized_head'][idx],tag+'actual failure returned head output')
    else:check.require(r['failure'] is None,tag+'failure capsule required')
    return previous,history,named

def check_inference(report,recorded,check,cap=None):
    counts=report['inference_counts'];rejected=report['attempted_precontrols']>report['attempted_controls']
    for mode in range(3):
        nominal=sum(phase(control)==mode for control in range(recorded))
        for graph in ('backward','actor','head'):
            allowed=(mode==1 and graph=='head') or (mode in (0,2) and graph in ('backward','actor'))
            expected=nominal if allowed else 0
            attempted,returned=[counts[f'{mode}_{graph}_{stage}'] for stage in ('attempted','returned')]
            extra=int(rejected and phase(recorded)==mode and allowed)
            check.require(expected<=returned<=attempted<=expected+extra,'exact issued/capsule inference '+str(recorded)+' '+str(mode)+graph)
            if cap is not None:
                for stage,value in [('attempted',attempted),('returned',returned)]:check.require(int(cap[f'count_{mode}_{graph}_{stage}'])==value,'capsule inference count '+str(mode)+graph+stage)
    check.require(report['forbidden_inference_calls']==0,'no forbidden learned BFM')

def main():
    check=Checks();request_path=BASE/'request.json';request_sha=sha(request_path);request=read(request_path)
    output=BASE/'results_v1';output.mkdir(exist_ok=False)
    try:
        check.require(sys.platform!='win32' and np.__version__=='1.26.4','pinned WSL NumPy arithmetic')
        for path,digest in request['input_sha256'].items():check.require(sha(local(path))==digest,'input entry '+path)
        p={k:local(v) for k,v in request['paths'].items()};release=check_release(p,check)
        c=read(p['contract']);Features,terms_fn=pure(p);goals=Features(load(p['motion']),load(p['original29']),c)
        norm=load(p['norm']);span=norm['joint_span'].astype(np.float64)
        check.require(norm['feature_mean'].shape==norm['feature_std'].shape==(1323,) and norm['feature_mean'].dtype==norm['feature_std'].dtype==np.float32,'public1323 normalization')
        check.require(np.isfinite(norm['feature_mean']).all() and np.isfinite(norm['feature_std']).all() and np.all(norm['feature_std']>0),'finite positive normalization')
        check.exact(norm['feature_mean'][1000:],norm['context_mean'],'context normalization mean only; never substituted into causal input')
        check.exact(norm['joint_span'],np.diff(np.asarray(c['joint_limits']),axis=1)[:,0].astype(np.float32),'existing rounded native span')
        a=load(p['main_trace']);r=read(p['main_report']);cap=load(p['main_failure']) if 'main_failure' in p else None
        previous,history,named=segment(a,r,0,1569,np.zeros(23,np.float32),history_zero(),goals,terms_fn,c,span,check,cap)
        check_inference(r,len(a['target']),check,cap)
        b=load(p['baseline']);n=len(a['target']);prefix=min(n,250);steps=min(int(a['physics_substeps'][:prefix].sum()),2500)
        for name in ('qpos','qvel','target','source_frame','global_control','controller_mode','state','history','previous_action','action','base_target',
                     'control_integration_before','control_history_before','control_previous_action_before','physics_qpos','physics_qvel','physics_torque','physics_actuator_torque',
                     'physics_time','physics_expected_time','physics_warning_counts','physics_warning_lastinfo','physics_substeps'):
            boundary_count=prefix+int(prefix==0 or (prefix>0 and a['physics_substeps'][prefix-1]==10))
            count=steps if name in ('physics_torque','physics_actuator_torque') else steps+1 if name.startswith('physics_') and name!='physics_substeps' else boundary_count if name in ('qpos','qvel') else prefix
            check.exact(a[name][:count],b[name][:count],'original BFM actual prefix '+name)
        labels=load(p['labels']);centers=load(p['centers']);kept=np.r_[np.arange(52),np.arange(75,1023)]
        check.exact(a['features'][:prefix],nominal_features(b['features'][:prefix],b['previous_action'][:prefix],b['history'][:prefix]),'original BFM actual prefix features1323')
        if n>250:
            check.exact(a['control_integration_before'][250],b['final_integration'],'original BFM250 ending full291')
            for name in ('state','history','previous_action'):check.exact(a[name][250],labels[name][0],'query250 '+name)
            for name in named[250]:check.exact(named[250][name],labels['history_'+name][0],'query250 named '+name)
            check.exact(a['features'][250],nominal_features(labels['features'][:1],labels['previous_action'][:1],labels['history'][:1])[0],'query250 causal features1323')
            check.exact(a['features'][250],nominal_features(centers['features'][2038:2039],centers['previous_action'][2038:2039],centers['history'][2038:2039])[0],'query250 center2038 causal features1323')
            check.require(int(centers['dataset'][2038])==2 and int(centers['control'][2038])==250 and int(centers['source_frame'][2038])==261,'query250 center identity')
            check.exact(a['qpos'][250],labels['teacher_qpos'][0],'query250 q');check.exact(a['qvel'][250],labels['teacher_qvel'][0],'query250 dq')
            w=load(p['witness'])
            check.require(str(w['context_condition'].item())=='causal','activation witness causal condition')
            check.exact(w['actual_context'],context(a['previous_action'][250],named[250]),'activation witness actual incoming context')
            for actual,wanted in [('features','features'),('normalized_head','normalized_target'),('raw_proposal','raw_proposal'),('target','target'),('delta','delta')]:check.exact(a[actual][250],w[wanted],'activation witness '+actual)
            for name in PARITIES:check.require(read(p[name])['passed'] is True,name)
        hold=read(p['hold_report'])
        if 'hold_trace' in p:
            check.require(r['full_segment_completed'] is True,'hold only after complete main')
            h=load(p['hold_trace']);check.exact(h['initial_integration'],a['final_integration'],'continuous hold full291')
            check.exact(h['physics_warning_counts'][0],a['physics_warning_counts'][-1],'continuous hold warning counts');check.exact(h['physics_warning_lastinfo'][0],a['physics_warning_lastinfo'][-1],'continuous hold warning info')
            hcap=load(p['hold_failure']) if 'hold_failure' in p else None
            previous,history,_=segment(h,hold,1569,250,previous,history,goals,terms_fn,c,span,check,hcap)
            check_inference(hold,1569+len(h['target']),check,hcap)
        else:check.require(r['full_segment_completed'] is False and hold['attempted_controls']==0 and hold['no_reset_or_skip_to_terminal'] is True,'hold not run after first main failure')
        selected=np.arange(250,min(n,1269),dtype=np.int64);rows=[]
        j=np.arange(len(selected));teacher_features=nominal_features(labels['features'][j],labels['previous_action'][j],labels['history'][j]);teacher_target=labels['expert_target'][j]
        check.exact(labels['control'][j],selected,'same-clock label control mapping')
        check.exact(labels['source_frame'][j],selected+11,'same-clock label frame mapping')
        clipped=np.any(a['raw_proposal'][selected]!=a['target'][selected],axis=1);changed=np.any(a['features'][selected]!=teacher_features,axis=1)
        for idx,control in enumerate(selected):
            rows.append(dict(control=int(control),target_clipped=bool(clipped[idx]),feature_changed=bool(changed[idx]),
                feature_RMS_standardized=float(np.sqrt(np.mean(((a['features'][control].astype(np.float64)-teacher_features[idx])/norm['feature_std'])**2))),
                same_clock_expert_target_RMSE_rad=float(np.sqrt(np.mean((a['target'][control]-teacher_target[idx])**2))),
                same_clock_joint_velocity_RMSE=float(np.sqrt(np.mean((a['qvel'][control,6:]-labels['teacher_qvel'][idx,6:])**2)))))
        arrays={name:a[name][selected] for name in ('features','target','raw_proposal','normalized_head','previous_action','action','state','history','qpos','qvel')}
        np.savez_compressed(output/'actual_rows.npz',control=selected,teacher_features=teacher_features,teacher_target=teacher_target,**arrays)
        mapped,unmapped=map_selection(centers,selected)
        maps=run_maps(a,mapped,centers,c,p['map_core'],p['map_original'],output,check)
        maps['unmapped_actual_controls']=unmapped;maps['requested_actual_controls']=len(selected)
        for path,digest in request['input_sha256'].items():check.require(sha(local(path))==digest,'input final '+path)
        check.require(sha(request_path)==request_sha,'request immutable')
        first=lambda mask:int(selected[np.flatnonzero(mask)[0]]) if np.any(mask) else None
        report=dict(passed=True,evidence_audit_passed=True,actual_controls=n,moving_controls=len(selected),original_requested_controls=1569,
            context_condition='causal',features=1323,context_order='incoming_previous_action23_then_pre_update_history300',original_lifecycle_completed=r['full_segment_completed'],hold_completed=hold['full_segment_completed'],release_subjects=release,
            exact_checks=len(check.rows),checks=check.rows,first_feature_departure=first(changed),first_learned_target_clip=first(clipped),learned_clipped_commands=int(clipped.sum()),rows=rows,
            main_failure=r['failure'],hold_failure=hold.get('failure'),fixed_maps=maps,request_sha256=request_sha,trace_sha256=sha(p['main_trace']),
            actual_rows_sha256=sha(output/'actual_rows.npz'),input_sha256=request['input_sha256'],all_inputs_unchanged=True,
            model_calls=0,BFM_calls=0,native_steps=0,optimizer_updates=0,replans=0,behavioral_qualification=False,
            limitations=['Saved semantic consistency and previously independent native replay do not qualify stability or real-time plant behavior.',
                'After departure, same-clock expert rows have different states; these comparisons are not replanned expert commands.',
                'Causal context consistency does not prove hidden-state sufficiency or attribute a rollout difference uniquely to history.'])
        write(output/'report.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ('checks','rows','input_sha256')},indent=2))
    except BaseException as error:
        write(output/'failure.json',dict(passed=False,error=repr(error),active_check=check.active,checks=check.rows,request_sha256=request_sha,model_calls=0,native_steps=0));raise

if __name__=='__main__':main()
