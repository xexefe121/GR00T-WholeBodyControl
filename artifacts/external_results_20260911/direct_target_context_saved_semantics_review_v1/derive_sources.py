"""One-time textual derivation from immutable 65000 saved auditor; no task data."""
from pathlib import Path
BASE=Path(__file__).resolve().parent


def replace(text,old,new):
    assert text.count(old)==1,(old,text.count(old))
    return text.replace(old,new)


def main():
    p=BASE/'audit_saved.py';s=p.read_text()
    s=replace(s,'from fixed_maps import run_maps','from fixed_maps import run_maps\nfrom context_math import context, features, nominal_features, map_selection')
    s=replace(s,"wanted_features=np.zeros(1000,np.float32) if phase(control)==2 else goals(q,dq,control+11)",
        "current=np.zeros(1000,np.float32) if phase(control)==2 else goals(q,dq,control+11)\n        wanted_features=features(current,previous,history,control)")
    s=replace(s,"prefix+'features1000'","prefix+'causal features1323'")
    s=replace(s,"check.exact(cap['integration_before'],a['control_integration_before'][idx],tag+'capsule original precontrol291')",
        "check.exact(cap['integration_before'],a['control_integration_before'][idx],tag+'capsule original precontrol291')\n        if rejected and phase(control)==1:\n            expected_x=features(goals(a['qpos'][-1],a['qvel'][-1],control+11),previous,history,control)\n            if 'head_input_features' in cap:check.exact(cap['head_input_features'][0],expected_x,tag+'rejected causal head input')\n            if 'features' in cap:check.exact(cap['features'],expected_x,tag+'rejected causal feature context')")
    s=replace(s,"norm=load(p['norm']);span=norm['joint_span'].astype(np.float64)",
        "norm=load(p['norm']);span=norm['joint_span'].astype(np.float64)\n        check.require(norm['feature_mean'].shape==norm['feature_std'].shape==(1323,) and norm['feature_mean'].dtype==norm['feature_std'].dtype==np.float32,'public1323 normalization')\n        check.require(np.isfinite(norm['feature_mean']).all() and np.isfinite(norm['feature_std']).all() and np.all(norm['feature_std']>0),'finite positive normalization')\n        check.exact(norm['feature_mean'][1000:],norm['context_mean'],'context normalization mean only; never substituted into causal input')")
    s=replace(s,"check.exact(a['features'][:prefix],b['features'][:prefix,kept],'original BFM actual prefix features1000')",
        "check.exact(a['features'][:prefix],nominal_features(b['features'][:prefix],b['previous_action'][:prefix],b['history'][:prefix]),'original BFM actual prefix features1323')")
    s=replace(s,"check.exact(a['features'][250],labels['features'][0,kept],'query250 pure features')\n            check.exact(a['features'][250],centers['features'][2038,kept],'query250 center2038 features')",
        "check.exact(a['features'][250],nominal_features(labels['features'][:1],labels['previous_action'][:1],labels['history'][:1])[0],'query250 causal features1323')\n            check.exact(a['features'][250],nominal_features(centers['features'][2038:2039],centers['previous_action'][2038:2039],centers['history'][2038:2039])[0],'query250 center2038 causal features1323')\n            check.require(int(centers['dataset'][2038])==2 and int(centers['control'][2038])==250 and int(centers['source_frame'][2038])==261,'query250 center identity')")
    s=replace(s,"w=load(p['witness'])","w=load(p['witness'])\n            check.require(str(w['context_condition'].item())=='causal','activation witness causal condition')\n            check.exact(w['actual_context'],context(a['previous_action'][250],named[250]),'activation witness actual incoming context')")
    s=replace(s,"teacher_features=labels['features'][j][:,kept];teacher_target=labels['expert_target'][j]",
        "teacher_features=nominal_features(labels['features'][j],labels['previous_action'][j],labels['history'][j]);teacher_target=labels['expert_target'][j]")
    s=replace(s,"maps=run_maps(a,selected,centers,c,p['map_core'],p['map_original'],output,check)",
        "mapped,unmapped=map_selection(centers,selected)\n        maps=run_maps(a,mapped,centers,c,p['map_core'],p['map_original'],output,check)\n        maps['unmapped_actual_controls']=unmapped;maps['requested_actual_controls']=len(selected)")
    s=replace(s,"original_lifecycle_completed=r['full_segment_completed']", "context_condition='causal',features=1323,context_order='incoming_previous_action23_then_pre_update_history300',original_lifecycle_completed=r['full_segment_completed']")
    s=replace(s,"'The direct1000 head omits action/history; history remains original BFM startup/terminal handoff bookkeeping.'",
        "'Causal context consistency does not prove hidden-state sufficiency or attribute a rollout difference uniquely to history.'")
    p.write_text(s)
    p=BASE/'release_checks.py';s=p.read_text().replace('unified65000','causal68000').replace('==65000','==68000').replace('65000 release kind','68000 causal release kind').replace('actual completed65000 fit','actual completed68000 fit').replace('ordinary65000_full58_same_weight_fp64_export','ordinary68000_matched_context_same_weight_fp64_export')
    s=replace(s,"b=read(p['binding']);launch=read(p['launch']);post=read(p['posthash']);owner=read(p['owner'])",
        "b=read(p['binding']);launch=read(p['launch']);post=read(p['posthash']);owner=read(p['owner'])\n    check.require(b['context_condition']=='causal' and b['architecture']==[1323,256,256,23],'explicit causal1323 activation')")
    s=replace(s,"source_review=read(p['source_review']);frozen=read(p['frozen'])",
        "paired=read(p['subject_paired_report'])\n    check.require(paired['completed'] is True and paired['conditions']==['blinded','causal'] and paired['ordinary_final_step']==68000 and paired['updates_per_condition']==3000,'completed matched68000 pair')\n    check.require(fit['condition']=='causal' and fit['features']==1323 and fit['context_features']==323 and fit['additional_updates']==3000,'actual causal endpoint')\n    check.require(expected['fit_report']==expected['causal_fit_report'],'selected causal report identity')\n    for condition in ('blinded','causal'):\n        value=read(p['subject_'+condition+'_fit_report'])\n        check.require(value['completed'] is True and value['condition']==condition and value['ordinary_final_step']==68000 and value['numerical_gate_passed'] is True and value['export_parity_passed'] is True,'matched condition completed '+condition)\n        check.require(paired['condition_report_sha256'][condition]==expected[condition+'_fit_report'],'paired condition report subject '+condition)\n    source_review=read(p['source_review']);frozen=read(p['frozen'])")
    s=replace(s,"check.require(witness_owner['owner_completion_accounting_passed'] is True", "check.require(witness['context_condition']=='causal' and witness['features']==1323 and witness['expected_head_calls']==witness['attempted_head_calls']==witness['returned_head_calls']==1,'single causal WSL witness')\n    check.require(witness_owner['owner_completion_accounting_passed'] is True")
    s=replace(s,"request['ordinary_final_step']==68000,name+' model activation subject'", "request['ordinary_final_step']==68000 and request['context_condition']=='causal',name+' model activation subject'")
    p.write_text(s)
    p=BASE/'prepare_request.py';s=p.read_text()
    s=replace(s,"binding=read(RUN/'evaluation_binding.json');launch=RUN/'evaluation_process';clearance=read(launch/'launch_clearance.json')",
        "binding=read(RUN/'evaluation_binding.json');assert binding['context_condition']=='causal' and binding['ordinary_final_step']==68000\n    launch=RUN/'evaluation_process';clearance=read(launch/'launch_clearance.json')")
    s=s.replace('one_pure_saved_ordinary65000_runtime_semantics_and_fixed_maps','one_pure_saved_causal68000_runtime_semantics_and_fixed_maps')
    s=replace(s,"original_requested_controls=1569,conditional_continuous_hold_controls=250,", "context_condition='causal',public_features=1323,original_requested_controls=1569,conditional_continuous_hold_controls=250,")
    s=replace(s,'All actually issued learned controls only, same-clock frozen query250 maps and six preclip tangent groups; no replanned expert truth.', 'Only actually issued learned controls with existing unique same-clock query250 maps; missing coverage explicit, duplicate maps fail; no replanned expert truth.')
    p.write_text(s)


if __name__=='__main__':main()
