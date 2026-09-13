"""Causal context adapter over already-qualified immutable labels; no inference."""
from pathlib import Path
import numpy as np
from direct_data import read,sha,array_path,mapping_to_nominal
from full_state_data import load_full_state_data
from context_contract import (exact,join_context,history_flat,advance_history,HISTORY_WIDTHS,
    inverse_applied_action,weighted_context_moments,ExpandedFeatures)

def archive_context(path):
    with np.load(path,allow_pickle=False) as z:
        values={key:z[key].copy() for key in ('previous_action','history','control','source_frame','state','expert_target')}
        named={key:z['history_'+key].copy() for key in HISTORY_WIDTHS}
        if 'dataset' in z:values['dataset']=z['dataset'].copy()
    exact(history_flat(named),values['history'],'nominal named/flat history')
    values['named']=named
    values['context']=join_context(values['previous_action'],values['history'])
    return values

def verify_chronology(rows,contract):
    n=len(rows['control']);dataset=rows.get('dataset',np.zeros(n,np.int64))
    earlier=np.flatnonzero((dataset[1:]==dataset[:-1])&(rows['control'][1:]==rows['control'][:-1]+1))
    later=earlier+1
    expected=advance_history({key:value[earlier] for key,value in rows['named'].items()},rows['state'][earlier],rows['previous_action'][earlier])
    for key,value in expected.items():exact(rows['named'][key][later],value,'chronological nominal '+key)
    exact(rows['previous_action'][later],inverse_applied_action(rows['expert_target'][earlier],contract),'nominal prior follows actual teacher target')
    return len(earlier)

def checked_physical(paths,pins,key):
    spec=read(paths['physical_manifest'])['arrays'][key];path=array_path(paths['physical_manifest'],spec)
    if pins.get(path.as_posix())!=spec['sha256'] or sha(path)!=spec['sha256']:raise ValueError('Physical context source not frozen: '+key)
    value=np.load(path,allow_pickle=False)
    if list(value.shape)!=spec['shape'] or str(value.dtype)!=spec['dtype']:raise ValueError('Physical context shape/dtype: '+key)
    return value

def load_context_data(request,pins):
    paths=request['paths'];data=load_full_state_data(paths,request['full_state_paths'],pins)
    archive=archive_context(paths['centers']);nominal_context=np.empty((9904,323),np.float32)
    contract=read(paths['contract']);chronology_pairs=verify_chronology(archive,contract)
    ids=mapping_to_nominal(archive['dataset'],archive['control'],data['dataset'],data['control'])
    exact(ids,data['center_map'],'nominal center map')
    nominal_context[ids]=archive['context'];assigned=np.zeros(9904,np.bool_);assigned[ids]=True
    for code,key in ((3,'pico'),(4,'walk002')):
        rows=archive_context(paths[key]);ids=mapping_to_nominal(np.full(len(rows['control']),code,np.int64),rows['control'],data['dataset'],data['control'])
        chronology_pairs+=verify_chronology(rows,contract)
        if assigned[ids].any():raise ValueError('Repeated context assignment.')
        exact(rows['source_frame'],data['frame'][ids],key+' source frame');nominal_context[ids]=rows['context'];assigned[ids]=True
    if not assigned.all():raise ValueError('All nominal contexts required.')
    center_context=nominal_context[data['center_map']].copy();exact(center_context,archive['context'],'center context')
    start=checked_physical(paths,pins,'center_index');successor=checked_physical(paths,pins,'successor_center_index')
    exact(data['physical_successor'],data['center_map'][successor],'physical successor context map')
    incoming=checked_physical(paths,pins,'incoming_history');incoming_prior=checked_physical(paths,pins,'incoming_raw_prior')
    exact(incoming,archive['history'][start],'physical incoming history');exact(incoming_prior,archive['previous_action'][start],'physical incoming prior')
    named={key:checked_physical(paths,pins,'advanced_history_'+key) for key in HISTORY_WIDTHS}
    advanced=checked_physical(paths,pins,'advanced_history');exact(history_flat(named),advanced,'physical named/flat advanced history')
    expected=advance_history({key:archive['named'][key][start] for key in HISTORY_WIDTHS},archive['state'][start],archive['previous_action'][start])
    for key in HISTORY_WIDTHS:exact(named[key],expected[key],'physical once-shifted '+key)
    target=checked_physical(paths,pins,'policy_applied_target');prior=checked_physical(paths,pins,'policy_actual_normalized_action')
    exact(prior,inverse_applied_action(target,read(paths['contract'])),'physical inverse actually applied target')
    physical_context=join_context(prior,advanced)
    mean,std,mean64,variance64=weighted_context_moments(nominal_context,data['cells'])
    raw_prior=checked_physical(paths,pins,'outgoing_raw_prior')
    clipped=checked_physical(paths,pins,'policy_native_target_clipped')
    data.update(nominal_context=nominal_context,center_context=center_context,physical_context=physical_context,
        context_mean=mean,context_std=std,context_mean64=mean64,context_variance64=variance64,
        context_proof=dict(all_nominal_rows_assigned_once=True,nominal_rows=9904,center_rows=3057,physical_rows=3054,
            full_state_context_rule='center_index = endpoint_row //116; current perturbation never shifts history',
            nominal_chronological_history_and_prior_pairs=chronology_pairs,
            first_row_context='Original source-bound first row per dataset, never synthesized',
            physical_history_once_shifted_exact=True,physical_actual_prior_exact=True,
            physical_applied_prior_differs_raw_rows=int(np.any(prior!=raw_prior,axis=1).sum()),
            physical_applied_prior_differs_raw_components=int((prior!=raw_prior).sum()),
            physical_native_clipped_rows=int(np.any(clipped,axis=1).sum()),
            no_model_calls=True,no_native_steps=True))
    return data

def condition_data(data,condition):
    result=data.copy()
    for feature,context,indices in (
        ('features','nominal_context',np.arange(9904,dtype=np.int64)),
        ('full_state_features','center_context',np.arange(354612,dtype=np.int64)//116),
        ('physical_features','physical_context',np.arange(3054,dtype=np.int64))):
        result[feature]=ExpandedFeatures(data[feature],data[context],indices,condition,data['context_mean'])
    return result
