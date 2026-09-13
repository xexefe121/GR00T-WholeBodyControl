"""Independent D3 saved-value algebra. No model, optimizer or producer imports."""
import numpy as np
from audit_math import position_errors, targets
from audit_context_math import metrics, WEIGHT
from audit_balanced_math import balanced_metrics, PRODUCER_WEIGHT_RULE
from audit_restoration import differences

BACKENDS=('initial_GPU32','final_GPU32','CPU64','GPU64','ORT64')
OLD_CORPORA=('nominal','full_state','physical')
CORPORA=OLD_CORPORA+('recovery',)
SIZES=(9904,354612,3054,1018)
PHASE_COUNTS=(99,819,100)
NAMES=('0.weight','0.bias','2.weight','2.bias','4.weight','4.bias')
SHAPES=((512,1323),(512,),(512,512),(512,),(23,512),(23,))

def source_errors(source):
    import torch
    errors=[]
    expected=dict(kind='direct_absolute_native23_target',condition='causal',ordinary_final_step=81000,
        optimizer_step=16000,fresh_optimizer=False,hidden_width=512,architecture=[1323,512,512,23],
        context_blinded=False,full_state_coefficient=WEIGHT,context_order='previous_action23_then_incoming_history300',
        response_weight_rule=PRODUCER_WEIGHT_RULE)
    for name,value in expected.items():errors+=differences(source.get(name),value,'source.'+name)
    actor=source['actor_state'];opt=source['optimizer_state']
    if tuple(actor)!=NAMES:errors.append('source.actor order')
    group=dict(params=list(range(6)),lr=1e-6,betas=(.9,.999),eps=1e-8,weight_decay=1e-5,
        amsgrad=False,maximize=False,foreach=False,capturable=False,differentiable=False,fused=False,decoupled_weight_decay=True)
    errors+=differences(opt['param_groups'],[group],'source.optimizer group')
    if set(opt['state'])!=set(range(6)):return errors+['source.optimizer identities']
    def finite(t,shape):return torch.is_tensor(t) and t.device.type=='cpu' and t.dtype==torch.float32 and tuple(t.shape)==shape and bool(torch.isfinite(t).all())
    for i,(name,shape) in enumerate(zip(NAMES,SHAPES)):
        if not finite(actor.get(name),shape):errors.append('source.actor '+name)
        state=opt['state'][i]
        if set(state)!={'step','exp_avg','exp_avg_sq'}:errors.append('source.state keys '+name);continue
        if not finite(state['step'],()) or float(state['step'])!=16000:errors.append('source.step '+name)
        for key in ('exp_avg','exp_avg_sq'):
            if not finite(state[key],shape):errors.append('source.moment '+name+'/'+key)
        if finite(state['exp_avg_sq'],shape) and not bool((state['exp_avg_sq']>=0).all()):errors.append('source.negative variance '+name)
    for key in ('feature_mean','feature_std'):
        if not finite(source[key],(1323,)):errors.append('source.normalization '+key)
    if finite(source['feature_std'],(1323,)) and not bool((source['feature_std']>0).all()):errors.append('source.normalization positivity')
    if not isinstance(source.get('rng'),dict) or set(source['rng'])!={'torch_cpu','torch_cuda','numpy','python'}:errors.append('source.RNG fields')
    for key,value in dict(ordinary_final_step=81000,optimizer_final_step=16000,architecture=[1323,512,512,23],
            first_layer_execution='split_old256_new256_original1000_plus323',export_first_layer_execution='monolithic_float64_1323').items():
        errors+=differences(source['request'].get(key),value,'source.request.'+key)
    return errors

def warm_initial_errors(initial,source):
    errors=source_errors(source)
    if errors:return errors
    for key in ('actor_state','optimizer_state','feature_mean','feature_std'):
        errors+=differences(initial[key],source[key],'initial.'+key)
    errors+=differences(initial['rng_after_restoration'],source['rng'],'initial.RNG')
    for key,value in dict(expansion_performed=False,ordinary_start_step=81000,optimizer_start_step=16000,hidden_width=512,condition='causal').items():
        errors+=differences(initial[key],value,'initial.'+key)
    return errors

def restoration_fields():
    return dict(restoration_completed=True,actor_exact=True,all_full512_parameters_preserved=True,
        learned_context_columns_preserved=True,learned_added_neurons_preserved=True,normalization_exact=True,
        warm_optimizer_exact=True,whole_optimizer_group_exact=True,optimizer_state_count=6,optimizer_start_step=16000,
        fresh_optimizer=False,RNG_exact=True,ordinary_start_step=81000,architecture=[1323,512,512,23],
        expansion_performed=False,weights_or_moments_zeroed=False,forward_execution='split_old256_new256_original1000_plus323',
        learning_rate_override_performed=False,model_forward_calls=0,optimizer_updates=0,native_steps=0)

def recovery_schema(rows,limits):
    """Independent literal connected-trajectory identities, never regenerate features."""
    specs={'causal_features':((1018,1323),np.float32),'features':((1018,1000),np.float32),'context':((1018,323),np.float32),
        'incoming_prior':((1018,23),np.float32),'incoming_history':((1018,300),np.float32),'expert_target':((1018,23),np.float64),
        'control':((1018,),np.int64),'source_frame':((1018,),np.int64),'phase':((1018,),np.int8),
        'first_student_state_query':((1018,),np.bool_),'plan_control':((1018,),np.int64),'plan_local':((1018,),np.int64),
        'feedback_clipped':((1018,23),np.bool_),'native_clipped':((1018,23),np.bool_)}
    for name,(shape,dtype) in specs.items():
        value=rows[name];assert value.shape==shape and value.dtype==dtype and np.isfinite(value).all(),name
    def same(a,b):assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
    c=np.arange(251,1269,dtype=np.int64);phase=np.repeat(np.arange(3,dtype=np.int8),PHASE_COUNTS)
    for name,value in [('control',c),('source_frame',c+11),('phase',phase),('first_student_state_query',np.arange(1018)==0),
            ('plan_control',251+5*((c-251)//5)),('plan_local',(c-251)%5)]:same(rows[name],value)
    same(rows['causal_features'][:,:1000],rows['features']);same(rows['causal_features'][:,1000:],rows['context'])
    same(rows['context'][:,:23],rows['incoming_prior']);same(rows['context'][:,23:],rows['incoming_history'])
    assert np.all(rows['expert_target']>=limits[:,0]) and np.all(rows['expert_target']<=limits[:,1])
    return tuple(np.flatnonzero(phase==i) for i in range(3))

def recovery_metrics(predictions,data,weights,energy,coefficient):
    result=balanced_metrics(metrics(predictions,data),weights,energy)
    p=predictions['recovery'];truth=data['recovery_teacher']
    labels=((truth-data['default'])/data['span'].astype(np.float64)).astype(np.float32)
    # Model arithmetic remains float32; reduce rounded squares in float64 independently.
    square=np.square(p-labels).astype(np.float64);cells=[]
    def errors(ids):return position_errors(p[ids],truth[ids],data['default'],data['span'],data['limits'])
    for phase,ids in enumerate(data['recovery_cells']):
        cells.append(dict(phase=phase,rows=len(ids),normalized_MSE=float(square[ids].mean()),first24=errors(ids[:24]),**errors(ids)))
    value=float(np.mean([c['normalized_MSE'] for c in cells]))
    result.update(recovery_cells=cells,recovery_objective=value,recovery_coefficient=coefficient,
        optimization_objective='balanced_old_objective_plus_recovery',
        combined_recovery_objective=result['balanced_weighted_objective']+coefficient*value)
    return result

def numerical_comparisons(outputs,data):
    result={}
    for corpus in CORPORA:
        for left,right in [('CPU64','GPU64'),('CPU64','ORT64'),('GPU64','ORT64')]:
            a=targets(outputs[left][corpus],data['default'],data['span'],data['limits'])[0]
            b=targets(outputs[right][corpus],data['default'],data['span'],data['limits'])[0]
            assert np.isfinite(a).all() and np.isfinite(b).all()
            result[corpus+'_'+left+'_'+right]=float(np.max(np.abs(a-b)))
    return result

def recovery_ledger_expectations(cells,objectives,old_total,coefficient):
    assert cells.ndim==2 and cells.shape[1]==3 and cells.dtype==np.float64 and np.isfinite(cells).all() and np.all(cells>=0)
    assert objectives.shape==cells.shape and objectives.dtype==np.float64 and np.isfinite(objectives).all() and np.all(objectives>=0)
    # D3 loss and coefficient product are float32; old total is float64.
    expected_weighted=(objectives[:,0].astype(np.float32)*np.float32(coefficient)).astype(np.float64)
    return dict(phase_mean=cells.mean(axis=1),weighted=expected_weighted,combined=old_total+expected_weighted)
