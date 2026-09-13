"""Fourteen fixed analytical head Jacobians total; no BFM or physical evaluation."""
import numpy as np

CONTROLS=(250,251,252,255,260,270,278)
BLOCKS={'joint_position':(0,23),'joint_velocity':(23,46),'root_angular_velocity':(46,49),
    'gravity':(49,52),'previous_target':(52,75),'root_linear_velocity':(75,78),'root_height':(78,79),
    'received_goals':(79,1023),'BFM_base':(1023,1046),'previous_raw_action':(1046,1069)}

def analyze(actor,mean,std,span,trace,contract,recorded_seven,ledger):
    weights=[actor.state_dict()['%d.weight'%k].numpy().astype(np.float64) for k in (0,2,4)]
    biases=[actor.state_dict()['%d.bias'%k].numpy().astype(np.float64) for k in (0,2,4)]
    mean,std,span=[np.asarray(a,np.float64) for a in (mean,std,span)]
    elu=lambda x:np.where(x>0,x,np.expm1(np.minimum(x,0)))
    norm=lambda x:float(np.linalg.svd(x,compute_uv=False)[0])
    limits=np.asarray(contract['joint_limits']);default=np.asarray(contract['default_q'])
    C=.25*np.asarray(contract['training_effort'])/np.asarray(contract['kp']);S=1/C
    rows=[];arrays={}
    for at,control in enumerate(CONTROLS):
        ledger['analytical_head_evaluations_attempted']+=1;ledger['analytical_control']=control
        x=trace['features'][control].astype(np.float64)
        a0=weights[0]@((x-mean)/std)+biases[0]
        a1=weights[1]@elu(a0)+biases[1]
        d0=np.where(a0>0,1.,np.exp(np.minimum(a0,0)))
        d1=np.where(a1>0,1.,np.exp(np.minimum(a1,0)))
        J=((((span[:,None]*weights[2])*d1[None,:])@weights[1])*d0[None,:])@weights[0]/std[None,:]
        mathematical=(weights[2]@elu(a1)+biases[2])*span
        error=float(np.max(np.abs(mathematical-recorded_seven[at])))
        assert np.isfinite(J).all() and error<1e-5
        prior=trace['previous_action'][control].astype(np.float64)
        previous_unclipped=default+prior*C
        mask=((previous_unclipped>limits[:,0])&(previous_unclipped<limits[:,1])).astype(float)
        Rtarget=J[:,52:75]*mask[None,:]+J[:,1046:1069]*S[None,:]
        Rraw=S[:,None]*Rtarget*C[None,:]
        B=np.eye(23)+J[:,1023:1046]
        proposal=trace['base_target'][control]+mathematical
        active=((proposal>limits[:,0])&(proposal<limits[:,1])).astype(float)
        rows.append(dict(control=control,analytical_float64_vs_recorded_Torch_max_delta_rad=error,
            feature_block_spectral_norm={name:norm(J[:,a:b]) for name,(a,b) in BLOCKS.items()},
            conditional_prior_target_equivalent_norm=norm(Rtarget),conditional_prior_raw_action_norm=norm(Rraw),
            conditional_prior_spectral_radius=float(np.max(np.abs(np.linalg.eigvals(Rtarget)))),
            conditional_BFM_base_total_target_norm=norm(B),native_clipped_outputs=int(np.sum(active==0)),
            previous_target_clipped_inputs=int(np.sum(mask==0))))
        arrays.update({f'control{control}_J_raw_features':J,f'control{control}_conditional_prior_target':Rtarget,
            f'control{control}_conditional_prior_raw':Rraw,f'control{control}_conditional_BFM_base':B})
        ledger['analytical_head_evaluations_returned']+=1
    return dict(controls=list(CONTROLS),rows=rows,analytical_head_evaluations=7,BFM_calls=0,physics_steps=0,
        convention='Smooth float64 evaluation of stored float32 weights, radian outputs/raw feature units; existing analytical formula.',
        limits=['BFM base, plant, history, goals held fixed for conditional recurrence; not closed-loop stability eigenvalues.',
            'All inputs are the same seven saved65000 failure states; no recovered trajectory or actual-state expert query.']),arrays
