"""Independent row enumeration, measured-history reconstruction and BFM audit.

Producer validators are never imported. Saved-array mode has no model, policy,
inference or dynamics. Selected inference mode executes only original baseline
graphs, with fixed counters, after the saved-array audit passes.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np


def local(path):
    p=str(path).replace('\\','/')
    if sys.platform!='win32' and len(p)>2 and p[1]==':':p='/mnt/'+p[0].lower()+p[2:]
    return Path(p)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,data):Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def arrays(path):
    with np.load(path,allow_pickle=False) as data:return {k:data[k].copy() for k in data.files}


def same(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()


class Proof:
    def __init__(self,folder):
        self.folder=folder;self.rows=[];self.context={};self.clip='';self.control=-1

    def equal(self,name,a,b):
        okay=same(a,b)
        self.rows.append((self.clip,self.control,name,okay))
        if not okay:
            self.fail(name,a,b)

    def require(self,name,condition):
        self.rows.append((self.clip,self.control,name,bool(condition)))
        if not condition:self.fail(name,np.asarray(condition),np.asarray(True))

    def fail(self,name,a,b):
        p=self.folder/'first_mismatch.npz'
        if not p.exists():
            np.savez_compressed(p,actual=np.asarray(a),expected=np.asarray(b),comparison=np.asarray(name),
                clip=np.asarray(self.clip),control=np.asarray(self.control),**self.context)
        raise ValueError('Independent mismatch: '+self.clip+' control'+str(self.control)+' '+name)

    def preserve_exception(self,error):
        path=self.folder/'exception_context.npz'
        if not path.exists():
            np.savez_compressed(path,clip=np.asarray(self.clip),control=np.asarray(self.control),
                error=np.asarray(repr(error)),**self.context)

    def save(self):
        np.savez_compressed(self.folder/'comparison_log.npz',
            clip=np.asarray([r[0] for r in self.rows]),control=np.asarray([r[1] for r in self.rows],np.int64),
            comparison=np.asarray([r[2] for r in self.rows]),passed=np.asarray([r[3] for r in self.rows],bool))
        return dict(comparisons=len(self.rows),counts_by_comparison=dict(Counter(r[2] for r in self.rows)),
            all_pass=all(r[3] for r in self.rows),comparison_log_sha256=sha(self.folder/'comparison_log.npz'))


def quaternion_matrix(q):
    # Independent code retains the original floating-point expression, including
    # signed zeros. Do not use a numerically equivalent vectorized gravity path.
    value=np.array(q,dtype=np.float64,copy=True)
    value/=np.linalg.norm(value)
    w,x,y,z=value
    return np.asarray(((1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)),
                       (2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)),
                       (2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y))))


def actual_observations(trace,contract,stop):
    default,kp,effort=[np.asarray(contract[k],np.float64) for k in ('default_q','kp','training_effort')]
    sizes=dict(actions=23,base_ang_vel=3,dof_pos=23,dof_vel=23,projected_gravity=3)
    history={key:np.zeros((4,size),np.float32) for key,size in sizes.items()}
    previous=np.zeros(23,np.float32)
    for control in range(stop+1):
        q,dq=trace['qpos'][control],trace['qvel'][control]
        relative=q[7:]-default
        gyro=dq[3:6]*.25
        gravity=quaternion_matrix(q[3:7]).T@np.array([0.,0.,-1.])
        state=np.r_[relative,dq[6:],gravity,gyro].astype(np.float32)
        flat=np.concatenate([history[key].reshape(-1) for key in sorted(history)]).copy()
        yield control,q,dq,previous.copy(),state,flat,{key:value.copy() for key,value in history.items()}
        if control<stop:
            terms=dict(actions=previous,base_ang_vel=gyro,dof_pos=relative,dof_vel=dq[6:],projected_gravity=gravity)
            for key in history:
                history[key][1:]=history[key][:-1].copy()
                history[key][0]=np.asarray(terms[key],np.float32)
            previous=((trace['target'][control]-default)*kp/(.25*effort)).astype(np.float32)


class CountedSession:
    def __init__(self,session,kind,counters):self.session=session;self.kind=kind;self.counters=counters
    def run(self,*args,**kwargs):
        self.counters[self.kind]+=1
        if self.counters[self.kind]>6847:raise RuntimeError('Selected graph-call limit exceeded.')
        return self.session.run(*args,**kwargs)


def validate_inputs(request):
    assert request['selected_rows']==6847
    assert request['expected_graph_calls']==dict(backward=6847,actor=6847)
    assert request['physics_authorized'] is False and request['fitting_authorized'] is False
    for path,digest in request['input_hashes'].items():
        assert 'walk008' not in path.lower()
        if sha(local(path))!=digest:raise ValueError('Audit input changed: '+path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--request',type=Path,required=True)
    p.add_argument('--phase',choices=['saved','inference'],required=True);args=p.parse_args()
    initial_request_sha256=sha(args.request)
    request=read(args.request);validate_inputs(request)
    folder=args.request.parent/('saved_array_audit' if args.phase=='saved' else 'baseline_inference_audit')
    folder.mkdir(exist_ok=False)
    proof=Proof(folder);counters=dict(backward=0,actor=0);started=time.perf_counter();selected=0
    summaries={}
    try:
        if args.phase=='inference':
            assert request['independent_inference_selected'] is True
            saved=read(local(request['saved_audit_report']))
            assert saved['pass_all'] is True and saved['selected_rows_checked']==6847
            assert saved['physics_steps']==saved['actor_inference_calls']==saved['backward_inference_calls']==0
        from student_linear_runtime import LinearFeatures
        # Importing this original feature implementation does not instantiate its
        # unused student/seed classes. Native model creation occurs only below.
        if args.phase=='inference':
            import mujoco
            assert mujoco.__version__=='3.2.3'
            def no_physics(*a,**k):raise RuntimeError('No dynamics authorized in independent label audit.')
            for name in ('mj_step','mj_step1','mj_step2'):setattr(mujoco,name,no_physics)
            from gear_sonic.utils.g1_true23_mjbatch_mpc import load_native_bundle
            from gear_sonic.utils.g1_true23_mjbatch_bfm_seed import Native23BFMRolloutSeed
        contract=read(local(request['bundle'])/'contract.json')
        default,kp,effort=[np.asarray(contract[k],np.float64) for k in ('default_q','kp','training_effort')]
        limits=np.asarray(contract['joint_limits'],np.float64)
        for case in request['cases']:
            proof.clip=case['clip'];proof.control=-1;proof.context={}
            trace=arrays(local(case['trace']));data=arrays(local(case['labels']));snap=arrays(local(case['snapshots']))
            total,stop=case['total_controls'],case['moving_stop'];count=stop-250
            qual=read(local(case['qualification']));snapshot_report=read(local(case['snapshot_report']))
            proof.require('root_qualified_trace_binding',sha(local(case['trace']))==qual['traces']['full']['sha256'])
            proof.require('root_full_source_and_both_quiet',qual['lifecycle_controls']==total and qual['source_controls']==stop-450 and qual['both_quiet_windows_pass'])
            proof.require('root_snapshot_independent_all_sample_proof',snapshot_report['all_recorded_samples_byteexact'] and
                snapshot_report['compared_physics_steps']==total*10 and snapshot_report['final291_matches_previously_reconstructed_endpoint_byteexact'])
            proof.require('root_snapshot_archive_hash',sha(local(case['snapshots']))==snapshot_report['snapshots_sha256'])
            proof.require('root_snapshot_original_trace_hash',snapshot_report['original_trace_sha256']==sha(local(case['trace'])))
            timeline=read(local(case['timeline']));phases={phase['name']:phase for phase in timeline['phases']}
            proof.require('original_phase_boundaries',phases['acquisition_ramp']['control_start']==250 and
                phases['acquisition_ramp']['control_stop']==350 and phases['source_motion']['control_start']==350 and
                phases['source_motion']['control_stop']==stop-100 and phases['return_ramp']['control_start']==stop-100 and
                phases['return_ramp']['control_stop']==stop and phases['returned_standing']['control_start']==stop)
            proof.equal('selected_controls',data['control'],np.arange(250,stop,dtype=np.int64))
            proof.equal('selected_received_frames',data['source_frame'],np.arange(261,stop+11,dtype=np.int64))
            proof.equal('selected_phases',data['phase'],np.r_[np.zeros(100,np.int64),np.ones(count-200,np.int64),np.full(100,2,np.int64)])
            proof.equal('native_joint_limits',data['joint_limits'],limits)
            proof.equal('native_joint_span',data['joint_span'],np.diff(limits,axis=1).ravel().astype(np.float32))
            proof.require('head_feature_and_BFM_state_schema',data['features'].shape==(count,1069) and data['features'].dtype==np.float32 and data['state'].shape==(count,52))
            proof.equal('teacher_all_boundary_qpos',data['teacher_qpos'],trace['qpos'][250:stop+1])
            proof.equal('teacher_all_boundary_qvel',data['teacher_qvel'],trace['qvel'][250:stop+1])
            proof.equal('label_full_integration_spec',data['integration_spec'],np.asarray(8191,dtype=data['integration_spec'].dtype))
            proof.require('complete291_schema',data['control_integration_before'].shape==(count,291))
            proof.equal('original_trace_identity',data['original_trace_sha256'],np.asarray(sha(local(case['trace']))))
            motion=arrays(local(case['reference']));original29=arrays(local(case['original29']))
            feature_builder=LinearFeatures(motion,original29,contract)
            if args.phase=='inference':
                native,c,original,_,_=load_native_bundle(local(request['bundle']),case['clip'])
                assert c==contract
                seed=Native23BFMRolloutSeed(native,c,original,local(request['onnx']),
                    dependency_directory=local(request['onnx_dependencies']),threads=1)
                for name in counters:seed.sessions[name]=CountedSession(seed.sessions[name],name,counters)
                identity=seed.identity()
                assert (identity['original_goal_horizon'],identity['position_gain'],identity['yaw_gain'])==(8,1.,2.)
                write(folder/(case['clip']+'_baseline_identity.json'),identity)
            before_calls=dict(counters);before_selected=selected;first_input=None
            for control,q,dq,previous,state,flat,named in actual_observations(trace,contract,stop):
                proof.control=control
                proof.context=dict(qpos=q,qvel=dq,previous_action=previous,history=flat,state=state,
                    source_frame=np.asarray(control+11),integration=snap['control_integration_before'][control],
                    **{'named_'+key:value for key,value in named.items()})
                proof.equal('actual_trace_previous_action',trace[case['previous_key']][control],previous)
                proof.equal('actual_trace_flat_history',trace[case['history_key']][control],flat)
                if control<250 or control>=stop:continue
                index=control-250;frame=control+11
                proof.equal('label_previous_action',data['previous_action'][index],previous)
                proof.equal('label_history300',data['history'][index],flat)
                proof.equal('label_actor_state52',data['state'][index],state)
                for key in named:proof.equal('label_named_history_'+key,data['history_'+key][index],named[key])
                proof.equal('label_actual_native_target',data['expert_target'][index],trace['target'][control])
                proof.require('actual_target_within_native_bounds',np.isfinite(data['expert_target'][index]).all() and
                    np.all(data['expert_target'][index]>=limits[:,0]) and np.all(data['expert_target'][index]<=limits[:,1]))
                proof.equal('label_full291',data['control_integration_before'][index],snap['control_integration_before'][control])
                proof.equal('label_time',data['control_time_before'][index],snap['time'][control])
                proof.equal('label_warning_counts',data['control_warning_counts_before'][index],snap['warning_counts'][control])
                proof.equal('label_warning_lastinfo',data['control_warning_lastinfo_before'][index],snap['warning_lastinfo'][control])
                proof.equal('native_trace_clock',snap['time'][control],trace['physics_time'][control*10])
                proof.equal('full291_qpos',data['control_integration_before'][index,1:31],q)
                proof.equal('full291_qvel',data['control_integration_before'][index,31:60],dq)
                proof.equal('full291_previous_actual_ctrl',data['control_integration_before'][index,89:112],trace['physics_torque'][control*10-1])
                proof.context.update(saved_raw=data['base_action'][index],saved_base=data['base_target'][index],
                    saved_features=data['features'][index],actual_native_target=trace['target'][control])
                base=default+data['base_action'][index]*.25*effort/kp
                proof.equal('saved_raw_to_unclipped_base_formula',data['base_target'][index],base)
                proof.equal('actual_target_minus_base_residual',data['residual_rad'][index],trace['target'][control]-base)
                features=feature_builder(q,dq,frame,base,previous)
                proof.equal('original1069_received_features',data['features'][index],features)
                if first_input is None:
                    first_input={key:hashlib.sha256(value.tobytes()).hexdigest() for key,value in
                        dict(qpos=q,qvel=dq,previous=previous,state=state,history=flat,features=features).items()}
                if args.phase=='inference':
                    sensed,_=seed._terms(q,dq,previous)
                    proof.equal('seed_sensor_formula',sensed,state)
                    goal=seed._goal(frame,q)
                    proof.context['goal']=goal
                    raw=seed.sessions['actor'].run(None,dict(state=state[None],last_action=previous[None],history=flat[None],z=goal))[0][0]*5
                    proof.context['inferred_raw']=raw
                    inferred_base=default+raw*.25*effort/kp
                    proof.context['inferred_base']=inferred_base
                    inferred_features=feature_builder(q,dq,frame,inferred_base,previous)
                    proof.context['inferred_features']=inferred_features
                    proof.equal('independent_actor_raw',data['base_action'][index],raw)
                    proof.equal('independent_unclipped_base',data['base_target'][index],inferred_base)
                    proof.equal('independent_actor_features1069',data['features'][index],inferred_features)
                selected+=1
                if selected%100==0 or selected==6847:
                    progress=dict(phase=args.phase,clip=case['clip'],control=control,selected_checked=selected,
                        expected_selected=6847,graph_calls=counters,physics_steps=0,elapsed_seconds=time.perf_counter()-started)
                    temp=folder/'progress.tmp';write(temp,progress);temp.replace(folder/'progress.json')
            summaries[case['clip']]=dict(selected_rows=selected-before_selected,
                independently_rebuilt_prefix_rows=stop+1,first_control250_input_sha256=first_input,
                graph_calls={key:counters[key]-before_calls[key] for key in counters})
        proof.clip='normalization';proof.control=-1;proof.context={}
        saved_norm=arrays(local(request['saved_normalization']));norm=arrays(local(request['original_normalization']))
        for key in ('feature_mean','feature_std'):proof.equal('unchanged_'+key,saved_norm[key],norm[key])
        assert selected==6847
        assert counters==(request['expected_graph_calls'] if args.phase=='inference' else dict(backward=0,actor=0))
        validate_inputs(request)
        assert sha(args.request)==initial_request_sha256,'Audit request changed during execution.'
        detail=proof.save()
        report=dict(kind='independent_broader_label_'+args.phase+'_audit',pass_all=True,
            selected_rows_checked=selected,cases=summaries,comparison_proof=detail,
            actor_inference_calls=counters['actor'],backward_inference_calls=counters['backward'],physics_steps=0,
            actual_BFM_output_authenticity_verified=args.phase=='inference',
            feature_normalization_refitted=False,labels_written=False,fitting_launched=False,
            producer_checks_imported=False,request_sha256=initial_request_sha256,source_sha256=sha(__file__),
            input_hashes=request['input_hashes'],elapsed_seconds=time.perf_counter()-started,
            original_wrapper_exit_status=read(local(request['preserved_exit_receipt']))['exit_code'])
        write(folder/'report.json',report)
        print(json.dumps(dict(pass_all=True,phase=args.phase,selected_rows=selected,graph_calls=counters,
            report_sha256=sha(folder/'report.json'),comparisons=detail['comparisons'])),flush=True)
    except BaseException as error:
        proof.preserve_exception(error)
        detail=proof.save()
        write(folder/'failure.json',dict(pass_all=False,error=repr(error),traceback=traceback.format_exc(),
            selected_rows_checked=selected,graph_calls=counters,comparisons=detail,
            physics_steps=0,labels_written=False,fitting_launched=False,request_sha256=initial_request_sha256,
            request_sha256_at_failure=sha(args.request)))
        raise


if __name__=='__main__':main()
