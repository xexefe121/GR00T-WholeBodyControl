"""Independent saved-boundary comparison. Never imports producer extraction."""
import hashlib
import json
from pathlib import Path
import numpy as np

BASE=Path(__file__).resolve().parent
PRODUCER=BASE.parent/'direct_target_width512_expert_recovery_v1'
INPUTS=PRODUCER/'inputs'
checks=[]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def subject(p):return dict(path=Path(p).as_posix(),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def require(value,name):
    if not value:raise AssertionError(name)
    checks.append(name)
def exact(a,b,name):
    a,b=np.asarray(a),np.asarray(b)
    require(a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),name)
def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def main():
    output=BASE/'boundary_review.json'
    if output.exists():raise FileExistsError(output)
    request_path=PRODUCER/'input_preparation_request.json';selection_path=INPUTS/'selection_receipt.json'
    request=read(request_path);selection=read(selection_path)
    report=dict(passed=False,boundary_review_pass=False,source_review_subject=subject(BASE/'source_review.json'),
        input_request_subject=subject(request_path),selection_subject=subject(selection_path),
        task_model_calls=0,native_calls=0,replans=0,optimizer_calls=0,
        labels_admissible=False,actual_recovery_execution_cleared=False)
    try:
        require(request['root_selected_saved_input_preparation'] is True,'saved input extraction selected')
        require(sha(request_path)==selection['request_sha256'],'actual extraction request bound')
        require(selection['passed'] is True and selection['model_calls']==selection['native_steps']==selection['replans']==0,'producer saved-only accounting')
        for role,item in request['subjects'].items():
            require(sha(item['path'])==item['sha256'],'input subject '+role)
        require(selection['input_sha256']=={v['path']:v['sha256'] for v in request['subjects'].values()},'exact input subject membership')
        require(selection['outputs']=={p.name:sha(p) for p in INPUTS.glob('*.npz')},'exact output archive membership and hashes')
        require(request['subjects']['trace']['sha256']=='b9061a1c9dc6aee65d713f16909a3135f072d94a5bc20af750b5958cc493629d','prespecified width trace')
        original=load(request['subjects']['trace']['path'])
        snap=load(INPUTS/'precontrol251.npz');prefix=load(INPUTS/'actual_prefix251.npz')
        contract=read(request['subjects']['contract']['path'])
        prepared=read(PRODUCER/'source_preparation_v2.json')
        for name,digest in prepared['source_sha256'].items():
            require(sha(PRODUCER/'source_snapshot_v1'/name)==digest,'executed extraction source '+name)
        control_fields=('target','source_frame','global_control','controller_mode','joint_error','root_error','physics_substeps',
                        'control_integration_before','control_history_before','control_previous_action_before','previous_action','action','history','delta')
        boundary_fields=('physics_qpos','physics_qvel','physics_time','physics_expected_time','physics_warning_counts','physics_warning_lastinfo')
        output_fields=('physics_torque','physics_actuator_torque','range_excess','velocity_ratio','effort_ratio','clock_error')
        expected_prefix=set(control_fields+boundary_fields+output_fields+('qpos','qvel','initial_integration','final_integration','final_previous_action','final_recorded_controls'))
        widths={'actions':23,'base_ang_vel':3,'dof_pos':23,'dof_vel':23,'projected_gravity':3}
        expected_prefix|={kind+key for key in widths for kind in ('control_history_before_','final_history_')}
        require(set(prefix)==expected_prefix,'exact prefix schema')
        expected_snap={'integration','integration_state_spec','qpos','qvel','history_flat','previous_action','warning_counts',
                       'warning_lastinfo','recorded_controls','expected_time','actual_time'}|{'history_'+key for key in widths}
        require(set(snap)==expected_snap,'exact snapshot schema')
        for key in control_fields:exact(prefix[key],original[key][:251],'prefix control '+key)
        for key in ('qpos','qvel'):exact(prefix[key],original[key][:252],'prefix state '+key)
        for key in boundary_fields:exact(prefix[key],original[key][:2511],'prefix physics boundary '+key)
        for key in output_fields:exact(prefix[key],original[key][:2510],'prefix physics output '+key)
        exact(prefix['global_control'],np.arange(251,dtype=np.int64),'global controls0 through250')
        exact(prefix['source_frame'],np.arange(11,262,dtype=np.int64),'original source frames11 through261')
        exact(prefix['controller_mode'],np.r_[np.zeros(250,np.int64),np.ones(1,np.int64)],'BFM250 and learned250 modes')
        exact(prefix['physics_substeps'],np.full(251,10,np.int64),'all prefix controls have10 actual steps')
        full=original['control_integration_before'][251]
        exact(snap['integration'],full,'full291 including hidden integration tail')
        require(full.shape==(291,) and full.dtype==np.float64 and np.isfinite(full).all(),'full291 finite float64')
        exact(prefix['final_integration'],full,'prefix endpoint full291')
        exact(prefix['initial_integration'],original['initial_integration'],'global0 initial full291 unchanged')
        exact(snap['integration_state_spec'],np.asarray(8191,np.int64),'integration state spec')
        exact(snap['recorded_controls'],np.asarray(251,np.int64),'snapshot recorded251')
        exact(prefix['final_recorded_controls'],snap['recorded_controls'],'prefix recorded251')
        exact(snap['qpos'],original['qpos'][251],'snapshot actual qpos251')
        exact(snap['qvel'],original['qvel'][251],'snapshot actual qvel251')
        exact(full[1:31],snap['qpos'],'full291 qpos coordinates')
        exact(full[31:60],snap['qvel'],'full291 qvel coordinates')
        exact(snap['qpos'],original['physics_qpos'][2510],'qpos actual native boundary2510')
        exact(snap['qvel'],original['physics_qvel'][2510],'qvel actual native boundary2510')
        for target,source in [('warning_counts','physics_warning_counts'),('warning_lastinfo','physics_warning_lastinfo')]:
            exact(snap[target],original[source][2510],'selected boundary '+target)
        clock=np.empty(2511,np.float64);clock[0]=0.
        for i in range(2510):clock[i+1]=clock[i]+.002
        exact(prefix['physics_time'],clock,'repeated native clock through2510')
        exact(prefix['physics_expected_time'],clock,'repeated expected clock through2510')
        for key in ('expected_time','actual_time'):exact(snap[key],clock[-1],key+' is accumulated clock')
        exact(full[0],clock[-1],'full291 accumulated time')
        default,kp,effort=[np.asarray(contract[k],np.float64) for k in ('default_q','kp','training_effort')]
        prior=((original['target'][250]-default)*kp/(.25*effort)).astype(np.float32)
        for value,label in [(snap['previous_action'],'snapshot'),(prefix['final_previous_action'],'prefix'),
                            (original['control_previous_action_before'][251],'original incoming251'),(original['action'][250],'actual outgoing250')]:
            exact(value,prior,label+' inverse actual applied target250')
        exact(original['control_previous_action_before'][1:252],original['action'][:251],'actual prior chronology')
        exact(snap['history_flat'],original['control_history_before'][251],'snapshot incoming history251')
        offset=0
        buffers={key:np.zeros((4,width),np.float32) for key,width in widths.items()}
        for key,width in widths.items():
            block=slice(offset,offset+4*width);offset+=4*width
            exact(snap['history_'+key],snap['history_flat'][block].reshape(4,width),'snapshot named '+key)
            exact(prefix['final_history_'+key],snap['history_'+key],'prefix final named '+key)
            exact(prefix['control_history_before_'+key],original['control_history_before'][:251,block].reshape(251,4,width),'prefix named '+key)
        # Independently rebuild 251 measured history pushes using the declared
        # online BFM observation arithmetic, without importing teacher helpers.
        for control in range(251):
            flat=np.concatenate([buffers[k].reshape(-1) for k in sorted(buffers)])
            exact(flat,original['control_history_before'][control],'measured incoming history '+str(control))
            q=original['qpos'][control];dq=original['qvel'][control]
            quaternion=q[3:7].copy();quaternion/=np.linalg.norm(quaternion)
            w,x,y,z=quaternion
            rotation=np.asarray(((1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)),
                                 (2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)),
                                 (2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y))))
            terms=dict(actions=original['control_previous_action_before'][control],base_ang_vel=dq[3:6]*.25,
                       dof_pos=q[7:]-default,dof_vel=dq[6:],projected_gravity=rotation.T@np.asarray([0.,0.,-1.]))
            for key in buffers:
                buffers[key][1:]=buffers[key][:-1].copy();buffers[key][0]=np.asarray(terms[key],np.float32)
        exact(np.concatenate([buffers[k].reshape(-1) for k in sorted(buffers)]),snap['history_flat'],'all251 measured pushes reach incoming251')
        for role,item in request['subjects'].items():require(sha(item['path'])==item['sha256'],'post-read input '+role)
        for name,digest in selection['outputs'].items():require(sha(INPUTS/name)==digest,'post-read output '+name)
        report.update(passed=True,boundary_review_pass=True,checks=len(checks),check_names=checks,
            boundary=dict(global_control=251,original_frame=262,prefix_controls=251,prefix_native_steps=2510,
                          original_BFM_controls=250,learned_prefix_controls=1,actual_time=float(clock[-1]),
                          full291_bitexact=True,all_prefix_arrays_bitexact=True,all251_history_pushes_bitexact=True,
                          incoming_prior_is_inverse_actual_target250=True),
            snapshot_subject=subject(INPUTS/'precontrol251.npz'),prefix_subject=subject(INPUTS/'actual_prefix251.npz'),
            input_sha256={v['path']:v['sha256'] for v in request['subjects'].values()},
            limitations=['Boundary identity qualifies continuation input only. Full original source and conditional hold still require actual recovery and independent audits.',
                         'Copied prefix controls0..250 are not new expert labels.'])
    except BaseException as exc:
        report.update(checks=len(checks),check_names=checks,error=dict(type=type(exc).__name__,message=str(exc)))
    report['review_source']=subject(Path(__file__))
    with output.open('x',newline='\n') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(report=subject(output),passed=report['passed'],checks=len(checks),error=report.get('error'))))
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
