"""Independent saved-array contracts only. No simulator, actor, solver imports."""
from pathlib import Path
import hashlib,json
import numpy as np
OUT=Path(__file__).resolve().parent
ROOT=OUT.parent/'student_actual_oracle_control1_v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
tp=OUT/'trace.frozen.npz';t=np.load(tp)
sp=ROOT/'inputs/actual_student_control1.npz';s=np.load(sp)
pp=ROOT/'inputs/student_control0_prefix.npz';p=np.load(pp)
cp=Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1/contract.json')
c=json.loads(cp.read_text());n=len(t['target']);checks=[]
def same(a,b,name):
    np.testing.assert_array_equal(a,b,err_msg=name);checks.append(name)

for k in p.files:
    kk={'physics_warning_counts':'physics_warning_number','physics_actuator_torque':'physics_actuator_force'}.get(k,k)
    if kk not in t.files or k in ('final_integration','integration_state_spec'):continue
    a,b=t[kk],p[k]
    if k=='initial_integration':same(a,b,k)
    elif b.ndim:same(a[:len(b)],b,'prefix_'+kk)
same(t['branch_initial_integration'],s['integration'],'branch_integration')
same(t['control_integration_before'][1],s['integration'],'control1_integration')
same(t['qpos'][1],s['qpos'],'boundary_q')
same(t['qvel'][1],s['qvel'],'boundary_dq')
same(t['control_history_before'][1],s['history_flat'],'boundary_history')
same(t['control_previous_action_before'][1],s['previous_action'],'boundary_previous_action')
same(t['physics_warning_number'][10],s['warning_counts'],'boundary_warning_counts')
same(t['physics_warning_lastinfo'][10],s['warning_lastinfo'],'boundary_warning_lastinfo')
same(t['physics_time'][10],s['actual_time'],'boundary_actual_clock')
same(t['physics_expected_time'][10],s['expected_time'],'boundary_expected_clock')
same(t['global_control'],np.arange(n),'global_controls')
same(t['source_frame'],np.arange(n)+11,'source_frames')
same(t['physics_substeps'],np.full(n,10),'full_control_substeps')
same(t['controller_mode'][1:],np.ones(n-1),'expert_modes')
same(t['qpos'],t['physics_qpos'][::10],'control_physics_q_boundaries')
same(t['qvel'],t['physics_qvel'][::10],'control_physics_dq_boundaries')
for k in ['physics_time','physics_expected_time','physics_warning_number','physics_warning_lastinfo','physics_qpos','physics_qvel']:
    assert len(t[k])==n*10+1,k
for k in ['physics_torque','physics_actuator_force']:
    assert t[k].shape==(n*10,23),k
expected=np.empty(n*10+1);expected[0]=t['physics_expected_time'][0]
for i in range(n*10):expected[i+1]=expected[i]+.002
same(t['physics_expected_time'],expected,'independent_accumulated_expected_clock')
same(t['physics_time'],expected,'actual_clock_matches_repeated_expected')
same(t['physics_warning_number'],np.zeros((n*10+1,8),dtype=t['physics_warning_number'].dtype),'zero_engine_warning_ledger')

keys=['actions','base_ang_vel','dof_pos','dof_vel','projected_gravity']
h={k:np.zeros((4,width),np.float32) for k,width in zip(keys,[23,3,23,23,3])}
previous=np.zeros(23,np.float32)
default=np.array(c['default_q']);kp=np.array(c['kp']);effort=np.array(c['training_effort'])
for i in range(n):
    flat=np.concatenate([h[k].ravel() for k in keys])
    np.testing.assert_array_equal(flat,t['control_history_before'][i],err_msg=f'history_{i}')
    np.testing.assert_array_equal(previous,t['previous_action'][i],err_msg=f'previous_{i}')
    np.testing.assert_array_equal(previous,t['control_previous_action_before'][i],err_msg=f'prior_{i}')
    for k in keys:np.testing.assert_array_equal(h[k],t['control_history_'+k][i],err_msg=f'named_{k}_{i}')
    quat=t['qpos'][i,3:7].copy();quat/=np.linalg.norm(quat);w,x,y,z=quat
    rot=np.asarray(((1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)),(2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)),(2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y))))
    terms=dict(actions=previous,base_ang_vel=t['qvel'][i,3:6]*.25,dof_pos=t['qpos'][i,7:]-default,dof_vel=t['qvel'][i,6:],projected_gravity=rot.T@np.array([0.,0.,-1.]))
    for k in keys:h[k][1:]=h[k][:-1].copy();h[k][0]=np.asarray(terms[k],np.float32)
    previous=p['action'][0].copy() if i==0 else ((t['target'][i]-default)*kp/(.25*effort)).astype(np.float32)
    np.testing.assert_array_equal(previous,t['action'][i],err_msg=f'action_{i}')
same(previous,t['final_previous_action'],'final_previous_action')
same(np.asarray(n),t['final_recorded_controls'],'final_recorded_controls')
for k in keys:same(h[k],t['final_history_'+k],'final_history_'+k)
result=dict(kind='independent_saved_array_checkpoint_contract_audit',passed=True,full_source_qualified=False,controls=n,source_controls=max(0,n-350),physics_steps=n*10,simulated_seconds=n*.02,
    hashes={str(x):sha(x) for x in [tp,sp,pp,cp,ROOT/'source_snapshot_v2/run_actual_student_oracle.py']},checks=checks,
    history_result=f'All{n} pre-control flattened/named histories, previous actions and resulting actions match direct formulas bit exactly. Student control0 raw combined action retained; expert controls1 onward normalize actual applied MPC targets.',
    limits='No policy/physics replay, full integration interior decoding, final lifecycle, quiet-standing or label qualification performed. Frozen checkpoint only; live source file can progress independently.')
(OUT/'report.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(passed=True,controls=n,source_controls=result['source_controls'],physics=n*10,trace_sha256=sha(tp),checks=len(checks)),indent=2))
