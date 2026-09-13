"""One fixed saved3805 fixture; private lane synchronization parity, no solves."""
from pathlib import Path
import sys,os,json,hashlib,copy,time,types,traceback
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'source_snapshot'))
import numpy as np
import mujoco
from single_lane_tracker import SingleLaneNative23Tracker
from gear_sonic.utils.g1_true23_mjbatch_mpc import Native23Tracker,load_native_bundle,load_motion_override
from gear_sonic.utils.g1_true23_mjbatch_model import position_servo_copy
OUT=HERE/'evidence';OUT.mkdir(exist_ok=False)
BASE=HERE.parent
FIXTURE=BASE/'hard_feasibility_zero_feedback_retry_3740_v1/restoration_03805_zero_feedback.npz'
GAINS=BASE/'restoration_zero_feedback_3805_v1/all_generated.npz'
BUNDLE=Path('/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1')
REFERENCE=Path('/mnt/e/codex-artifacts/sonic23_teleop_six_hour_20260910/mjbatch_intent_floor_inputs_v1/pico/reference.npz')
FIELDS=('state','qpos','qvel','ctrl','qacc_warmstart','time','warning','qfrc_actuator','xpos','xquat','qfrc_applied','xfrc_applied')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def saved(p):return {k:p.line.bind(k).copy() for k in FIELDS}

def exact(a,b,name):
    if a.shape!=b.shape or a.dtype!=b.dtype or a.tobytes()!=b.tobytes():
        d=np.abs(a-b)
        raise AssertionError(dict(name=name,shape_left=a.shape,shape_right=b.shape,maximum_difference=float(np.nanmax(d)),changed_values=int(np.count_nonzero(a!=b))))

def instrument(p):
    p.evidence=[]
    original=p._record_feasibility
    def record(self,*args,**kwargs):
        original(*args,**kwargs)
        self.evidence.append(dict(knot=self._rollout_knot,substep=kwargs['substep'],valid=self._rollout_valid.copy(),**saved(self)))
    p._record_feasibility=types.MethodType(record,p)

results=[]
def pair(old,new,x,targets,name,gains=None):
    old.evidence=[];new.evidence=[]
    start=time.perf_counter();left=old.rollout(x,targets,gains);told=(time.perf_counter()-start)*1000
    lw=copy.deepcopy(getattr(old,'last_rollout_feasibility',None));ls=saved(old)
    start=time.perf_counter();right=new.rollout(x,targets,gains);tnew=(time.perf_counter()-start)*1000
    rw=copy.deepcopy(getattr(new,'last_rollout_feasibility',None));rs=saved(new)
    arrays={}
    for prefix,values,line,steps in [('original',left,ls,old.evidence),('optimized',right,rs,new.evidence)]:
        arrays.update({prefix+'_'+k:v for k,v in zip(('states','targets','cost'),values)})
        arrays.update({prefix+'_final_'+k:v for k,v in line.items()})
        if steps:
            arrays.update({prefix+'_substep_'+k:np.asarray([r[k] for r in steps]) for k in steps[0]})
    np.savez_compressed(OUT/(name+'.npz'),**arrays)
    entry=dict(name=name,baseline_ms=told,optimized_ms=tnew,feasibility_baseline=lw,feasibility_optimized=rw,
        observed_mixed_valid_invalid=any(r['valid'].any() and not r['valid'].all() for r in old.evidence),
        observed_substep_records=len(old.evidence),output_sha256=sha(OUT/(name+'.npz')),
        parity_pass=False)
    results.append(entry)
    (OUT/'progress.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    for k,a,b in zip(('states','targets','cost'),left,right):exact(a,b,name+'/'+k)
    assert lw==rw,(name,'feasibility witness')
    for k in ls:exact(ls[k],rs[k],name+'/final/'+k)
    assert len(old.evidence)==len(new.evidence),(name,'substep record count')
    for i,(a,b) in enumerate(zip(old.evidence,new.evidence)):
        for k in a:exact(np.asarray(a[k]),np.asarray(b[k]),f'{name}/substep/{i}/{k}')
    entry['parity_pass']=True
    (OUT/'progress.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in entry.items() if not k.startswith('feasibility')}),flush=True)
    return left

def main():
    assert mujoco.__version__=='3.2.3' and np.__version__=='1.26.4'
    native,c,original,timeline,manifest=load_native_bundle(BUNDLE,'pico')
    motion,_=load_motion_override(REFERENCE,BUNDLE,'pico',native,c,original,timeline,manifest)
    z=np.load(FIXTURE);f={k:z[k].copy() for k in z.files};g=np.load(GAINS)
    data=mujoco.MjData(native);mujoco.mj_setState(native,data,f['initial_integration'],mujoco.mjtState(int(f['integration_state_spec'])))
    x=np.r_[data.qpos,data.qvel];targets=f['targets']
    k=g['backward_feedforward'][-1].copy();K=g['backward_feedback'][-1].copy()
    assert np.max(np.abs(k))>0 and np.max(np.abs(K))>0
    np.savez(OUT/'declared_inputs.npz',initial_integration=f['initial_integration'],targets=targets,feedforward=k,feedback=K)
    for hard in (False,True):
        servo=position_servo_copy(native,*[np.asarray(c[v]) for v in ('kp','kd','native_effort')])
        opts=dict(horizon=30,threads=1,all_joint_limit_margin=.05,all_joint_limit_weight=2000,relative_foot_weight=400,hard_feasibility=hard)
        old=Native23Tracker(servo,c,motion,**opts);new=SingleLaneNative23Tracker(servo,c,motion,**opts)
        saved(old);saved(new)
        if hard:instrument(old);instrument(new)
        old.window(3815);new.window(3815);mode='hard' if hard else 'soft'
        seed=pair(old,new,x,targets,mode+'_unguided_before')
        gain_tuple=(seed[0][:,0].copy(),k,K)
        before=new.single_lane_steps
        pair(old,new,x,targets,mode+'_guided_saved_nonzero',gain_tuple)
        assert new.single_lane_steps==before,'guided unexpectedly collapsed'
        pair(old,new,x,targets,mode+'_unguided_after')
    assert any(r['observed_mixed_valid_invalid'] for r in results if r['name']=='hard_guided_saved_nonzero'),'no mixed-lane witness obtained; no sweep authorized'
    modules={name:module.__file__ for name,module in sys.modules.items() if name.startswith('gear_sonic.') and getattr(module,'__file__',None)}
    assert all(str(HERE/'source_snapshot') in path for path in modules.values())
    result=dict(kind='private_single_lane_active_row_parity',passed=True,results=results,
        fixture_sha256=sha(FIXTURE),saved_backward_sha256=sha(GAINS),saved_backward_index=-1,
        feedforward_max_abs=float(np.max(np.abs(k))),feedback_max_abs=float(np.max(np.abs(K))),
        guidance_semantics='Saved nonzero final backward k/K used around same certified seed nominal states; declared parity stress, not reproduction of original restoration line search.',
        native_version=mujoco.__version__,numpy_version=np.__version__,cpu_threads=1,shared_host_not_quiet=True,
        timing_scope='One paired call per stage including identical audit hooks; no standalone performance qualification or speedup claim.',
        loaded_modules=modules,source_hashes={str(p.relative_to(HERE)):sha(p) for p in HERE.rglob('*.py')},
        fullsolver_calls=0,policy_calls=0,physical_controller_trials=0,production_integration=False,realtime_qualified=False)
    (HERE/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('ALL SIX PARITY COMPARISONS PASS',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        (OUT/'failure.json').write_text(json.dumps(dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc(),completed=results),indent=2)+'\n')
        raise
