"""One saved native23 state/seed: exact public-contract comparison and CPU1 timing.

No iLQR solve, full controller rollout, inference, or training. Each rollout is only
the already prescribed H30 seed-scoring operation. Shared sources remain untouched.
"""
from pathlib import Path
import sys
import os
import time
import json
import hashlib
import copy
import gc
import platform

HERE=Path(__file__).resolve().parent
OUTPUT=HERE/'validated'
sys.path.insert(0,str(HERE/'source_snapshot'))
import mujoco
import numpy as np
from single_lane_tracker import SingleLaneNative23Tracker
from gear_sonic.utils.g1_true23_mjbatch_mpc import Native23Tracker,load_native_bundle,load_motion_override
from gear_sonic.utils.g1_true23_mjbatch_model import position_servo_copy

TASK=Path('/mnt/e/codex-artifacts/sonic23_teleop_six_hour_20260910')
BUNDLE=Path('/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1')
REFERENCE=TASK/'mjbatch_intent_floor_inputs_v1/pico/reference.npz'
FIXTURE=HERE.parent/'hard_feasibility_zero_feedback_retry_3740_v1/restoration_03805_zero_feedback.npz'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def exact(a,b,label):
    assert a.shape==b.shape and a.dtype==b.dtype,(label,'shape/dtype')
    assert a.tobytes()==b.tobytes(),(label,'not bit exact',float(np.nanmax(np.abs(a-b))))

def saved_line(planner):
    return {name:planner.line.bind(name).copy() for name in (
        'state','qpos','qvel','ctrl','qacc_warmstart','time','warning','qfrc_actuator','xpos','xquat')}

def pair_check(old,new,x,targets,name,gains=None):
    left=old.rollout(x,targets,gains); old_witness=copy.deepcopy(old.last_rollout_feasibility)
    right=new.rollout(x,targets,gains); new_witness=copy.deepcopy(new.last_rollout_feasibility)
    for label,a,b in zip(('states','targets','cost'),left,right):exact(a,b,name+'/'+label)
    assert old_witness==new_witness,(name,'feasibility witness changed')
    left_line,right_line=saved_line(old),saved_line(new)
    for key in left_line:exact(left_line[key],right_line[key],name+'/line/'+key)
    arrays={f'baseline_{k}':v for k,v in zip(('states','targets','cost'),left)}
    arrays.update({f'optimized_{k}':v for k,v in zip(('states','targets','cost'),right)})
    arrays.update({'line_'+k:v for k,v in right_line.items()})
    np.savez_compressed(OUTPUT/(name+'.npz'),**arrays)
    return dict(name=name,states_targets_cost_bitexact=True,fullintegration_and_all_bound_fields_bitexact=True,
                feasibility_witness_exact=True,feasibility=old_witness,
                costs=[float(v) if np.isfinite(v) else None for v in left[2]],
                output_sha256=sha(OUTPUT/(name+'.npz'))),left

def main():
    OUTPUT.mkdir(exist_ok=True)
    assert not list(OUTPUT.glob('*.npz')),'do not overwrite completed comparison evidence'
    assert mujoco.__version__=='3.2.3' and np.__version__=='1.26.4'
    native,c,original,timeline,manifest=load_native_bundle(BUNDLE,'pico')
    motion,override=load_motion_override(REFERENCE,BUNDLE,'pico',native,c,original,timeline,manifest)
    with np.load(FIXTURE,allow_pickle=False) as z:fixture={k:z[k].copy() for k in z.files}
    state=mujoco.MjData(native)
    mujoco.mj_setState(native,state,fixture['initial_integration'],mujoco.mjtState(int(fixture['integration_state_spec'])))
    x=np.r_[state.qpos,state.qvel]
    targets=fixture['targets'];assert targets.shape==(30,23)
    kp,kd,effort=[np.asarray(c[k]) for k in ('kp','kd','native_effort')]
    tests=[];timings={}
    for hard in (False,True):
        mode='hard' if hard else 'soft'
        servo=position_servo_copy(native,kp,kd,effort)
        options=dict(horizon=30,threads=1,all_joint_limit_margin=.05,all_joint_limit_weight=2000,
                     relative_foot_weight=400,hard_feasibility=hard)
        old=Native23Tracker(servo,c,motion,**options)
        new=SingleLaneNative23Tracker(servo,c,motion,**options)
        # Derived Batch fields must be bound before stepping to be populated;
        # binding a new derived field afterward returns its initialization buffer.
        saved_line(old);saved_line(new)
        old.window(3815);new.window(3815)
        check,seed=pair_check(old,new,x,targets,mode+'_positive_seed');tests.append(check)
        if hard:assert all(check['feasibility']['feasible'])
        # Guided path, including its public nine candidate lanes, stays unchanged.
        gains=(seed[0][:,0].copy(),np.zeros((30,23)),np.zeros((30,23,58)))
        before=new.single_lane_steps
        check,_=pair_check(old,new,x,targets,mode+'_guided_zero_gains',gains);tests.append(check)
        assert new.single_lane_steps==before and new.guided_advance_calls==30
        # Re-enter unguided after guided to test restored integration/binding state.
        check,_=pair_check(old,new,x,targets,mode+'_after_guided');tests.append(check)
        if hard:
            check,_=pair_check(old,new,x,fixture['original_warm_targets'],mode+'_infeasible_original_seed');tests.append(check)
            assert not any(check['feasibility']['feasible'])
            invalid=targets.copy();invalid[0,0]=np.nan
            check,_=pair_check(old,new,x,invalid,mode+'_nonfinite_target');tests.append(check)
            assert not any(check['feasibility']['feasible'])
        # Fixed five alternating paired repeats, no sweep. Include verification of
        # each timed result; collect only rollout duration, not comparison/I/O.
        elapsed={'baseline9_ms':[],'optimized1_ms':[]}
        for repeat in range(5):
            ordered=((old,'baseline9_ms'),(new,'optimized1_ms')) if repeat%2==0 else ((new,'optimized1_ms'),(old,'baseline9_ms'))
            values={}
            for planner,label in ordered:
                tick=time.perf_counter();value=planner.rollout(x,targets);duration=(time.perf_counter()-tick)*1000
                elapsed[label].append(duration);values[label]=value
            for label,a,b in zip(('states','targets','cost'),values['baseline9_ms'],values['optimized1_ms']):exact(a,b,mode+'/timed/'+label)
        b=float(np.median(elapsed['baseline9_ms']));o=float(np.median(elapsed['optimized1_ms']))
        timings[mode]=dict(**elapsed,baseline_median_ms=b,optimized_median_ms=o,
                           saved_median_ms=b-o,speedup=b/o,reduction_percent=100*(b-o)/b,
                           fixture='actual3805 certified30 targets, unchanged horizon/cost/model',
                           CPU_threads=1,shared_host_not_quiet=True)
        del old,new,servo;gc.collect()
    source_hashes={str(p.relative_to(HERE)):sha(p) for p in HERE.rglob('*.py')}
    modules={name:module.__file__ for name,module in sys.modules.items() if name.startswith('gear_sonic.') and getattr(module,'__file__',None)}
    assert all(str(HERE/'source_snapshot') in path for path in modules.values()),modules
    result=dict(kind='private_native23_one_lane_unguided_dynamics_contract_benchmark',
        mujoco=mujoco.__version__,numpy=np.__version__,python=sys.version,platform=platform.platform(),
        cpu_model=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),None),
        cpu_affinity=sorted(os.sched_getaffinity(0)),environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
        tests=tests,timing=timings,loaded_modules=modules,private_source_sha256=source_hashes,
        fixture_sha256=sha(FIXTURE),reference_sha256=sha(REFERENCE),model_manifest=manifest,
        retained_public_lanes=9,computed_unguided_dynamics_lanes=1,guided_dynamics_lanes=9,
        cost_arithmetic_remains_nine_lanes=True,fullsolver_calls=0,policy_inference_calls=0,
        physical_controller_trials=0,production_integration=False,realtime_qualification=False)
    (HERE/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(tests=len(tests),all_comparisons_bitexact=True,timing=timings)),flush=True)

if __name__=='__main__':main()
