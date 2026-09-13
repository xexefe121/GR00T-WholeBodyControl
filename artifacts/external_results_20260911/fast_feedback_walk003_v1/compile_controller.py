"""Compile a motion-specific feedback policy from the qualified saved plans."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
BASE=Path(__file__).resolve().parent;NEW=BASE.parent
sys.path.insert(0,str(NEW/'direct_target_width251_collection_v1/source_draft_v1'))
from collection_math import difference_function,committed_target
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def local(p):
    p=str(p).replace('\\','/')
    return Path('/mnt/'+p[0].lower()+p[2:] if sys.platform!='win32' and p[1:2]==':' else p)
def main():
    assert sys.platform!='win32' and np.__version__=='1.26.4'
    report_path=NEW/'direct_target_width251_collection_v1/actual_v1/results_v1/report.json'
    assert sha(report_path)=='7e650844a98a3e9d5462a1a8ff45e0cbdc6847323e26b82f5b2a5685c07cf316'
    report=read(report_path);assert report['passed'] is True
    expert=NEW/'direct_target_width512_expert_recovery_v2'
    trace_path=expert/'nominal/trace.npz'
    assert sha(trace_path)=='1e05044095fbee4d6391cae21afd282095faa48c8b55b14df6ec1d1f33fe7227'
    core=expert/'source_snapshot_v1/gear_sonic/utils/g1_true23_mjbatch_ilqr_core.py'
    difference=difference_function(core)
    limits=read(local('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1/contract.json'))['joint_limits']
    limits=np.asarray(limits)
    states=np.empty((1018,59));targets=np.empty((1018,23));gains=np.empty((1018,23,58))
    coverage=np.zeros(1018,np.int64)
    for item in report['plan_subjects']:
        path=local(item['path']);assert sha(path)==item['sha256']
        start=item['control'];count=min(5,1269-start);rows=slice(start-251,start-251+count)
        with np.load(path,allow_pickle=False) as z:
            states[rows]=z['nominal_states'][:count];targets[rows]=z['targets'][:count];gains[rows]=z['gains'][:count]
        coverage[rows]+=1
    assert np.all(coverage==1)
    with np.load(trace_path,allow_pickle=False) as z:
        errors=[]
        for i,control in enumerate(range(251,1269)):
            target,*_=committed_target(difference,states[i],targets[i],gains[i],z['qpos'][control],z['qvel'][control],limits)
            errors.append(float(np.max(np.abs(target-z['target'][control]))))
    assert max(errors)==0.,max(errors)
    assert np.isfinite(states).all() and np.isfinite(targets).all() and np.isfinite(gains).all()
    # A measured-state perturbation must actually affect the feedback command.
    changed=0
    for i in range(0,1018,17):
        state=states[i].copy();a=committed_target(difference,states[i],targets[i],gains[i],state[:30],state[30:],limits)[0]
        state[30+6]+=.01;b=committed_target(difference,states[i],targets[i],gains[i],state[:30],state[30:],limits)[0]
        changed+=int(np.any(a!=b))
    assert changed>0
    output=BASE/'controller.npz';assert not output.exists()
    np.savez_compressed(output,nominal_states=states,feedforward_targets=targets,feedback_gains=gains,control=np.arange(251,1269,dtype=np.int64))
    result=dict(kind='motion_specific_time_varying_state_feedback',compiled=True,controls=1018,
        source_collection_sha256=sha(report_path),source_trace_sha256=sha(trace_path),
        original_committed_target_max_error_rad=max(errors),perturbed_state_changes_command_cases=changed,
        controller_sha256=sha(output),core_path=str(core),core_sha256=sha(core),
        collection_math_sha256=report['source_sha256']['collection_math.py'],plan_subjects=report['plan_subjects'],
        offline_planning_required=True,arbitrary_live_motion_supported=False,model_calls=0,native_steps=0)
    with (BASE/'controller_manifest.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:result[k] for k in ('compiled','controls','controller_sha256','original_committed_target_max_error_rad','perturbed_state_changes_command_cases')}))
if __name__=='__main__':main()
