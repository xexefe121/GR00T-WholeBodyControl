"""Reconstruct causal controller memory from actual expert prefixes.

Teacher states/commands update memory only. This is dataset construction, not
an executed actor rollout. Integration snapshots require exact trace provenance.
"""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np,mujoco
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
if sys.platform!='win32':
    for p in ('/mnt/e/codex_sonic_runtime/bfm_seed_20260910/onnx_deps','/root/.venvs/g1_true23_mjlab/lib/python3.11/site-packages'):sys.path.append(p)
from gear_sonic.scripts.evaluate_g1_true23_causal_dynamics import load_case
from gear_sonic.utils.g1_true23_native_targets import NativeTargetController
from gear_sonic.utils.g1_true23_bfmzero_stream import Packet
from gear_sonic.utils.g1_true23_controller_state import wrapper_contract,SNAPSHOT_VERSION


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def archive(path):
    with np.load(path) as z:return {k:z[k].copy() for k in z.files}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--library',type=Path,required=True)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    base=Path('E:/codex-artifacts' if sys.platform=='win32' else '/mnt/e/codex-artifacts')/'sonic23_teleop_resume_20260911'
    fw=base/'onboard_factory_firmware_v1';bank=base/'causal_dynamics_v1/focused_walk002_task_closure_bank_v1'
    expert=archive(bank/'expert.npz');rows=expert['resets'];meta=json.loads((bank/'bank.json').read_text())
    sources=[('walk003',['fast_feedback_walk003_v1/baseline_v1/nominal/trace.npz','fast_feedback_walk003_v1/baseline_v1/post_lifecycle_hold_5s/trace.npz'],None),
        ('walk002',['walk002_terminal_bfm_hybrid_v1/trace.npz','walk002_terminal_bfm_hybrid_v1/post_lifecycle_hold_5s/trace.npz'],
         'walk002_hybrid_all_control_snapshots_independent_v1/control_snapshots.npz'),
        ('pico',['pico_full_control_lm_v1/trace.npz'],'pico_all_control_snapshots_independent_v1/control_snapshots.npz')]
    result={};filled=np.zeros(len(rows),bool);provenance=[];scratch=None;contract=None
    for clip,paths,snap_path in sources:
        ci=next(i for i,r in enumerate(meta['clips']) if r['name']==clip)
        model,c,motion,original,timeline=load_case(clip);scratch=mujoco.MjData(model)
        controller=NativeTargetController(fw/'native_target_ppo_v2/actor_00185.onnx',fw,c,model=model,tasks=meta['tasks'],
            standing_qpos=timeline['configured_standing_qpos'],now=-.22,native_preview_guard=True,native_standing_capture=True,
            native_preview_delay_substeps=6,native_preview_library=args.library)
        contract=wrapper_contract(controller)
        def receive(f,now):
            if f<=controller.receiver.gate.last_sequence:return
            accepted=controller.receive(Packet(0,f,f*.02,{k:v[f] for k,v in motion.items() if k!='fps'},f==len(motion['joint_pos'])-1),
                original['source_task_position_w'][f],original['source_task_quaternion_wxyz'][f],now)
            if not accepted:raise ValueError(controller.receiver.gate.fault)
        for f in range(11):receive(f,(f-11)*.02)
        snapshots=None if snap_path is None else archive(base/snap_path)
        source_hashes=[digest(base/path) for path in paths]
        if snapshots is not None and str(snapshots['original_trace_sha256'].item())!=source_hashes[0]:
            raise ValueError('Integration snapshot belongs to another teacher trace')
        indices=np.flatnonzero(rows[:,0]==ci);by_frame={}
        for index in indices:by_frame.setdefault(int(rows[index,1]),[]).append(index)
        control=0;exact_count=0
        for part,path in enumerate(paths):
            trace=archive(base/path)
            for t,target in enumerate(trace['target']):
                q,v=trace['qpos'][t],trace['qvel'][t];f=int(trace['source_frame'][t]);now=control*.02
                receive(f,now)
                matches=[i for i in by_frame.get(f,[]) if not filled[i] and
                    np.max(abs(rows[i,2:32]-q))<1e-9 and np.max(abs(rows[i,32:61]-v))<1e-9]
                previous=controller.previous_native
                values=dict(prior=controller.receiver.history.prior.copy(),history=controller.receiver.history.vector().copy(),
                    loco_history=controller.policy.history[1:].copy(),velocity=controller.velocity.copy(),
                    phase=np.array(controller.policy.phase),walking=np.array(controller.policy.walking),
                    last_applied=c['default_q'].copy() if previous is None else previous.copy(),
                    valid=np.array(controller.has_applied_target),motion_seen=np.array(controller.native_motion_seen),
                    stationary_since=np.array(np.nan if controller.standing_stationary_since is None else controller.standing_stationary_since),
                    logical_time=np.array(now),closure_alpha=np.array(controller.body_goal.alpha),
                    absolute_control=np.array(control),qpos=q.copy(),qvel=v.copy())
                x=controller.observation(q,v,now)
                if not controller.has_applied_target:values['loco_history']=controller.policy.history[:4].copy()
                values['features']=x
                if matches:
                    if part==0 and snapshots is not None:
                        if int(snapshots['control'][t])!=control:raise ValueError('Snapshot control mismatch')
                        np.testing.assert_allclose(snapshots['qpos'][t],q,rtol=0,atol=1e-12)
                        np.testing.assert_allclose(snapshots['qvel'][t],v,rtol=0,atol=1e-12)
                        state=snapshots['control_integration_before'][t]
                        warning=np.stack((snapshots['warning_lastinfo'][t],snapshots['warning_counts'][t]),axis=-1)
                        exact=True;exact_count+=len(matches)
                    else:
                        mujoco.mj_resetData(model,scratch);scratch.qpos[:]=q;scratch.qvel[:]=v;scratch.time=now
                        mujoco.mj_forward(model,scratch);scratch.qacc_warmstart[:]=0
                        state=np.empty(mujoco.mj_stateSize(model,mujoco.mjtState.mjSTATE_INTEGRATION))
                        mujoco.mj_getState(model,scratch,state,mujoco.mjtState.mjSTATE_INTEGRATION)
                        warning=np.zeros((8,2),dtype=np.int32);exact=False
                    values.update(integration=state,warning=warning,exact_integration=np.array(exact))
                    for key,value in values.items():
                        if key not in result:result[key]=np.empty((len(rows),*np.asarray(value).shape),dtype=np.asarray(value).dtype)
                        for index in matches:result[key][index]=value
                    filled[matches]=True
                controller.commit_applied(q,v,target);control+=1
        missing=indices[~filled[indices]]
        if len(missing):raise ValueError(f'{clip}: {len(missing)} expert rows have no matching physical prefix, first {missing[:5]}')
        provenance.append(dict(clip=clip,traces=paths,trace_sha256=source_hashes,integration_source=snap_path,rows=len(indices),exact_integration_rows=exact_count))
        print(json.dumps(provenance[-1]),flush=True)
    if not filled.all():raise ValueError('Incomplete controller memory bank')
    result['expert_rows']=rows
    np.savez_compressed(args.output/'memory.npz',**result)
    manifest=dict(snapshot_version=SNAPSHOT_VERSION,wrapper=contract,expert_sha256=digest(bank/'expert.npz'),
        memory_sha256=digest(args.output/'memory.npz'),sources=provenance,rows=len(rows),
        teacher_history_reconstruction_only=True,actor_proposals_applied=False,future_reference_frames=0,
        missing_solver_policy='identical fresh solver initialization in both arms; excluded from exact continuation',simulation_ready=False)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(finished=True,rows=len(rows))),flush=True)


if __name__=='__main__':main()
