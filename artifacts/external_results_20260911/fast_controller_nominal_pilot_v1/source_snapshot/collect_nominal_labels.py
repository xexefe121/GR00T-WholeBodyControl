"""Build one dataset from the already qualified walk003 actual hybrid history."""
import json
from pathlib import Path
import time
import mujoco
import numpy as np
from student_linear_runtime import *
from gear_sonic.utils.g1_true23_mjbatch_mpc import load_native_bundle,load_motion_override
from gear_sonic.utils.g1_true23_bfm_seed_observations import BFMHistory


def main():
    frozen=assert_frozen();dest=BASE/'labels';dest.mkdir(exist_ok=False)
    assert mujoco.__version__=='3.2.3'
    audit=json.loads((TEACHER/'recorded_source_audit_v2.json').read_text())
    assert audit['recorded_source_tracking_pass'] and len(audit['gates'])==15 and all(audit['gates'].values())
    assert audit['completed_controls']==1569 and audit['source_controls']==819
    assert sha(TEACHER/'trace.npz')==audit['trace_sha256']
    native,c,original,timeline,manifest=load_native_bundle(BUNDLE,'walk003')
    motion,_=load_motion_override(REFERENCE,BUNDLE,'walk003',native,c,original,timeline,manifest)
    original29=archive(BUNDLE/'walk003/original29.npz');teacher=archive(TEACHER/'trace.npz')
    switch=next(p['control_start'] for p in timeline['phases'] if p['name']=='returned_standing')
    assert switch==1269
    seed=Native23BFMRolloutSeed(native,c,original,ONNX,dependency_directory=DEPS,threads=1)
    features=LinearFeatures(motion,original29,c);history=BFMHistory();previous=np.zeros(23,np.float32)
    rows={k:[] for k in ('features','residual_rad','base_target','expert_target','base_action','previous_action','history','state','source_frame','control')}
    started=time.perf_counter()
    for control in range(switch):
        q,v=teacher['qpos'][control],teacher['qvel'][control]
        sensed,terms=seed._terms(q,v,previous);hist=history.before_update(terms)
        np.testing.assert_array_equal(previous,teacher['previous_action'][control])
        np.testing.assert_array_equal(hist,teacher['history'][control])
        np.testing.assert_array_equal(sensed,teacher['state'][control])
        raw,base,_=infer_base(seed,q,v,previous,hist,control+11)
        target=teacher['target'][control]
        x=features(q,v,control+11,base,previous)
        values=dict(features=x,residual_rad=target-base,base_target=base,expert_target=target,base_action=raw,
            previous_action=previous.copy(),history=hist,state=sensed,source_frame=control+11,control=control)
        for k,value in values.items():rows[k].append(value)
        previous=((target-np.asarray(c['default_q']))*np.asarray(c['kp'])/(.25*np.asarray(c['training_effort']))).astype(np.float32)
        np.testing.assert_array_equal(previous,teacher['action'][control])
    arrays={k:np.asarray(v) for k,v in rows.items()}
    span=np.diff(np.asarray(c['joint_limits']),axis=1).ravel().astype(np.float32)
    np.savez_compressed(dest/'labels.npz',**arrays,joint_span=span,joint_limits=np.asarray(c['joint_limits']),
        teacher_qpos=teacher['qpos'][:switch+1],teacher_qvel=teacher['qvel'][:switch+1])
    request=dict(kind=KIND,samples=switch,clip='walk003',teacher_actual_history_exact=True,
        teacher_actual_targets=True,teacher_preclip_MPC_targets_used=False,features=FEATURES,goal_offsets=OFFSETS.tolist(),
        frozen_BFM_goal='original native h8,pos1,yaw2',student_goal='declaredv4 native plus immutableoriginal29 tasks',
        network_clock_or_clip_inputs=False,received_goal_preview_seconds=.74,raw_pose_support_seconds=.76,
        target='ACTUALnativePDtarget minus BFMunclippedtarget on same actualteacherstate/history',
        learned_phase='all1269 entry/acquisition/source/return controls; no settledinitialhold claim',
        terminal='fixedoriginalBFMyaw4 after returned_standing event;300originalcontrols plusseparate250extension',
        elapsed_seconds=time.perf_counter()-started,labels_sha256=sha(dest/'labels.npz'),seed_identity=seed.identity(),
        input_hashes=frozen['input_sha256'],source_frozen_receipt_sha256=sha(BASE/'frozen_inputs.json'))
    (dest/'report.json').write_text(json.dumps(request,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:request[k] for k in ('samples','teacher_actual_history_exact','elapsed_seconds','labels_sha256')}),flush=True)


if __name__=='__main__':main()
