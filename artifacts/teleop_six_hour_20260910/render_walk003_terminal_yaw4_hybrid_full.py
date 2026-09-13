"""Full36.38s yaw4 hybrid, original lifecycle and separate settling extension."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import mujoco
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from artifacts.teleop_six_hour_20260910.render_walk003_allmargin_nominal_full import grid,font
from gear_sonic.scripts.evaluate_g1_true23_bfmzero import load_case_motion
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL,PHYSICS
from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model

E=Path('E:/codex-artifacts/sonic23_teleop_six_hour_20260910')
CASE=E/'bfm_online_intent_v2/walk003_terminal_bfm_yaw4_hybrid_v1'
EXTRA=CASE/'post_lifecycle_hold_5s'
OUT=E/'visual_walk003_terminal_bfm_yaw4_full_v1'
OLDVIS=E/'visual_walk003_allmargin_nominal_full_v1'
PRODUCER=E/'mjbatch_full_v1/walk003_v4_native323_allmargin_full_v1'
INDEPENDENT=Path('E:/codex_sonic_runtime/bfm_seed_20260910/hybrid_walk003_yaw4_history_quiet_audit_v1/report.json')
WIDTH,HEIGHT,TOP,BOTTOM=960,640,88,144
SWITCH,BOUNDARY,FINAL=12690,15690,18190


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def main():
    OUT.mkdir(exist_ok=False);frames_dir=OUT/'frames';frames_dir.mkdir()
    report=json.loads((CASE/'report.json').read_text())
    extra_report=json.loads((EXTRA/'report.json').read_text())
    audit=json.loads((CASE/'recorded_source_audit_v2.json').read_text())
    oldreceipt=json.loads((OLDVIS/'render_receipt.json').read_text())
    producer_report=json.loads((PRODUCER/'report.json').read_text())
    assert audit['recorded_source_tracking_pass'] and all(audit['gates'].values())
    for r,p in ((report,CASE),(extra_report,EXTRA)):
        assert r['quiet_standing_diagnostic']['quiet_standing_diagnostic_pass']
        assert all(r['quiet_standing_diagnostic']['gates'].values())
        assert r['failure'] is None and r['range_excess_max']==0 and r['engine_warning_counts']==[0]*8
        assert sha(p/'trace.npz')==r['trace_sha256']
    assert report['BFM_terminal_yaw_gain']==extra_report['BFM_terminal_yaw_gain']==4.
    assert report['completed_controls']==1569 and extra_report['completed_controls']==250
    assert sha(CASE/'report.json')==audit['report_sha256']
    motion_path=E/'mjbatch_intent_floor_inputs_v1/walk003/reference.npz'
    motion,timeline,motion_path=load_case_motion('walk003',motion_path)
    assert sha(motion_path)==audit['reference_sha256']
    original_path=ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1/walk003/original29.npz'
    assert sha(original_path)==audit['original_intent']['original29_sha256']
    tasks=read(original_path)['source_task_position_w']
    life,extension=read(CASE/'trace.npz'),read(EXTRA/'trace.npz')
    for key in ('qpos','qvel','physics_qpos','physics_qvel'):
        np.testing.assert_array_equal(extension[key][0],life[key][-1])
    for t in (life,extension):
        assert np.all(t['physics_substeps']==10)
        assert not np.any(t['physics_warning_counts'])
        np.testing.assert_array_equal(t['qpos'],t['physics_qpos'][::10])
    q=np.r_[life['physics_qpos'],extension['physics_qpos'][1:]]
    dq=np.r_[life['physics_qvel'],extension['physics_qvel'][1:]]
    physical_time=np.r_[life['physics_time'],extension['physics_time'][1:]]
    assert q.shape==(FINAL+1,30)
    np.testing.assert_allclose(physical_time,np.arange(FINAL+1)*.002,atol=1e-8,rtol=0)
    #15fps nominal sampling is rounded to actual2ms physical states; exact
    #switch, original-lifecycle boundary and final extension states added.
    regular=np.rint(np.arange(0,FINAL*.002,1/15)/.002).astype(int)
    steps=np.unique(np.r_[regular,SWITCH,BOUNDARY,FINAL]).astype(int)
    times=steps*.002
    frames=np.minimum(np.where(steps==0,10,(steps-1)//10+11),len(motion['joint_pos'])-1)
    actual=q[steps]
    desired=np.c_[motion['body_pos_w'][frames,0],motion['body_quat_w'][frames,0],motion['joint_pos'][frames]]
    source=next(p for p in timeline['phases'] if p['name']=='source_motion')
    assert source['requested_controls']==819
    inputs=[CASE/'report.json',CASE/'trace.npz',CASE/'provenance.json',CASE/'recorded_source_audit_v2.json',
        CASE/'comparison.json',CASE/'verified_switch_receipt.json',EXTRA/'report.json',EXTRA/'trace.npz',INDEPENDENT,
        OLDVIS/'render_receipt.json',OLDVIS/'renderer_snapshot.py',PRODUCER/'report.json',motion_path,original_path,
        Path(__file__),Path(__file__).with_name('render_walk003_allmargin_nominal_full.py'),
        ROOT.parent/'GR00T-WholeBodyControl'/MODEL,ROOT/PHYSICS]
    hashes={str(p):sha(p) for p in inputs}
    assert sha(Path(__file__).with_name('render_walk003_allmargin_nominal_full.py'))==sha(OLDVIS/'renderer_snapshot.py')
    (OUT/'renderer_snapshot.py').write_bytes(Path(__file__).read_bytes())
    _,model,_=prepare_true23_model(ROOT.parent/'GR00T-WholeBodyControl'/MODEL,ROOT/PHYSICS)
    model.vis.global_.offwidth,model.vis.global_.offheight=WIDTH,HEIGHT
    model.vis.headlight.ambient[:]=[.5,.5,.5];model.vis.headlight.diffuse[:]=[.8,.8,.8]
    data=mujoco.MjData(model);renderer=mujoco.Renderer(model,HEIGHT,WIDTH);option=mujoco.MjvOption()
    #Start with previous full-route fixed camera and bounds. Check all new
    #geometry/markers before rendering; enlarge only if necessary for visibility.
    low=np.asarray(oldreceipt['full_world_geometry_bound']['minimum'])
    high=np.asarray(oldreceipt['full_world_geometry_bound']['maximum'])
    ids=np.flatnonzero((model.geom_bodyid!=0)&(model.geom_rgba[:,3]>0));radii=model.geom_rbound[ids,None]
    for pose in np.r_[life['qpos'],extension['qpos'],actual,desired]:
        data.qpos[:]=pose;mujoco.mj_kinematics(model,data)
        low=np.minimum(low,(data.geom_xpos[ids]-radii).min(0)-.04)
        high=np.maximum(high,(data.geom_xpos[ids]+radii).max(0)+.04)
    low=np.minimum(low,tasks.min(axis=(0,1))-.04);high=np.maximum(high,tasks.max(axis=(0,1))+.04)
    camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
    oldcamera=oldreceipt['fixed_camera'];camera.lookat[:]=oldcamera['lookat']
    for k in ('distance','azimuth','elevation'):setattr(camera,k,oldcamera[k])
    corners=np.asarray([[x,y,z] for x in (low[0],high[0]) for y in (low[1],high[1]) for z in (low[2],high[2])])
    for _ in range(80):
        renderer.update_scene(data,camera=camera,scene_option=option);cam=renderer.scene.camera[0]
        forward=np.asarray(cam.forward);up=np.asarray(cam.up);right=np.cross(forward,up)
        offsets=corners-np.asarray(cam.pos);depth=offsets@forward
        tangent=float(cam.frustum_top/cam.frustum_near)
        ratio=max(float(np.abs((offsets@up)/depth).max()/tangent),float(np.abs((offsets@right)/depth).max()/(tangent*WIDTH/HEIGHT)))
        if np.all(depth>0) and ratio<=.93:break
        camera.distance*=1.025
    else:raise RuntimeError('Could not frame complete hybrid route')
    f28,f23,f21=font(28),font(23),font(21)
    def picture(pose,frame):
        data.qpos[:]=pose;mujoco.mj_kinematics(model,data)
        renderer.update_scene(data,camera=camera,scene_option=option);grid(renderer.scene,low[:2],high[:2])
        for point in tasks[frame]:
            g=renderer.scene.geoms[renderer.scene.ngeom]
            mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,np.full(3,.03),point,np.eye(3).ravel(),np.array([.1,.9,1.,.85]))
            renderer.scene.ngeom+=1
        return Image.fromarray(renderer.render().copy())
    def combined(slot):
        step,t,frame=int(steps[slot]),float(times[slot]),int(frames[slot])
        canvas=Image.new('RGB',(2*WIDTH,TOP+HEIGHT+BOTTOM),'#121820')
        canvas.paste(picture(desired[slot],frame),(0,TOP));canvas.paste(picture(actual[slot],frame),(WIDTH,TOP))
        d=ImageDraw.Draw(canvas)
        d.text((18,10),'DECLARED v4 NATIVE23 REFERENCE',font=f28,fill='#5bbeff')
        d.text((18,49),'Original29 world hand/head goals in cyan; fixed route',font=f23,fill='#c0ccd7')
        d.text((WIDTH+18,10),'HYBRID: FULL SOURCE + TERMINAL STANDING PASS',font=f28,fill='#9ee7ad')
        d.text((WIDTH+18,49),'MPC source/return -> native BFM standing, yaw gain4',font=f23,fill='#c0ccd7')
        d.line((WIDTH,TOP,WIDTH,TOP+HEIGHT),fill='#475569',width=2)
        if step>BOUNDARY:phase=f'SEPARATE post-lifecycle hold +{t-31.38:.3f}/5.000s'
        elif step==BOUNDARY:phase='ORIGINAL LIFECYCLE COMPLETE; separate5s hold begins'
        elif step==SWITCH:phase='SWITCH NOW at25.380s: next physics step uses BFM'
        else:
            control=max(0,(step-1)//10)
            phase=next((p['name'].replace('_',' ') for p in timeline['phases'] if p['control_start']<=control<p['control_stop']),'initial state')
        source_elapsed=float(np.clip(t-7.,0,16.38))
        d.text((18,TOP+HEIGHT+8),f'WALK003 | physics {t:.3f}/36.380s | source {source_elapsed:.3f}/16.380s | {phase}',font=f23,fill='white')
        speed=float(np.abs(dq[step,6:]).max());root=float(np.linalg.norm(actual[slot,:3]-desired[slot,:3]))
        controller='recorded MPC targets' if step<SWITCH else 'native BFM: position1 / yaw4 / horizon8'
        d.text((18,TOP+HEIGHT+41),f'819/819 source + all entry/return | bounds0 / warnings0 | {controller} | joint speed {speed:.5f}rad/s | root {root:.3f}m',font=f21,fill='#a8eab6')
        d.text((18,TOP+HEIGHT+75),'Original15 source gates PASS; final3s quiet-standing PASS in lifecycle AND separate extension | no reset/root forces/alignment',font=f21,fill='#ffd49a')
        d.text((18,TOP+HEIGHT+109),'OFFLINE | >=740ms packets / up to760ms raw poses | MPC planning p95 10.09s per100ms | live teleoperation/hardware NOT qualified',font=f21,fill='#ffb6a4')
        return canvas
    contact=[0,3500,8000,SWITCH,BOUNDARY,FINAL]
    assert all(s in steps for s in contact)
    kept={};concat=['ffconcat version 1.0']
    try:
        for slot,step in enumerate(steps):
            frame=combined(slot);path=frames_dir/f'frame_{int(step):05d}.png';frame.save(path)
            if int(step) in contact:kept[int(step)]=frame.copy()
            duration=times[slot+1]-times[slot] if slot+1<len(times) else .002
            concat.extend((f"file 'frames/{path.name}'",'option framerate 500',f'duration {duration:.6f}'))
            if slot%100==0:print(json.dumps(dict(rendered=slot,total=len(steps),seconds=float(times[slot]))),flush=True)
    finally:renderer.close()
    (OUT/'frames.ffconcat').write_text('\n'.join(concat)+'\n')
    ffmpeg,ffprobe=shutil.which('ffmpeg'),shutil.which('ffprobe');assert ffmpeg and ffprobe
    video=OUT/'full_lifecycle_and_separate_hold.fixed_world.mp4'
    subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-threads','1','-f','concat','-safe','0','-i',str(OUT/'frames.ffconcat'),
        '-fps_mode','vfr','-enc_time_base','1:500','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-video_track_timescale','500',
        '-threads','1','-movflags','+faststart',str(video)],check=True)
    info=json.loads(subprocess.check_output([ffprobe,'-v','error','-select_streams','v:0','-show_frames',
        '-show_entries','frame=best_effort_timestamp_time','-of','json',str(video)]))
    vt=np.asarray([float(f['best_effort_timestamp_time']) for f in info['frames']])
    np.testing.assert_allclose(vt,times,atol=1e-6,rtol=0)
    sheet=Image.new('RGB',(1920,1308),'#121820')
    for slot,step in enumerate(contact):sheet.paste(kept[step].resize((960,436),Image.Resampling.LANCZOS),((slot%2)*960,(slot//2)*436))
    sheet.save(OUT/'contact_sheet.fixed_world.png')
    for step,name in ((SWITCH,'exact_controller_switch'),(BOUNDARY,'exact_original_lifecycle_end'),(FINAL,'exact_separate_hold_end')):
        kept[step].save(OUT/(name+'.fixed_world.png'))
    assert all(sha(path)==digest for path,digest in hashes.items()),'Input changed during render'
    receipt=dict(kind='walk003_yaw4_hybrid_full_original_lifecycle_plus_separate_hold_visual',inputs=hashes,
        renderer_snapshot_sha256=sha(OUT/'renderer_snapshot.py'),source_controls=819,source_requested=819,
        original_lifecycle_controls=1569,separate_extension_controls=250,total_physics_steps=FINAL,total_seconds=36.38,
        original_lifecycle_seconds=31.38,separate_extension_seconds=5.,switch_seconds=25.38,
        recorded_source_tracking_pass=True,lifecycle_quiet_standing_pass=True,extension_quiet_standing_pass=True,
        numerical_physics_rerun=False,native_model_physics_changed=False,
        visual_sampling_nominal_hz=15,nominal_samples_rounded_to_native2ms=True,
        visual_physics_steps=steps.tolist(),visual_source_frames=frames.tolist(),exact_added_event_steps=[SWITCH,BOUNDARY,FINAL],
        extension_reference='hold final existing reference/markers; no fabricated or appended original-source frames',
        desired_sample_convention='active50Hz source sample at each post-step physical state; unchanged original sample count',
        original29_world_goal_markers=True,marker_root_alignment=False,camera_tracking=False,root_alignment=False,time_warp=False,
        fixed_camera=dict(lookat=camera.lookat.tolist(),distance=camera.distance,azimuth=camera.azimuth,elevation=camera.elevation),
        previous_camera_retained=bool(camera.distance==oldcamera['distance']),full_geometry_projection_ratio=ratio,
        full_geometry_bounds=dict(minimum=low.tolist(),maximum=high.tolist()),
        source_controller='offline recorded MPC targets through source+return; native BFM standing yawgain4 after25.38s',
        preview_packet_seconds=.74,raw_pose_support_upper_bound_seconds=.76,original_MPC_planning_p95_seconds=producer_report['planning_ms_p50_p95_max'][1]/1000,
        timing_qualified=False,live_teleoperation_qualified=False,hardware_authorized=False,
        video=str(video),video_sha256=sha(video),frames=len(vt),maximum_timestamp_error_seconds=float(np.max(np.abs(vt-times))),
        final_display_hold_seconds=.002,contact_steps=contact,contact_sheet=str(OUT/'contact_sheet.fixed_world.png'))
    (OUT/'render_receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:receipt[k] for k in ('video','frames','total_seconds','maximum_timestamp_error_seconds','contact_sheet')}),flush=True)


if __name__=='__main__':main()
