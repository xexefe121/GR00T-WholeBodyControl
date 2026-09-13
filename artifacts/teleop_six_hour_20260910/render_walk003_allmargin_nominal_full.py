"""Full recorded-pass walk003: fixed-world reference/physics, original29 markers."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from gear_sonic.scripts.evaluate_g1_true23_bfmzero import load_case_motion
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL,PHYSICS
from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model

ARCHIVE=Path('E:/codex-artifacts/sonic23_teleop_six_hour_20260910')
CASE=ARCHIVE/'bfm_online_intent_v2/walk003_allmargin_wsl_independent_replay_v1'
PRODUCER=ARCHIVE/'mjbatch_full_v1/walk003_v4_native323_allmargin_full_v1'
OUTPUT=ARCHIVE/'visual_walk003_allmargin_nominal_full_v1'
WIDTH,HEIGHT,TOP,BOTTOM=960,640,88,144
EVENT_STEP=15563


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def font(size):return ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',size)
def read(path):
    with np.load(path,allow_pickle=False) as z:return {key:z[key].copy() for key in z.files}


def grid(scene,lo,hi):
    # Same tested fixed-world grid construction as the PICO full-failure renderer.
    for axis in range(2):
        for value in np.arange(np.floor(lo[axis]*2)/2,np.ceil(hi[axis]*2)/2+.1,.5):
            a,b=np.r_[lo,.003],np.r_[hi,.003]
            a[axis]=b[axis]=value
            g=scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array([.38,.42,.47,1.]))
            mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_CAPSULE,.002,a,b)
            scene.ngeom+=1
    for end,color in (([1,0,.006],[1,.25,.2,1]),([0,1,.006],[.2,.8,.4,1])):
        g=scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array(color,float))
        mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_CAPSULE,.006,np.array([0,0,.006]),np.array(end,float))
        scene.ngeom+=1


def main():
    OUTPUT.mkdir(exist_ok=False)
    frames_dir=OUTPUT/'frames';frames_dir.mkdir()
    report=json.loads((CASE/'report.json').read_text())
    audit=json.loads((CASE/'recorded_source_audit_v2.json').read_text())
    standing=json.loads((CASE/'saved_evidence_and_final_second_review_v1.json').read_text())
    producer_report=json.loads((PRODUCER/'report.json').read_text())
    request=json.loads((PRODUCER/'request.json').read_text())
    assert audit['recorded_source_tracking_pass'] and all(audit['gates'].values())
    assert report['all_recorded_physics_bit_exact'] and report['failure'] is None
    assert report['completed_controls']==1569 and report['source_completed_controls']==819
    assert report['engine_warning_counts']==[0]*8 and report['range_excess_max']==0.
    assert sha(CASE/'trace.npz')==report['trace_sha256']==audit['trace_sha256']
    assert sha(CASE/'report.json')==audit['report_sha256']
    assert sha(PRODUCER/'trace.npz')==producer_report['trace_sha256']
    motion_path=ARCHIVE/'mjbatch_intent_floor_inputs_v1/walk003/reference.npz'
    motion,timeline,motion_path=load_case_motion('walk003',motion_path)
    assert sha(motion_path)==report['reference_sha256']==audit['reference_sha256']
    original_path=ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1/walk003/original29.npz'
    assert sha(original_path)==report['original29_sha256']
    original_tasks=read(original_path)['source_task_position_w']
    trace=read(CASE/'trace.npz')
    producer_trace=read(PRODUCER/'trace.npz')
    for key in ('physics_qpos','physics_qvel','physics_torque','qpos','qvel','target','source_frame','physics_substeps'):
        np.testing.assert_array_equal(trace[key],producer_trace[key])
    total_steps=len(trace['physics_torque']);assert total_steps==15690
    np.testing.assert_array_equal(trace['physics_substeps'],np.full(1569,10))
    np.testing.assert_array_equal(trace['qpos'],trace['physics_qpos'][::10])
    np.testing.assert_array_equal(trace['physics_warning_counts'],np.zeros((15691,8),dtype=int))
    assert abs(trace['physics_time'][EVENT_STEP]-31.126)<1e-8
    assert abs(trace['physics_qvel'][EVENT_STEP,6+10]-7.834165774114366)<1e-12
    # All regular25fps samples, exact speed event and exact final physics state.
    visual_steps=np.unique(np.r_[np.arange(0,total_steps+1,20),EVENT_STEP,total_steps]).astype(int)
    times=visual_steps*.002
    frame_indices=np.where(visual_steps==0,10,(visual_steps-1)//10+11)
    actual=trace['physics_qpos'][visual_steps]
    desired=np.c_[motion['body_pos_w'][frame_indices,0],motion['body_quat_w'][frame_indices,0],motion['joint_pos'][frame_indices]]
    source=next(p for p in timeline['phases'] if p['name']=='source_motion')
    source_start_s=source['control_start']*.02
    source_seconds=source['requested_controls']*.02
    assert source_start_s==7. and source_seconds==16.38 and times[-1]==31.38
    planning_p95_seconds=producer_report['planning_ms_p50_p95_max'][1]/1000
    inputs=[CASE/'report.json',CASE/'trace.npz',CASE/'provenance.json',CASE/'recorded_source_audit_v2.json',
        CASE/'saved_evidence_and_final_second_review_v1.json',PRODUCER/'report.json',PRODUCER/'request.json',
        PRODUCER/'trace.npz',motion_path,original_path,ROOT.parent/'GR00T-WholeBodyControl'/MODEL,ROOT/PHYSICS,
        Path(__file__),Path(__file__).with_name('render_pico_relativefoot_failed_full.py')]
    hashes={str(p):sha(p) for p in inputs}
    (OUTPUT/'renderer_snapshot.py').write_bytes(Path(__file__).read_bytes())
    _,model,_=prepare_true23_model(ROOT.parent/'GR00T-WholeBodyControl'/MODEL,ROOT/PHYSICS)
    model.vis.global_.offwidth,model.vis.global_.offheight=WIDTH,HEIGHT
    model.vis.headlight.ambient[:]=[.5,.5,.5]
    model.vis.headlight.diffuse[:]=[.8,.8,.8]
    data=mujoco.MjData(model)
    renderer=mujoco.Renderer(model,HEIGHT,WIDTH)
    opt=mujoco.MjvOption()
    # Full-world bound covers every actual50Hz state, every desired frame and
    # original29 markers; native geometry bounding spheres also included.
    visual_geom_ids=np.flatnonzero((model.geom_bodyid!=0)&(model.geom_rgba[:,3]>0))
    radii=model.geom_rbound[visual_geom_ids,None]
    xyz_lo=original_tasks.min(axis=(0,1))-.04
    xyz_hi=original_tasks.max(axis=(0,1))+.04
    bound_poses=np.r_[trace['qpos'],np.c_[motion['body_pos_w'][:,0],motion['body_quat_w'][:,0],motion['joint_pos']],actual]
    for pose in bound_poses:
        data.qpos[:]=pose;mujoco.mj_kinematics(model,data)
        centers=data.geom_xpos[visual_geom_ids]
        xyz_lo=np.minimum(xyz_lo,(centers-radii).min(axis=0))
        xyz_hi=np.maximum(xyz_hi,(centers+radii).max(axis=0))
    xyz_lo-=.08;xyz_hi+=.08
    lo,hi=xyz_lo[:2],xyz_hi[:2]
    camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:]=(xyz_lo+xyz_hi)/2
    camera.distance=max(3.7,float(np.max(hi-lo))*1.1)
    camera.azimuth,camera.elevation=135,-22
    corners=np.asarray([[x,y,z] for x in (xyz_lo[0],xyz_hi[0]) for y in (xyz_lo[1],xyz_hi[1]) for z in (xyz_lo[2],xyz_hi[2])])
    # Fit conservative full-world AABB into the actual MuJoCo GL camera frustum.
    for _ in range(80):
        renderer.update_scene(data,camera=camera,scene_option=opt)
        cam=renderer.scene.camera[0]
        forward=np.asarray(cam.forward);up=np.asarray(cam.up);right=np.cross(forward,up)
        offsets=corners-np.asarray(cam.pos)
        depth=offsets@forward
        vertical=(offsets@up)/depth
        horizontal=(offsets@right)/depth
        tangent=float(cam.frustum_top/cam.frustum_near)
        ratio=max(float(np.abs(vertical).max()/tangent),float(np.abs(horizontal).max()/(tangent*WIDTH/HEIGHT)))
        if np.all(depth>0) and ratio<=.93:break
        camera.distance*=1.025
    else:raise RuntimeError('Could not frame entire trajectory')
    f28,f23,f21=font(28),font(23),font(21)
    def picture(qpos,frame):
        data.qpos[:]=qpos;mujoco.mj_kinematics(model,data)
        renderer.update_scene(data,camera=camera,scene_option=opt);grid(renderer.scene,lo,hi)
        for point in original_tasks[frame]:
            g=renderer.scene.geoms[renderer.scene.ngeom]
            mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,np.full(3,.03),point,np.eye(3).ravel(),np.array([.1,.9,1.,.85]))
            renderer.scene.ngeom+=1
        return Image.fromarray(renderer.render().copy())
    def combined(slot):
        step,t,frame=int(visual_steps[slot]),times[slot],int(frame_indices[slot])
        canvas=Image.new('RGB',(WIDTH*2,TOP+HEIGHT+BOTTOM),'#121820')
        canvas.paste(picture(desired[slot],frame),(0,TOP))
        canvas.paste(picture(actual[slot],frame),(WIDTH,TOP))
        d=ImageDraw.Draw(canvas)
        d.text((18,10),'DECLARED v4 NATIVE23 REFERENCE',font=f28,fill='#5bbeff')
        d.text((18,49),'Explicit retarget + floor lift; original world frame',font=f23,fill='#c0ccd7')
        d.text((WIDTH+18,10),'PHYSICAL REPLAY: FULL RECORDED TRACKING PASS',font=f28,fill='#9ee7ad')
        d.text((WIDTH+18,49),'Pinned WSL native3.2.3; actual-target replay bit exact',font=f23,fill='#c0ccd7')
        d.line((WIDTH,TOP,WIDTH,TOP+HEIGHT),fill='#475569',width=2)
        control=max(0,(step-1)//10)
        phase=next((p['name'].replace('_',' ') for p in timeline['phases'] if p['control_start']<=control<p['control_stop']),'initial state')
        if step==0:phase='initial state'
        source_elapsed=np.clip(t-source_start_s,0,source_seconds)
        root=float(np.linalg.norm(actual[slot,:3]-desired[slot,:3]))
        maxspeed=float(np.abs(trace['physics_qvel'][step,6:]).max())
        d.text((18,TOP+HEIGHT+8),f'WALK003 | physics {t:.3f} /31.380s | source {source_elapsed:.3f} /16.380s | {phase}',font=f23,fill='white')
        d.text((18,TOP+HEIGHT+41),f'FULL819/819 source + entry/return | all15 recorded gates PASS | bounds0 / warnings0 | root {root:.3f}m | peak joint speed {maxspeed:.3f}rad/s',font=f21,fill='#a8eab6')
        event='EXACT31.126s: right ankle pitch +7.834rad/s (26.1% hardware limit)' if step==EVENT_STEP else 'Cyan: ORIGINAL29 world hand/head goals | fixed0.5m grid | no alignment or camera tracking'
        d.text((18,TOP+HEIGHT+75),event+' | Standing residual motion remains',font=f21,fill='#ffd49a')
        d.text((18,TOP+HEIGHT+109),f'OFFLINE | >=740ms packets / up to760ms raw poses | producer planning p95 {planning_p95_seconds:.2f}s per100ms | live MPC/hardware NOT qualified',font=f21,fill='#ffb6a4')
        return canvas
    contact_steps=[0,3500,8000,11680,EVENT_STEP,total_steps]
    assert all(step in visual_steps for step in contact_steps)
    kept={};concat=['ffconcat version 1.0']
    try:
        for slot,step in enumerate(visual_steps):
            frame=combined(slot);path=frames_dir/f'frame_{int(step):05d}.png';frame.save(path)
            if int(step) in contact_steps:kept[int(step)]=frame.copy()
            duration=times[slot+1]-times[slot] if slot+1<len(times) else .002
            concat.extend((f"file 'frames/{path.name}'",'option framerate 500',f'duration {duration:.6f}'))
            if slot%100==0:print(json.dumps(dict(rendered=slot,total=len(visual_steps),physics_seconds=float(times[slot]))),flush=True)
    finally:renderer.close()
    (OUTPUT/'frames.ffconcat').write_text('\n'.join(concat)+'\n')
    ffmpeg,ffprobe=shutil.which('ffmpeg'),shutil.which('ffprobe');assert ffmpeg and ffprobe
    video=OUTPUT/'full_source_and_lifecycle.fixed_world.mp4'
    subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-threads','1','-f','concat','-safe','0','-i',str(OUTPUT/'frames.ffconcat'),
        '-fps_mode','vfr','-enc_time_base','1:500','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-video_track_timescale','500',
        '-threads','1','-movflags','+faststart',str(video)],check=True)
    info=json.loads(subprocess.check_output([ffprobe,'-v','error','-select_streams','v:0','-show_frames',
        '-show_entries','frame=best_effort_timestamp_time','-of','json',str(video)]))
    video_times=np.asarray([float(f['best_effort_timestamp_time']) for f in info['frames']])
    np.testing.assert_allclose(video_times,times,atol=1e-6,rtol=0)
    sheet=Image.new('RGB',(1920,1308),'#121820')
    for slot,step in enumerate(contact_steps):
        sheet.paste(kept[step].resize((960,436),Image.Resampling.LANCZOS),((slot%2)*960,(slot//2)*436))
    sheet.save(OUTPUT/'contact_sheet.fixed_world.png')
    kept[EVENT_STEP].save(OUTPUT/'exact_ankle_speed_event.fixed_world.png')
    kept[total_steps].save(OUTPUT/'exact_final_standing_state.fixed_world.png')
    assert all(sha(path)==digest for path,digest in hashes.items()),'Input changed during render'
    receipt=dict(kind='full_walk003_recorded_pass_native323_vs_declared_v4_fixed_world',inputs=hashes,
        renderer_snapshot_sha256=sha(OUTPUT/'renderer_snapshot.py'),clip='walk003',
        recorded_controls=1569,recorded_physics_steps=total_steps,recorded_seconds=31.38,
        source_controls=819,source_controls_requested=819,source_seconds=source_seconds,initialization_seconds=7.,
        full_source_completed=True,full_lifecycle_completed=True,recorded_source_tracking_pass=True,
        physical_trace_bit_exact_to_producer=True,physics_reexecuted=False,
        source_control_hz=50,visual_sample_hz=25,visual_physics_step_indices=visual_steps.tolist(),
        visual_source_frame_indices=frame_indices.tolist(),speed_event_physics_step=EVENT_STEP,
        speed_event_timestamp_seconds=31.126,final_timestamp_seconds=31.38,
        desired_sample_convention='step0 uses frame10; each post-step physical state paired with active50Hz source command frame ((step-1)//10+11)',
        original29_world_hand_head_markers=True,markers_aligned_to_actual_root=False,
        camera_tracking=False,root_alignment=False,time_warp=False,pose_transforms=False,
        full_world_geometry_bound=dict(minimum=xyz_lo.tolist(),maximum=xyz_hi.tolist(),
            native_geometry_bounding_spheres=True,original_markers_included=True,conservative_projection_ratio=ratio),
        fixed_camera=dict(lookat=camera.lookat.tolist(),distance=camera.distance,azimuth=camera.azimuth,elevation=camera.elevation),
        declared_reference_packet_preview_seconds=.74,raw_pose_support_upper_bound_seconds=.76,
        offline_planning=True,producer_planning_p95_seconds=planning_p95_seconds,planning_block_budget_seconds=.1,
        standing_motion=standing['final_second'],settled_standing_claim=False,
        timing_qualified=False,live_mpc_qualified=False,hardware_authorized=False,
        video=str(video),video_sha256=sha(video),video_frames=len(video_times),
        maximum_timestamp_error_seconds=float(np.max(np.abs(video_times-times))),
        terminal_display_hold_seconds=.002,contact_sheet=str(OUTPUT/'contact_sheet.fixed_world.png'),contact_steps=contact_steps)
    (OUTPUT/'render_receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps({key:receipt[key] for key in ('video','video_frames','recorded_seconds','source_seconds','maximum_timestamp_error_seconds','contact_sheet')}),flush=True)


if __name__=='__main__':main()
