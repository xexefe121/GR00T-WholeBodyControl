"""Entire available PICO run through the exact recorded fall-threshold state."""
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
CASE=ARCHIVE/'mjbatch_full_v1/pico_v4_native323_relativefoot_full_v1'
CHECKPOINT=CASE/'trace.npz'
REVIEW=ARCHIVE/'bfm_online_intent_v2/pico_relativefoot_checkpoint_1749_review_v1.json'
GEOMETRY=ARCHIVE/'bfm_online_intent_v2/pico_relativefoot_failure_geometry_v1.json'
BASE_RENDERER=ROOT/'artifacts/teleop_six_hour_20260910/render_pico_relativefoot_checkpoint500.py'
WIDTH,HEIGHT,TOP,BOTTOM=640,480,88,132


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def font(size):
    return ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',size)


def grid(scene,lo,hi):
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
    output=ARCHIVE/'visual_pico_relativefoot_failed_full_v1'
    output.mkdir(exist_ok=False)
    frames_dir=output/'frames'
    frames_dir.mkdir()
    review=json.loads(REVIEW.read_text())
    report=json.loads((CASE/'report.json').read_text())
    request=json.loads((CASE/'request.json').read_text())
    assert sha(CHECKPOINT)==report['trace_sha256']
    assert sha(CASE/'trace.partial.npz')==review['trace_sha256']
    assert sha(CASE/'request.json')==review['request_sha256']
    assert report['request_sha256']==review['request_sha256']
    assert review['strict_physical_limits_pass'] and review['metadata']['completed_full_controls']==872
    assert report['failure']['kind']=='fall_or_nonfinite' and report['range_excess_max']==0.
    assert request['mujoco']=='3.2.3' and request['feedback_correction_clip_rad']==.1
    motion,timeline,motion_path=load_case_motion('pico',review['motion_override'])
    assert sha(motion_path)==review['motion_sha256']
    original_path=Path('C:/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/pico_freedancing_v1/optical_reference_v2/original29.npz')
    assert sha(original_path)==review['original29_sha256']
    with np.load(original_path,allow_pickle=False) as original_archive:
        original_tasks=original_archive['source_task_position_w'].copy()
    with np.load(CHECKPOINT,allow_pickle=False) as z:
        trace={key:z[key].copy() for key in z.files}
    with np.load(CASE/'trace.partial.npz',allow_pickle=False) as z:
        metadata=json.loads(z['checkpoint_metadata'].item())
        for key in ('qpos','qvel','target','source_frame','physics_substeps','physics_qpos','physics_qvel','physics_torque','root_error','joint_error'):
            np.testing.assert_array_equal(trace[key],z[key])
    assert all(metadata[key]==value for key,value in review['metadata'].items())
    assert metadata['failure']==report['failure']
    planning_ms=np.asarray([p['solve_ms'] for p in metadata['plans']])
    assert len(planning_ms)==175
    planning_p95_seconds=float(np.percentile(planning_ms,95)/1000)
    actual=trace['qpos']
    source_indices=np.r_[10,trace['source_frame']].astype(int)
    desired=np.column_stack((motion['body_pos_w'][source_indices,0],motion['body_quat_w'][source_indices,0],motion['joint_pos'][source_indices]))
    steps=np.r_[0,np.cumsum(trace['physics_substeps'])].astype(int)
    times=steps*.002
    np.testing.assert_array_equal(actual,trace['physics_qpos'][steps])
    np.testing.assert_allclose(actual[1:,:3]-desired[1:,:3],trace['root_error'],atol=1e-12,rtol=0)
    np.testing.assert_allclose(actual[1:,7:]-desired[1:,7:],trace['joint_error'],atol=1e-12,rtol=0)
    assert abs(times[-1]-metadata['simulation_time'])<1e-9
    assert len(trace['physics_substeps'])==873 and np.all(trace['physics_substeps'][:-1]==10) and trace['physics_substeps'][-1]==5
    assert steps[-1]==8725
    # Retain the exact penultimate and final2ms failure states, beyond the25fps grid.
    actual=np.insert(actual,len(actual)-1,trace['physics_qpos'][-2],axis=0)
    desired=np.insert(desired,len(desired)-1,desired[-1],axis=0)
    source_indices=np.insert(source_indices,len(source_indices)-1,source_indices[-1])
    steps=np.insert(steps,len(steps)-1,steps[-1]-1)
    times=steps*.002
    source=next(p for p in timeline['phases'] if p['name']=='source_motion')
    source_start_s=source['control_start']*.02
    assert source_start_s==7. and abs(times[-1]-source_start_s-10.45)<1e-12
    assert abs(times[-1]-times[-2]-.002)<1e-12
    inputs=[REVIEW,GEOMETRY,CHECKPOINT,CASE/'trace.partial.npz',CASE/'report.json',CASE/'request.json',motion_path,original_path,BASE_RENDERER,
            ROOT.parent/'GR00T-WholeBodyControl'/MODEL,ROOT/PHYSICS,Path(__file__)]
    input_hashes={str(p):sha(p) for p in inputs}
    (output/'renderer_snapshot.py').write_bytes(Path(__file__).read_bytes())
    _,model,_=prepare_true23_model(ROOT.parent/'GR00T-WholeBodyControl'/MODEL,ROOT/PHYSICS)
    model.vis.global_.offwidth,model.vis.global_.offheight=WIDTH,HEIGHT
    model.vis.headlight.ambient[:]=[.5,.5,.5]
    model.vis.headlight.diffuse[:]=[.8,.8,.8]
    data=mujoco.MjData(model)
    xy=np.concatenate((actual[:,:2],desired[:,:2]))
    lo,hi=xy.min(0)-.7,xy.max(0)+.7
    camera=mujoco.MjvCamera()
    camera.type=mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:]=[*((lo+hi)/2),.55]
    camera.distance=max(3.7,float(np.max(hi-lo))*1.35)
    camera.azimuth,camera.elevation=135,-18
    renderer=mujoco.Renderer(model,HEIGHT,WIDTH)
    opt=mujoco.MjvOption()
    f24,f20,f18=font(24),font(20),font(18)
    def picture(qpos,frame_index):
        data.qpos[:]=qpos
        mujoco.mj_kinematics(model,data)
        renderer.update_scene(data,camera=camera,scene_option=opt)
        grid(renderer.scene,lo,hi)
        # Reference landmarks stay in their original world coordinates in BOTH panes.
        # They are not translated onto the robot or used to align the camera.
        for point in original_tasks[frame_index]:
            g=renderer.scene.geoms[renderer.scene.ngeom]
            mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.023,.023,.023]),point,np.eye(3).ravel(),np.array([.1,.9,1.,.8]))
            renderer.scene.ngeom+=1
        return Image.fromarray(renderer.render().copy())
    def combined(i):
        canvas=Image.new('RGB',(2*WIDTH,TOP+HEIGHT+BOTTOM),'#121820')
        canvas.paste(picture(desired[i],source_indices[i]),(0,TOP));canvas.paste(picture(actual[i],source_indices[i]),(WIDTH,TOP))
        draw=ImageDraw.Draw(canvas)
        draw.text((16,10),'DECLARED v4 NATIVE23 REFERENCE',font=f24,fill='#5bbeff')
        draw.text((16,47),'Explicit retarget + floor lift; same world frame',font=f20,fill='#c0ccd7')
        draw.text((WIDTH+16,10),'PHYSICAL MPC: COMPLETE FAILED TRACE',font=f24,fill='#ffb65b')
        draw.text((WIDTH+16,47),'Native3.2.3; H30; relative-foot cost; feedback +/-0.1',font=f20,fill='#c0ccd7')
        draw.line((WIDTH,TOP,WIDTH,TOP+HEIGHT),fill='#475569',width=2)
        control=max(0,int(source_indices[i])-11)
        phase=next((p['name'].replace('_',' ') for p in timeline['phases'] if p['control_start']<=control<p['control_stop']),'initial state')
        source_time=max(0.,times[i]-source_start_s)
        drift=float(np.linalg.norm(actual[i,:3]-desired[i,:3]))
        leg=float(np.sqrt(np.mean((actual[i,7:19]-desired[i,7:19])**2)))
        draw.text((16,TOP+HEIGHT+8),f'PICO | physics {times[i]:.3f} s | source {source_time:.3f} / {source["requested_controls"]*.02:.3f} s | {phase}',font=f20,fill='white')
        draw.text((16,TOP+HEIGHT+37),f'FAIL at source10.450s | incomplete522 full+1 partial /5780 | hard bounds0 | root {drift:.3f}m / legs {leg:.3f}rad',font=f18,fill='#ffb6a4')
        ending='FALL REJECT: tilt1.200083rad; height0.394862m; exact final5/10 substeps | source/return unfinished' if i==len(actual)-1 else 'Full7s entry preserved | cyan spheres: ORIGINAL29 world hand/head goals | fixed0.5m grid'
        draw.text((16,TOP+HEIGHT+69),ending,font=f18,fill='#c0ccd7')
        draw.text((16,TOP+HEIGHT+99),f'OFFLINE | >=740ms packets / up to760ms raw poses | planning p95 {planning_p95_seconds:.2f}s per100ms | no full-source pass',font=f18,fill='#ffb6a4')
        return canvas
    contact=[0,350,500,700,820,len(actual)-1]
    visual_indices=np.unique(np.r_[np.arange(0,len(actual),2,dtype=int),len(actual)-2,len(actual)-1])
    assert visual_indices[-1]==874 and steps[visual_indices[-2]]==8724 and steps[visual_indices[-1]]==8725
    kept={}
    concat=['ffconcat version 1.0']
    try:
        for visual_slot,i in enumerate(visual_indices):
            frame=combined(i)
            path=frames_dir/f'frame_{i:05d}.png'
            frame.save(path)
            if i in contact:
                kept[i]=frame.copy()
            duration=times[visual_indices[visual_slot+1]]-times[i] if visual_slot+1<len(visual_indices) else .002
            concat.extend((f"file 'frames/{path.name}'",'option framerate 500',f'duration {duration:.6f}'))
            if i%100==0:
                print(json.dumps(dict(rendered=int(i),total=len(actual))),flush=True)
    finally:
        renderer.close()
    (output/'frames.ffconcat').write_text('\n'.join(concat)+'\n')
    video=output/'full_available_failed_source.fixed_world.mp4'
    ffmpeg=shutil.which('ffmpeg')
    ffprobe=shutil.which('ffprobe')
    assert ffmpeg and ffprobe
    subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-f','concat','-safe','0','-i',str(output/'frames.ffconcat'),
                    '-fps_mode','vfr','-enc_time_base','1:500','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',
                    '-video_track_timescale','500','-threads','1','-movflags','+faststart',str(video)],check=True)
    frame_info=json.loads(subprocess.check_output([ffprobe,'-v','error','-select_streams','v:0','-show_frames',
                                                  '-show_entries','frame=best_effort_timestamp_time,pkt_duration_time','-of','json',str(video)]))
    video_times=np.array([float(row['best_effort_timestamp_time']) for row in frame_info['frames']])
    np.testing.assert_allclose(video_times,times[visual_indices],atol=1e-6,rtol=0)
    subprocess.run([ffmpeg,'-v','error','-i',str(video),'-fps_mode','passthrough','-enc_time_base','1:500','-f','null','-'],check=True)
    sheet=Image.new('RGB',(1600,1314),'#121820')
    for slot,i in enumerate(contact):
        sheet.paste(kept[i].resize((800,438),Image.Resampling.LANCZOS),((slot%2)*800,(slot//2)*438))
    sheet.save(output/'contact_sheet.fixed_world.png')
    kept[contact[-1]].save(output/'exact_final_failure_state.fixed_world.png')
    for path,digest in input_hashes.items():
        assert sha(path)==digest,path
    receipt=dict(kind='native323_PICO_relative_foot_MPC_entire_failed_trace_vs_declared_v4_reference',inputs=input_hashes,
                 source_request=str(CASE/'request.json'),independent_checkpoint_review=str(REVIEW),renderer_snapshot_sha256=sha(output/'renderer_snapshot.py'),
                 clip='pico',recorded_controls=len(trace['physics_substeps']),full_control_slots=int(np.sum(trace['physics_substeps']==10)),
                 final_control_substeps=5,recorded_physics_steps=int(steps[-1]),recorded_seconds=float(times[-1]),
                 initialization_seconds=source_start_s,actual_source_seconds=float(times[-1]-source_start_s),
                 full_requested_source_seconds=source['requested_controls']*.02,source_control_slots_shown=523,source_full_controls_shown=522,
                 source_partial_control_substeps=5,source_controls_requested=5780,failure=report['failure'],actual_joint_bound_excess_max_rad=0.,
                 original29_world_hand_head_markers=True,markers_aligned_to_actual_root=False,
                 visual_sample_rate_hz=25,visual_pose_row_indices=visual_indices.tolist(),visual_physics_step_indices=steps[visual_indices].tolist(),source_control_rate_unchanged_hz=50,
                 declared_reference_packet_preview_seconds=.74,minimum_reference_preview_seconds=.74,raw_pose_support_upper_bound_seconds=.76,
                 offline_planning=True,planning_ms_p50_p95_max=np.percentile(planning_ms,[50,95,100]).tolist(),planning_block_budget_ms=100.,
                 source_pose_indices=source_indices.tolist(),physical_step_indices=steps.tolist(),
                 final_desired_pose_convention='active50Hz command sample883; exact physical steps8724 and8725 at17.448 and17.450seconds explicitly included',
                 fixed_camera=dict(lookat=camera.lookat.tolist(),distance=camera.distance,azimuth=camera.azimuth,elevation=camera.elevation),
                 camera_tracking=False,root_alignment=False,time_warp=False,pose_transforms=False,
                 physics_reexecuted=False,partial_scope_visible=True,hardware_authorized=False,visual_pass_claim=False,
                 feedback_correction_clip_rad=.1,full_source_completed=False,received_stream_controller=False,
                 video=str(video),video_sha256=sha(video),video_frames=len(video_times),variable_frame_timestamps_verified=True,
                 maximum_timestamp_error_s=float(np.max(np.abs(video_times-times[visual_indices]))),final_frame_duration_s=.002,
                 video_padding_after_physical_stop_s=.002,full_video_decoded=True,
                 contact_sheet=str(output/'contact_sheet.fixed_world.png'),contact_indices=contact)
    (output/'render_receipt.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps({k:receipt[k] for k in ('video','video_frames','recorded_seconds','actual_source_seconds','maximum_timestamp_error_s','contact_sheet')}),flush=True)


if __name__=='__main__':
    main()
