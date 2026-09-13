"""Fixed-world saved MPC replay; preserve the final partial 2ms physics steps."""
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
from gear_sonic.scripts.evaluate_g1_true23_bfmzero import load_motion
from gear_sonic.utils.g1_true23_generalist_benchmark import MODEL,PHYSICS
from gear_sonic.utils.g1_true23_step1b_mujoco import prepare_true23_model

CASE=ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native323_replay_v1/walk002_source3_feedback_v2'
WIDTH,HEIGHT,TOP,BOTTOM=640,480,88,104


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
    output=CASE/'visual_comparison_v1'
    output.mkdir(exist_ok=False)
    frames_dir=output/'frames'
    frames_dir.mkdir()
    report=json.loads((CASE/'report.json').read_text())
    provenance=json.loads((CASE/'provenance.json').read_text())
    assert report['metric_revision']==2 and report['mode']=='feedback' and report['failure']
    assert sha(CASE/'trace.npz')==report['trace_sha256']
    assert sha(CASE/'provenance.json')==report['provenance_sha256']
    for path,digest in provenance['hashes'].items():
        local=Path(path)
        if not local.is_absolute():
            local=ROOT/local
        # The immutable runner snapshot binds loaded code if a live source was
        # subsequently edited. Data/model inputs must still match their hashes.
        if local.name=='evaluate_g1_true23_mjbatch_plan_replay.py':
            assert sha(CASE/'runner_snapshot.py')==digest
        else:
            assert sha(local)==digest,str(local)
    motion,timeline,motion_path=load_motion(report['clip'])
    with np.load(CASE/'trace.npz',allow_pickle=False) as z:
        trace={key:z[key].copy() for key in z.files}
    actual=trace['qpos']
    source_indices=np.r_[10,trace['source_frame']].astype(int)
    desired=np.column_stack((motion['body_pos_w'][source_indices,0],motion['body_quat_w'][source_indices,0],motion['joint_pos'][source_indices]))
    steps=np.r_[0,np.cumsum(trace['physics_substeps'])].astype(int)
    times=steps*.002
    np.testing.assert_array_equal(actual,trace['physics_qpos'][steps])
    np.testing.assert_allclose(actual[1:,:3]-desired[1:,:3],trace['root_error'],atol=1e-12,rtol=0)
    np.testing.assert_allclose(actual[1:,7:]-desired[1:,7:],trace['joint_error'],atol=1e-12,rtol=0)
    assert abs(times[-1]-report['simulated_seconds'])<1e-9
    assert np.all(trace['physics_substeps'][:-1]==10) and trace['physics_substeps'][-1]==7
    source=next(p for p in timeline['phases'] if p['name']=='source_motion')
    source_start_s=source['control_start']*.02
    assert source_start_s==7. and abs(times[-1]-source_start_s-1.174)<1e-12
    inputs=[CASE/'report.json',CASE/'provenance.json',CASE/'trace.npz',CASE/'runner_snapshot.py',motion_path,
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
    def picture(qpos):
        data.qpos[:]=qpos
        mujoco.mj_kinematics(model,data)
        renderer.update_scene(data,camera=camera,scene_option=opt)
        grid(renderer.scene,lo,hi)
        return Image.fromarray(renderer.render().copy())
    def combined(i):
        canvas=Image.new('RGB',(2*WIDTH,TOP+HEIGHT+BOTTOM),'#121820')
        canvas.paste(picture(desired[i]),(0,TOP));canvas.paste(picture(actual[i]),(WIDTH,TOP))
        draw=ImageDraw.Draw(canvas)
        draw.text((16,10),'REQUESTED NATIVE23 REFERENCE',font=f24,fill='#5bbeff')
        draw.text((16,47),'Original native poses; no root alignment',font=f20,fill='#c0ccd7')
        draw.text((WIDTH+16,10),'RECORDED MPC FEEDBACK REPLAY',font=f24,fill='#ffb65b')
        draw.text((WIDTH+16,47),'Physical MuJoCo 3.2.3 states; offline plan',font=f20,fill='#c0ccd7')
        draw.line((WIDTH,TOP,WIDTH,TOP+HEIGHT),fill='#475569',width=2)
        control=max(0,i-1)
        phase=next((p['name'].replace('_',' ') for p in timeline['phases'] if p['control_start']<=control<p['control_stop']),'initial state')
        source_time=max(0.,times[i]-source_start_s)
        drift=float(np.linalg.norm(actual[i,:3]-desired[i,:3]))
        leg=float(np.sqrt(np.mean((actual[i,7:19]-desired[i,7:19])**2)))
        draw.text((16,TOP+HEIGHT+8),f'WALK002 | physics {times[i]:.3f} s | source {source_time:.3f} / {source["requested_controls"]*.02:.3f} s | {phase}',font=f20,fill='white')
        draw.text((16,TOP+HEIGHT+37),f'PARTIAL RUN: limit failure at 8.174 s | root error {drift:.3f} m | leg RMSE {leg:.3f} rad',font=f20,fill='#ffb6a4')
        ending='FINAL CONTROL: 7 / 10 substeps; desired active sample at 8.180 s' if i==len(actual)-1 else 'Full 7.000 s initialization preserved | same fixed world camera | 0.5 m grid'
        draw.text((16,TOP+HEIGHT+69),ending,font=f18,fill='#c0ccd7')
        return canvas
    contact=[0,250,350,375,400,len(actual)-1]
    kept={}
    concat=['ffconcat version 1.0']
    try:
        for i in range(len(actual)):
            frame=combined(i)
            path=frames_dir/f'frame_{i:05d}.png'
            frame.save(path)
            if i in contact:
                kept[i]=frame.copy()
            duration=times[i+1]-times[i] if i+1<len(times) else .002
            concat.extend((f"file 'frames/{path.name}'",'option framerate 500',f'duration {duration:.6f}'))
            if i%100==0:
                print(json.dumps(dict(rendered=i,total=len(actual))),flush=True)
    finally:
        renderer.close()
    (output/'frames.ffconcat').write_text('\n'.join(concat)+'\n')
    video=output/'full_initialization_and_failed_source.fixed_world.mp4'
    ffmpeg=shutil.which('ffmpeg')
    ffprobe=shutil.which('ffprobe')
    assert ffmpeg and ffprobe
    subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-f','concat','-safe','0','-i',str(output/'frames.ffconcat'),
                    '-fps_mode','vfr','-enc_time_base','1:500','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',
                    '-video_track_timescale','500','-movflags','+faststart',str(video)],check=True)
    frame_info=json.loads(subprocess.check_output([ffprobe,'-v','error','-select_streams','v:0','-show_frames',
                                                  '-show_entries','frame=best_effort_timestamp_time,pkt_duration_time','-of','json',str(video)]))
    video_times=np.array([float(row['best_effort_timestamp_time']) for row in frame_info['frames']])
    np.testing.assert_allclose(video_times,times,atol=1e-6,rtol=0)
    subprocess.run([ffmpeg,'-v','error','-i',str(video),'-f','null','-'],check=True)
    sheet=Image.new('RGB',(1600,1260),'#121820')
    for slot,i in enumerate(contact):
        sheet.paste(kept[i].resize((800,420),Image.Resampling.LANCZOS),((slot%2)*800,(slot//2)*420))
    sheet.save(output/'contact_sheet.fixed_world.png')
    kept[contact[-1]].save(output/'failure_boundary.fixed_world.png')
    for path,digest in input_hashes.items():
        assert sha(path)==digest,path
    receipt=dict(kind='native323_failed_mpc_feedback_replay_vs_original_native_requested_reference',inputs=input_hashes,
                 source_plan=provenance['plan'],renderer_snapshot_sha256=sha(output/'renderer_snapshot.py'),
                 clip='walk002',recorded_controls=len(actual)-1,full_control_slots=int(np.sum(trace['physics_substeps']==10)),
                 final_control_substeps=7,recorded_physics_steps=int(steps[-1]),recorded_seconds=float(times[-1]),
                 initialization_seconds=source_start_s,actual_source_seconds=float(times[-1]-source_start_s),
                 full_requested_source_seconds=source['requested_controls']*.02,
                 source_pose_indices=source_indices.tolist(),physical_step_indices=steps.tolist(),
                 final_desired_pose_convention='active 50Hz command sample419 at nominal8.180s; actual terminal boundary8.174s',
                 fixed_camera=dict(lookat=camera.lookat.tolist(),distance=camera.distance,azimuth=camera.azimuth,elevation=camera.elevation),
                 camera_tracking=False,root_alignment=False,time_warp=False,pose_transforms=False,
                 physics_reexecuted=False,partial_failure_visible=True,hardware_authorized=False,visual_pass_claim=False,
                 video=str(video),video_sha256=sha(video),video_frames=len(video_times),variable_frame_timestamps_verified=True,
                 maximum_timestamp_error_s=float(np.max(np.abs(video_times-times))),final_frame_duration_s=.002,
                 video_padding_after_physical_stop_s=.002,full_video_decoded=True,
                 contact_sheet=str(output/'contact_sheet.fixed_world.png'),contact_indices=contact)
    (output/'render_receipt.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps({k:receipt[k] for k in ('video','video_frames','recorded_seconds','actual_source_seconds','maximum_timestamp_error_s','contact_sheet')}),flush=True)


if __name__=='__main__':
    main()
