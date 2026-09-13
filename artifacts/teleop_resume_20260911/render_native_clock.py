"""Render the complete saved independent-clock trajectory; never step physics."""
import argparse,json,shutil,subprocess
from pathlib import Path
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,help='Optional fresh output directory for a separately captioned render')
    kind=p.add_mutually_exclusive_group()
    kind.add_argument('--causal-trace',action='store_true',help='Render a saved learned evaluation with its actual gate verdict.')
    kind.add_argument('--learned-clock',action='store_true',help='Render a learned controller run with independently advancing physics.')
    kind.add_argument('--factory-trace',action='store_true',help='Render a saved unpaced received-factory rollout, with timing limitation explicit.')
    p.add_argument('--fps',type=int,choices=[10,25,50],default=10)
    p.add_argument('--fast',action='store_true',help='Lower raster resolution and disable shadows; keep the complete original-speed physical trajectory')
    a=p.parse_args()
    learned=a.causal_trace or a.learned_clock or a.factory_trace
    out=a.output or a.run/('video_fast' if a.fast else 'video');out.mkdir(exist_ok=False)
    report=json.loads((a.run/'report.json').read_text())
    factory=bool(a.factory_trace or report.get('factory_locomotion') or report.get('native_factory_conditioned') or report.get('locomotion_conditioned'))
    standing=bool(report.get('standing_diagnostic'))
    with np.load(a.run/'trace.npz') as z:
        if a.factory_trace:
            # This is the exact initial-state archive used by run_factory_pose_sim.
            bank=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911/causal_dynamics_v1/bank')
            with np.load(bank/(report['clip']+'.npz')) as initial:
                first=initial['states'][10].copy()
            physical=np.c_[z['physics_qpos'],z['physics_qvel']]
            states=np.c_[np.r_[first[None],physical],np.arange(len(physical)+1)*.002]
        else:
            states=(np.c_[z['physics_qpos'],z['physics_qvel'],z['physics_time']]
                    if a.causal_trace else z['states'].copy())
    clip=report['clip'] if learned else 'walk003'
    if clip not in ('walk003','walk002','pico','walk008'):raise ValueError('unsupported reference clip')
    np.testing.assert_allclose(states[:,59],np.arange(len(states))*.002,atol=1e-9,rtol=0)
    reference=Path('E:/codex-artifacts/sonic23_teleop_six_hour_20260910/mjbatch_intent_floor_inputs_v1')/clip/'reference.npz'
    with np.load(reference) as z:desired=np.c_[z['body_pos_w'][:,0],z['body_quat_w'][:,0],z['joint_pos']]
    bundle=ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1'
    timeline=json.loads((bundle/clip/'timeline.json').read_text())
    lifecycle_steps=timeline['total_requested_controls']*10
    model=mujoco.MjModel.from_xml_path(str(bundle/'native_prepared.xml'))
    with np.load(bundle/'prepared_model_arrays.npz') as z:
        for k in z.files:getattr(model,k)[:]=z[k]
    mujoco.mj_setConst(model,mujoco.MjData(model))
    width,height=(320,240) if a.fast else (640,480)
    if a.fast:model.vis.quality.offsamples=0
    model.vis.global_.offwidth=width;model.vis.global_.offheight=height
    model.vis.headlight.ambient[:]=.55;model.vis.headlight.diffuse[:]=.8
    renderer=mujoco.Renderer(model,height=height,width=width);data=mujoco.MjData(model)
    camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance=2.8;camera.azimuth=135;camera.elevation=-20
    font=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',21)
    steps=np.unique(np.r_[np.arange(0,len(states),500//a.fps),lifecycle_steps,len(states)-1])
    steps=steps[steps<len(states)]
    contact_ids=set(np.linspace(0,len(steps)-1,8).round().astype(int));kept=[]
    frames=out/'frames';frames.mkdir();concat=['ffconcat version 1.0']
    try:
        for slot,step in enumerate(steps):
            frame=min(10 if step==0 else (step-1)//10+11,len(desired)-1)
            if standing:frame=11
            target=desired[frame];actual=states[step,:30]
            # One shared camera: no separate pose alignment or hidden root shift.
            camera.lookat[:]=(target[:3]+actual[:3])*.5;camera.lookat[2]=.65
            camera.distance=max(2.8,2.8+np.linalg.norm(target[:2]-actual[:2]))
            canvas=Image.new('RGB',(1280,640),'#121820');draw=ImageDraw.Draw(canvas)
            draw.text((12,8),'NATIVE23 REFERENCE',font=font,fill='#80c6ff')
            heading='LEARNED CONTROLLER — INDEPENDENT CLOCK' if a.learned_clock else 'LEARNED CONTROLLER — ACTUAL SIM'
            if factory:heading='FACTORY CONTROLLER — UNPACED SIM' if a.factory_trace else 'FACTORY CONTROLLER — INDEPENDENT CLOCK'
            if report.get('locomotion_conditioned') or report.get('policy') in ('native_locomotion_full_body_conditioned','native23_received_task_commands'):
                heading='LEARNED CONTROLLER — UNPACED SIM' if a.factory_trace else 'LEARNED CONTROLLER — INDEPENDENT CLOCK'
            draw.text((652,8),heading if learned else 'INDEPENDENT 500 Hz PHYSICS',font=font,fill='#a5edb4')
            for column,pose in enumerate((target,actual)):
                data.qpos[:]=pose;mujoco.mj_kinematics(model,data)
                renderer.update_scene(data,camera=camera)
                if a.fast:
                    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
                    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=False
                rendered=Image.fromarray(renderer.render().copy())
                if a.fast:rendered=rendered.resize((640,480))
                canvas.paste(rendered,(column*640,42))
            seconds=step*.002;phase='motion lifecycle' if step<=lifecycle_steps else 'continuous standing hold'
            if report.get('physical_complete') is False:phase='PARTIAL ROLLOUT — stopped before completion'
            if standing:phase='continuous standing diagnostic'
            draw.text((12,530),f'{seconds:.2f} / {(len(states)-1)*.002:.2f} s | {phase} | shared follow camera',font=font,fill='white')
            if learned:
                draw.text((12,562),f"XY error now: {np.linalg.norm(actual[:2]-target[:2]):.3f} m | original speed | saved physical trajectory",font=font,fill='#b4edc0')
                verdict='FAILED TRACKING — TELEOP NOT READY' if not report['source']['passed'] else 'TRACKING PASSED — other readiness gates still required'
                if standing:
                    verdict=('STANDING HOLD PASSED' if report['physical_complete'] and report['final_quiet']['passed'] else 'STANDING HOLD FAILED')+' — TELEOP NOT READY'
                timing=(f" | control misses: {report['controller_deadline_misses']} | late physics ticks: {report['plant_finishes_over2ms_late']}"
                        if a.learned_clock else ' | no independent timing qualification')
                draw.text((12,596),verdict+timing,font=font,fill='#ffd0a3')
            else:
                draw.text((12,562),f"Control misses: {report['controller_deadline_misses']} | physics ticks >2 ms late: {report['plant_steps_over2ms_late']} | XY error now: {np.linalg.norm(actual[:2]-target[:2]):.3f} m",font=font,fill='#b4edc0')
                draw.text((12,596),'PREPARED MOTION CONTROLLER | full-body live teleoperation still unqualified',font=font,fill='#ffd0a3')
            path=frames/f'{slot:05d}.png';canvas.save(path)
            duration=(steps[slot+1]-step)*.002 if slot+1<len(steps) else .002
            concat.extend((f"file 'frames/{path.name}'",'option framerate 500',f'duration {duration:.6f}'))
            if slot in contact_ids:kept.append(canvas.resize((640,320)))
            if slot%100==0:print(json.dumps(dict(rendered=slot,total=len(steps))),flush=True)
    finally:renderer.close()
    (out/'frames.ffconcat').write_text('\n'.join(concat)+'\n')
    video=out/(f'{clip}_learned_independent_clock.mp4' if a.learned_clock else
               f'{clip}_learned_evaluation.mp4' if a.causal_trace else 'walk003_full_motion_30s_hold.mp4')
    if factory:video=out/(f'{clip}_factory_received_unpaced.mp4' if a.factory_trace else f'{clip}_factory_received_independent_clock.mp4')
    if a.factory_trace and report.get('policy')=='native23_received_task_commands':
        video=out/f'{clip}_learned_received_unpaced.mp4'
    subprocess.run([shutil.which('ffmpeg'),'-hide_banner','-loglevel','error','-threads','1','-f','concat','-safe','0','-i',str(out/'frames.ffconcat'),'-fps_mode','vfr','-enc_time_base','1:500','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-video_track_timescale','500','-threads','1','-movflags','+faststart',str(video)],check=True)
    probe=json.loads(subprocess.check_output([shutil.which('ffprobe'),'-v','error','-select_streams','v:0','-show_frames','-show_entries','frame=best_effort_timestamp_time','-of','json',str(video)]))
    timestamps=np.array([float(f['best_effort_timestamp_time']) for f in probe['frames']])
    np.testing.assert_allclose(timestamps,steps*.002,atol=1e-6,rtol=0)
    sheet=Image.new('RGB',(1280,1280))
    for i,im in enumerate(kept):sheet.paste(im,((i%2)*640,(i//2)*320))
    sheet.save(out/'contact_sheet.png')
    (out/'render.json').write_text(json.dumps(dict(video=str(video),frames=len(steps),duration_s=(len(states)-1)*.002,
        new_dynamics=False,root_alignment=False,time_warp=False,shared_camera=True,
        independent_clock=a.learned_clock,render_size=[width,height],shadows=not a.fast,
        physical_complete=report.get('physical_complete'),
        requested_duration_s=report.get('requested_seconds',report.get('requested_controls',0)*.02),
        full_teleoperation_qualified=False),indent=2)+'\n')
    print(str(video),flush=True)

if __name__=='__main__':main()
