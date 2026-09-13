"""Original-speed reference/baseline/candidate video from saved physics only."""
import argparse,json,shutil,subprocess,sys
from pathlib import Path
import mujoco,numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from gear_sonic.scripts.evaluate_g1_true23_causal_dynamics import load_case


def read_run(path):
    report=json.loads((path/'report.json').read_text())
    with np.load(path/'trace.npz',allow_pickle=False) as z:states=z['states'].copy()
    np.testing.assert_allclose(states[:,59],np.arange(len(states))*.002,atol=1e-9,rtol=0)
    return report,states


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--baseline',type=Path,nargs='+',required=True);ap.add_argument('--candidate',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--fps',type=int,choices=[10,25],default=25)
    ap.add_argument('--baseline-label',default='RETAINED ACTOR 185')
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    if len(a.baseline) not in (1,3):raise ValueError('Supply one baseline or all three retained repetitions')
    runs=[read_run(path) for path in a.baseline]+[read_run(a.candidate)]
    clip=runs[0][0]['clip']
    if any(report['clip']!=clip for report,_ in runs):raise ValueError('Comparison requires the same complete recording')
    model,_,motion,original,timeline=load_case(clip)
    desired=np.c_[motion['body_pos_w'][:,0],motion['body_quat_w'][:,0],motion['joint_pos']]
    requested=(int(timeline['total_requested_controls'])+1500)*10
    if any(len(states)>requested+1 for _,states in runs):raise ValueError('Unexpected longer comparison run')
    model.vis.global_.offwidth=640;model.vis.global_.offheight=480
    model.vis.quality.offsamples=0;model.vis.headlight.ambient[:]=.55;model.vis.headlight.diffuse[:]=.8
    renderer=mujoco.Renderer(model,height=480,width=640);data=mujoco.MjData(model)
    camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE;camera.azimuth=135;camera.elevation=-20
    font=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',23);small=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',19)
    frames=a.output/'frames';frames.mkdir();concat=['ffconcat version 1.0'];kept=[]
    steps=np.unique(np.r_[np.arange(0,requested,500//a.fps),requested])
    selected=set(np.linspace(0,len(steps)-1,8).round().astype(int))
    grid_rows=(len(runs)+3)//3
    height=grid_rows*650+28
    baseline_titles=[a.baseline_label] if len(a.baseline)==1 else [f'RETAINED ACTOR 185 / REPLAY {i}' for i in (1,2,3)]
    titles=['RECEIVED REFERENCE']+baseline_titles+['RESTORED-MEMORY TRAINING / SEED 0']
    try:
        for number,step in enumerate(steps):
            frame=min(10 if step==0 else (step-1)//10+11,len(desired)-1)
            target=desired[frame]
            poses=[target]+[states[step,:30] if step<len(states) else None for _,states in runs]
            visible=np.stack([p[:3] for p in poses if p is not None])
            camera.lookat[:]=visible.mean(0);camera.lookat[2]=.65
            spread=max(np.linalg.norm(p[:2]-target[:2]) for p in poses if p is not None)
            camera.distance=2.8+spread
            canvas=Image.new('RGB',(1920,height),'#111923');draw=ImageDraw.Draw(canvas)
            for column,(title,pose) in enumerate(zip(titles,poses)):
                x=(column%3)*640;y=(column//3)*650
                draw.text((x+14,y+8),title,font=font,fill='#99cdf5' if column==0 else '#dfebf7')
                if pose is None:
                    duration=(len(runs[column-1][1])-1)*.002
                    draw.text((x+34,y+206),f'PHYSICS STOPPED AT {duration:.3f} s',font=font,fill='#ff7777')
                    draw.text((x+34,y+246),'No continuation after failure.',font=small,fill='#dddddd')
                else:
                    data.qpos[:]=pose;mujoco.mj_kinematics(model,data);renderer.update_scene(data,camera=camera)
                    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
                    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=False
                    for position,color in zip(original['source_task_position_w'][frame],
                            ([.1,.8,1.,.8],[.7,.3,1.,.8],[1.,.7,.1,.8])):
                        if renderer.scene.ngeom>=renderer.scene.maxgeom:raise ValueError('Goal marker scene capacity')
                        mujoco.mjv_initGeom(renderer.scene.geoms[renderer.scene.ngeom],mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([.022,0.,0.]),position,np.eye(3).ravel(),np.array(color,dtype=np.float32))
                        renderer.scene.ngeom+=1
                    canvas.paste(Image.fromarray(renderer.render().copy()),(x,y+44))
                    if column:draw.text((x+14,y+532),f'Root XY error: {np.linalg.norm(pose[:2]-target[:2]):.3f} m',font=small,fill='white')
                if column:
                    report,states=runs[column-1]
                    verdict='TRACKING PASS' if report['source']['passed'] else 'TRACKING FAIL'
                    physical='full clock duration' if report['physical_complete'] else 'stopped early'
                    draw.text((x+14,y+563),f"{verdict} | {physical}",font=small,fill='#ffcf91')
                    draw.text((x+14,y+592),f"Control misses {report['controller_deadline_misses']} | late physics {report['plant_finishes_over2ms_late']}",font=small,fill='#bbbbbb')
                    packet=report.get('packet_report') or {};fault=packet.get('fault')
                    if fault and step*.002>=fault['time']:
                        draw.text((x+14,y+619),f"INPUT FAULT @ {fault['time']:.3f}s | only {packet['consumed']} packets consumed",font=small,fill='#ff7777')
                    else:draw.text((x+14,y+619),'Received-only input | original physical limits',font=small,fill='#bbbbbb')
            phase='motion' if step<=int(timeline['total_requested_controls'])*10 else '30-second standing hold'
            draw.text((14,532),f'{clip} | {step*.002:.2f} / {requested*.002:.2f} s | {phase}',font=small,fill='white')
            draw.text((14,563),'Original speed. Shared camera; no pose realignment.',font=small,fill='#bbbbbb')
            draw.text((14,592),'Dots: original left hand, right hand and head goals.',font=small,fill='#bbbbbb')
            if len(a.baseline)==3:
                x,y=1294,670
                notes=['ALL THREE RETAINED REPLAYS SHOWN','Same actor, same wrapper, independent clocks.',
                    'Replay 2 loses input at 0.442 s.','Its 58.34 s runtime is not motion completion.',
                    'Only 34 of 1428 input packets were consumed.','All tracking and standing-hold verdicts fail.',
                    'Trained seed-0 candidate is not selected.','Two-seed pilot: every arm passes 1/192 trials.',
                    'Standing period shown in full; no invented physics.']
                for i,note in enumerate(notes):draw.text((x,y+i*43),note,font=font if i==0 else small,fill='#ffcf91' if i in (0,2,3) else '#dddddd')
            draw.text((14,height-26),'Independent-clock replay; matched short trials decide training. Full-body teleoperation remains unqualified.',font=small,fill='#ffcf91')
            path=frames/f'{number:05d}.png';canvas.save(path,compress_level=1)
            concat.extend((f"file 'frames/{path.name}'",'option framerate 500'))
            duration=(steps[number+1]-step)*.002 if number+1<len(steps) else .002
            concat.append(f'duration {duration:.9f}')
            if number in selected:kept.append(canvas.copy())
            if number%250==0:print(json.dumps(dict(rendered=number,total=len(steps))),flush=True)
    finally:renderer.close()
    (a.output/'frames.ffconcat').write_text('\n'.join(concat)+'\n')
    video=a.output/'native23_controller_memory_comparison.mp4'
    subprocess.run([shutil.which('ffmpeg'),'-hide_banner','-loglevel','error','-threads','1','-f','concat','-safe','0',
        '-i',str(a.output/'frames.ffconcat'),'-fps_mode','vfr','-enc_time_base','1:500','-c:v','libx264','-crf','19',
        '-pix_fmt','yuv420p','-video_track_timescale','500','-threads','1','-movflags','+faststart',str(video)],check=True)
    probe=json.loads(subprocess.check_output([shutil.which('ffprobe'),'-v','error','-select_streams','v:0','-show_frames',
        '-show_entries','frame=best_effort_timestamp_time','-of','json',str(video)]))
    last=float(probe['frames'][-1]['best_effort_timestamp_time'])
    if abs(last-requested*.002)>.0021:raise ValueError('Video does not retain the complete requested timeline')
    # Keep eight full-width snapshots in a separate contact sheet.
    tile_height=height//2
    sheet=Image.new('RGB',(960,len(kept)*tile_height))
    for index,item in enumerate(kept):sheet.paste(item.resize((960,tile_height)),(0,index*tile_height))
    sheet.save(a.output/'contact_sheet.png')
    (a.output/'render.json').write_text(json.dumps(dict(video=str(video),timeline_seconds=requested*.002,
        last_frame_seconds=last,baseline=[str(path) for path in a.baseline],candidate=str(a.candidate),
        baseline_label=a.baseline_label,
        stopped_physics_is_never_extended=True,simulation_ready=False),indent=2)+'\n')
    print(str(video),flush=True)


if __name__=='__main__':main()
