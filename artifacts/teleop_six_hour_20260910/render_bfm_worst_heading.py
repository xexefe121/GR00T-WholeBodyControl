"""Render the full PICO noisy run's worst heading mismatch, including context."""
from pathlib import Path
import json
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image,ImageDraw
from render_bfm_comparison import ROOT,MODEL,PHYSICS,prepare_true23_model,add_grid,font,sha
from gear_sonic.scripts.evaluate_g1_true23_bfmzero import load_motion


def yaw(q):
    w,x,y,z=q.T
    return np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))


def run():
    case=Path(__file__).resolve().parent/'bfm_observable_closed_loop_v1/pico_fixed_bias_noise'
    output=case/'visual_comparison_v1'
    report=json.loads((case/'report.json').read_text())
    actual=np.load(case/'trace.npz')['qpos'];motion,timeline,path=load_motion('pico')
    desired=np.c_[motion['body_pos_w'][10:10+len(actual),0],motion['body_quat_w'][10:10+len(actual),0],motion['joint_pos'][10:10+len(actual)]]
    phase=next(p for p in timeline['phases'] if p['name']=='source_motion')
    start,stop=phase['control_start']+1,phase['control_stop']+1
    angles=np.abs(np.angle(np.exp(1j*(yaw(actual[:,3:7])-yaw(desired[:,3:7])))))
    worst=start+int(np.argmax(angles[start:stop]));begin=max(start,worst-100);end=min(stop-1,worst+100)
    _,model,_=prepare_true23_model(ROOT.parent/'GR00T-WholeBodyControl'/MODEL,ROOT/PHYSICS)
    width,height,top,bottom=640,480,86,74
    model.vis.global_.offwidth=width;model.vis.global_.offheight=height
    model.vis.headlight.ambient[:]=[.5]*3;model.vis.headlight.diffuse[:]=[.8]*3;model.vis.headlight.specular[:]=[.1]*3
    data=mujoco.MjData(model);renderer=mujoco.Renderer(model,height,width)
    all_xy=np.r_[actual[:,:2],desired[:,:2]];lo,hi=all_xy.min(0)-.8,all_xy.max(0)+.8
    camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:]=[*((lo+hi)/2),.65];camera.distance=max(4.3,float(np.max(hi-lo))*1.4)
    camera.azimuth=135;camera.elevation=-20
    options=mujoco.MjvOption();options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT]=False
    f24,f20,f18=font(24),font(20),font(18)
    def pose(q):
        data.qpos[:]=q;mujoco.mj_forward(model,data)
        renderer.update_scene(data,camera=camera,scene_option=options);add_grid(renderer.scene,(lo,hi))
        return Image.fromarray(renderer.render().copy())
    def frame(i):
        canvas=Image.new('RGB',(2*width,top+height+bottom),'#121820')
        canvas.paste(pose(desired[i]),(0,top));canvas.paste(pose(actual[i]),(width,top))
        draw=ImageDraw.Draw(canvas)
        draw.text((20,12),'DESIRED NATIVE23 REFERENCE',fill=(91,190,255),font=f24)
        draw.text((width+20,12),'ACTUAL MUJOCO SIMULATION',fill=(255,182,91),font=f24)
        draw.text((20,49),'Kinematic source poses',fill='#c0ccd7',font=f20)
        draw.text((width+20,49),'Observable feedback + sensor bias/noise',fill='#c0ccd7',font=f20)
        draw.line((width,top,width,top+height),fill='#475569',width=2)
        draw.text((20,top+height+10),f'PICO  |  timeline {i/50:.2f}s  |  source {(i-start)/50:.2f}s  |  WORST HEADING WINDOW',fill='white',font=f20)
        root_error=np.linalg.norm(actual[i,:3]-desired[i,:3])
        draw.text((20,top+height+42),f'Heading error {angles[i]*180/np.pi:.1f} deg   |   Root error {root_error:.3f} m   |   Same fixed world camera',fill='#bcc8d4',font=f18)
        return canvas
    video=output/'worst_heading_window.source_vs_actual.fixed_world.mp4'
    picture=output/'worst_heading.source_vs_actual.fixed_world.png'
    try:
        frame(worst).save(picture)
        with imageio.get_writer(str(video),fps=25,codec='libx264',quality=8,macro_block_size=16) as writer:
            for i in range(begin,end+1,2):writer.append_data(np.asarray(frame(i)))
    finally:renderer.close()
    receipt=dict(worst_qpos_index=worst,worst_timeline_s=worst/50,worst_source_elapsed_s=(worst-start)/50,
        heading_error_deg=float(angles[worst]*180/np.pi),window_qpos_indices=[begin,end],
        video=str(video),snapshot=str(picture),fixed_world_camera=True,no_pose_transforms=True,
        trace_sha256=sha(case/'trace.npz'),reference_sha256=sha(path),video_sha256=sha(video))
    (output/'worst_heading_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)


if __name__=='__main__':run()
