"""Compare saved native physics and received reference poses; no new dynamics."""
from pathlib import Path
import json
import numpy as np,mujoco
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[2]
FW=Path('E:/codex-artifacts/sonic23_teleop_resume_20260911/onboard_factory_firmware_v1')
bundle=ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1'
model=mujoco.MjModel.from_xml_path(str(bundle/'native_prepared.xml'))
with np.load(bundle/'prepared_model_arrays.npz',allow_pickle=False) as z:
    for key in z.files:getattr(model,key)[:]=z[key]
mujoco.mj_setConst(model,mujoco.MjData(model))
w,h=440,350
model.vis.global_.offwidth=w;model.vis.global_.offheight=h
model.vis.headlight.ambient[:]=.6;model.vis.headlight.diffuse[:]=.8
data=mujoco.MjData(model);renderer=mujoco.Renderer(model,width=w,height=h)
font=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',20)
small=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',16)
out=FW/'full_body_failure_snapshots_v1';out.mkdir(exist_ok=False)
records=[]
try:
    for clip,times in [('walk002',[5.,7.,9.,10.8]),('pico',[40.,48.,50.5,51.7])]:
        bank=FW.parent/'causal_dynamics_v1/focused_walk002_task_closure_bank_v1'/f'{clip}.npz'
        with np.load(bank,allow_pickle=False) as z:r={k:z[k].copy() for k in ('root','root_rotation','joint','feet','tasks')}
        traces=[]
        for phase in ('eval_bootstrap','eval_00200'):
            path=FW/'full_body_command_space_ppo_v1'/phase/clip/'trace.npz'
            with np.load(path,allow_pickle=False) as z:traces.append({k:z[k].copy() for k in ('physics_qpos','frame')})
        rows=[]
        for t in times:
            step=round(t/.002);control=(step-1)//10
            assert step<=min(len(z['physics_qpos']) for z in traces)
            frame=int(traces[0]['frame'][control]);assert frame==traces[1]['frame'][control]
            q=np.empty(4);mujoco.mju_mat2Quat(q,r['root_rotation'][frame].ravel())
            poses=[np.r_[r['root'][frame],q,r['joint'][frame]],*[z['physics_qpos'][step-1] for z in traces]]
            rows.append((t,frame,poses,np.r_[r['feet'][frame],r['tasks'][frame]]))
        visible=np.flatnonzero((model.geom_bodyid!=0)&(model.geom_rgba[:,3]>0))
        radius=model.geom_rbound[visible,None]
        low=np.full(3,np.inf);high=np.full(3,-np.inf)
        for t,frame,poses,points in rows:
            low=np.minimum(low,points.min(0)-.05);high=np.maximum(high,points.max(0)+.05)
            for q in poses:
                data.qpos[:]=q;mujoco.mj_kinematics(model,data)
                low=np.minimum(low,(data.geom_xpos[visible]-radius).min(0)-.04)
                high=np.maximum(high,(data.geom_xpos[visible]+radius).max(0)+.04)
        cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:]=(low+high)/2;cam.distance=2.5;cam.azimuth=145.;cam.elevation=-18.
        corners=np.array([[x,y,z] for x in (low[0],high[0]) for y in (low[1],high[1]) for z in (low[2],high[2])])
        for _ in range(150):
            renderer.update_scene(data,camera=cam);view=renderer.scene.camera[0]
            delta=corners-view.pos;depth=delta@view.forward;tangent=view.frustum_top/view.frustum_near
            ratio=max(np.max(np.abs(delta@view.up/depth))/tangent,
                np.max(np.abs(delta@np.cross(view.forward,view.up)/depth))/(tangent*w/h))
            if np.all(depth>0) and ratio<.92:break
            cam.distance*=1.03
        else:raise RuntimeError('Could not fit comparison geometry')
        sheet=Image.new('RGB',(w*3,(h+37)*len(rows)+95),'#111820');draw=ImageDraw.Draw(sheet)
        draw.text((12,8),f'{clip}: reference / retained baseline / all23 checkpoint200',font=font,fill='white')
        draw.text((12,37),'Saved physics, same fixed world camera. Amber: requested ankles. Cyan: requested hand/head points.',font=small,fill='#8cddff')
        draw.text((12,62),'All23 controller FAILED. This image is diagnosis, not a successful rollout.',font=small,fill='#ffb58b')
        for i,(t,frame,poses,points) in enumerate(rows):
            for j,q in enumerate(poses):
                data.qpos[:]=q;mujoco.mj_kinematics(model,data);renderer.update_scene(data,camera=cam)
                for k,point in enumerate(points):
                    geom=renderer.scene.geoms[renderer.scene.ngeom]
                    mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_SPHERE,np.full(3,.025),point,np.eye(3).ravel(),
                        np.array([1.,.68,.18,1.]) if k<2 else np.array([.15,.85,1.,1.]))
                    renderer.scene.ngeom+=1
                y=95+i*(h+37)
                label=['Requested native pose','Retained baseline','All23 checkpoint200'][j]
                draw.text((j*w+10,y+6),f'{t:.2f}s | {label}',font=small,fill='white')
                sheet.paste(Image.fromarray(renderer.render().copy()),(j*w,y+32))
        path=out/f'{clip}.png';sheet.save(path)
        records.append(dict(clip=clip,times=times,reference_frames=[row[1] for row in rows],
            output=str(path),fixed_world_camera=True,root_alignment=False,physics_rerun=False,
            camera=dict(lookat=cam.lookat.tolist(),distance=cam.distance,azimuth=cam.azimuth,elevation=cam.elevation)))
finally:renderer.close()
(out/'render.json').write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))
