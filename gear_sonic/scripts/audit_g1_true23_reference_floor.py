"""Independent compiled-native collision geometry audit of full motion references."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from gear_sonic.utils.g1_true23_mpc_student import load_inputs,sha256


def vertical_support(model,data,geom,normal):
    rotation=data.geom_xmat[geom].reshape(3,3)
    direction=rotation.T@normal
    size=model.geom_size[geom];kind=int(model.geom_type[geom])
    if kind==int(mujoco.mjtGeom.mjGEOM_SPHERE):return float(size[0])
    if kind==int(mujoco.mjtGeom.mjGEOM_CAPSULE):return float(size[0]+size[1]*abs(direction[2]))
    if kind==int(mujoco.mjtGeom.mjGEOM_CYLINDER):return float(size[0]*np.linalg.norm(direction[:2])+size[1]*abs(direction[2]))
    if kind==int(mujoco.mjtGeom.mjGEOM_BOX):return float(np.abs(direction)@size)
    if kind==int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):return float(np.linalg.norm(direction*size))
    if kind==int(mujoco.mjtGeom.mjGEOM_MESH):
        mesh=int(model.geom_dataid[geom]);start=int(model.mesh_vertadr[mesh]);count=int(model.mesh_vertnum[mesh])
        return float(-np.min(model.mesh_vert[start:start+count]@direction))
    raise ValueError(f"unsupported active collision shape {kind}")


def stats(values):return np.percentile(values,[0,50,95,99,100]).tolist()


def run(args):
    assert mujoco.__version__=="3.2.3"
    args.output.mkdir(parents=True,exist_ok=False)
    results=[];started=time.perf_counter()
    for clip in args.clips:
        model,contract,_,_,timeline=load_inputs(args.bundle,clip)
        path=args.references/clip/"reference.npz"
        with np.load(path,allow_pickle=False) as a:motion={key:a[key].copy() for key in a.files}
        count=len(motion["joint_pos"]);dt=1/float(motion["fps"][0])
        floor=model.geom("floor").id;data=mujoco.MjData(model)
        floor_body=model.geom_bodyid[floor]
        assert floor_body==0 and model.geom_type[floor]==mujoco.mjtGeom.mjGEOM_PLANE
        feet=[model.body(f"{side}_ankle_roll_link").id for side in ("left","right")]
        active=[g for g in range(model.ngeom) if g!=floor and ((model.geom_contype[g]&model.geom_conaffinity[floor]) or (model.geom_contype[floor]&model.geom_conaffinity[g]))]
        foot_geoms=[[g for g in active if model.geom_bodyid[g]==body] for body in feet]
        assert all(len(group)==4 for group in foot_geoms)
        assert all(model.geom_type[g]==mujoco.mjtGeom.mjGEOM_SPHERE for group in foot_geoms for g in group)
        heights=np.empty((count,len(active)));foot_height=np.empty((count,2));com=np.empty((count,3))
        contact_depth=np.zeros(count);body_floor_depth=np.zeros(count)
        self_pairs=defaultdict(lambda:dict(frames=0,maximum_depth_m=0.))
        floor_analytic_error=0.;body_fk_error=0.
        for frame in range(count):
            data.qpos[:]=np.r_[motion["body_pos_w"][frame,0],motion["body_quat_w"][frame,0],motion["joint_pos"][frame]]
            data.qvel[:]=0.;data.qacc_warmstart[:]=0.
            mujoco.mj_forward(model,data)
            normal=data.geom_xmat[floor].reshape(3,3)[:,2]
            np.testing.assert_allclose(normal,[0,0,1],atol=1e-12,rtol=0)
            origin=data.geom_xpos[floor]
            heights[frame]=[(data.geom_xpos[g]-origin)@normal-vertical_support(model,data,g,normal) for g in active]
            foot_height[frame]=[min(heights[frame,active.index(g)] for g in group) for group in foot_geoms]
            com[frame]=data.subtree_com[1]
            body_fk_error=max(body_fk_error,float(np.max(np.abs(data.xpos[1:]-motion["body_pos_w"][frame]))))
            seen=set()
            for contact in data.contact:
                ga,gb=int(contact.geom1),int(contact.geom2)
                if floor in (ga,gb):
                    other=gb if ga==floor else ga
                    if any(other in group for group in foot_geoms):
                        floor_analytic_error=max(floor_analytic_error,abs(float(contact.dist)-heights[frame,active.index(other)]))
                    elif contact.dist<0:body_floor_depth[frame]=max(body_floor_depth[frame],-float(contact.dist))
                elif contact.dist<0:
                    depth=-float(contact.dist);contact_depth[frame]=max(contact_depth[frame],depth)
                    pair=tuple(sorted((model.body(int(model.geom_bodyid[ga])).name,model.body(int(model.geom_bodyid[gb])).name)))
                    key=" | ".join(pair)
                    if key not in seen:self_pairs[key]["frames"]+=1;seen.add(key)
                    self_pairs[key]["maximum_depth_m"]=max(self_pairs[key]["maximum_depth_m"],depth)
        lift=np.maximum(-foot_height.min(1),0.)
        all_lift=np.maximum(-heights.min(1),0.)
        velocity=np.diff(lift)/dt;acceleration=np.diff(lift,n=2)/dt**2
        com_acc=np.gradient(np.gradient(com[:,2],dt),dt)
        possible_flight=(foot_height.min(1)>.002)&(heights.min(1)>.002)
        ballistic=possible_flight&(np.abs(com_acc+9.81)<2.)
        original=None;translation=None
        if args.compare:
            original=args.compare/clip/"reference.npz"
            with np.load(original,allow_pickle=False) as a:
                translation=motion["body_pos_w"][:,0]-a["body_pos_w"][:,0]
                rigid_error=np.max(np.abs((motion["body_pos_w"]-a["body_pos_w"])-translation[:,None]))
                expected=np.zeros_like(motion["body_lin_vel_w"])
                expected[:,:,2]=np.gradient(translation[:,2],dt)[:,None]
                invariant=dict(joint_pos_bit_exact=np.array_equal(motion["joint_pos"],a["joint_pos"]),
                               joint_vel_bit_exact=np.array_equal(motion["joint_vel"],a["joint_vel"]),
                               body_quat_bit_exact=np.array_equal(motion["body_quat_w"],a["body_quat_w"]),
                               body_ang_vel_bit_exact=np.array_equal(motion["body_ang_vel_w"],a["body_ang_vel_w"]),
                               fps_bit_exact=np.array_equal(motion["fps"],a["fps"]),
                               rigid_allbody_translation_error_m=float(rigid_error),
                               horizontal_translation_max_abs_m=float(np.max(np.abs(translation[:,:2]))),
                               vertical_velocity_derivative_error=float(np.max(np.abs(motion["body_lin_vel_w"]-a["body_lin_vel_w"]-expected))),
                               translation_lift_stats_m=stats(translation[:,2]),
                               translation_velocity_abs_stats_mps=stats(np.abs(np.diff(translation[:,2])/dt)),
                               translation_acceleration_abs_stats_mps2=stats(np.abs(np.diff(translation[:,2],n=2)/dt**2)))
        else:invariant=None
        source=next(p for p in timeline["phases"] if p["name"]=="source_motion")
        first,stop=source["frame_start"],source["frame_stop"]
        record=dict(clip=clip,reference_sha256=sha256(path),frames=count,fps=1/dt,
                    floor_geom=int(floor),floor_z=float(data.geom_xpos[floor,2]),foot_geom_ids=foot_geoms,
                    collision_sphere_radius_m=[[float(model.geom_size[g,0]) for g in group] for group in foot_geoms],
                    foot_clearance_min_m=foot_height.min(0).tolist(),source_foot_clearance_min_m=foot_height[first:stop].min(0).tolist(),
                    first3source_foot_clearance_min_m=foot_height[first:first+150].min(0).tolist(),
                    frames_requiring_lift=int(np.sum(lift>1e-8)),minimum_lift_stats_m=stats(lift),
                    minimum_lift_velocity_abs_stats_mps=stats(np.abs(velocity)),
                    minimum_lift_acceleration_abs_stats_mps2=stats(np.abs(acceleration)),
                    all_body_floor_lift_max_m=float(all_lift.max()),nonfoot_floor_contact_depth_max_m=float(body_floor_depth.max()),
                    self_collision_depth_max_m=float(contact_depth.max()),self_collision_frames=int(np.sum(contact_depth>1e-6)),
                    self_collision_pairs=sorted([dict(pair=k,**v) for k,v in self_pairs.items()],key=lambda v:v["maximum_depth_m"],reverse=True),
                    both_feet_and_allbody_clear_2mm_frames=int(possible_flight.sum()),
                    ballistic_compatible_com_acceleration_frames=int(ballistic.sum()),
                    flight_interpretation="geometry-only clear frames are possible flight; no source contact labels or physical flight proof; raw upward lift stays zero there",
                    actual_mujoco_foot_contact_vs_analytic_clearance_max_abs_m=float(floor_analytic_error),
                    native_body_fk_max_abs_error_m=body_fk_error,comparison=invariant)
        np.savez_compressed(args.output/f"{clip}_floor_arrays.npz",active_geom_ids=active,clearance=heights,foot_clearance=foot_height,
                            minimum_lift=lift,all_body_minimum_lift=all_lift,com=com,com_vertical_acceleration=com_acc,
                            possible_flight=possible_flight,ballistic_compatible=ballistic,self_collision_depth=contact_depth,
                            nonfoot_floor_contact_depth=body_floor_depth)
        results.append(record)
        print(json.dumps({k:v for k,v in record.items() if k!="self_collision_pairs"}),flush=True)
    report=dict(mujoco=mujoco.__version__,full_all_frames_audited=True,physical_model_modified=False,root_forces_applied=False,
                native_model_binding={name:sha256(args.bundle/name) for name in ("native_prepared.xml","prepared_model_arrays.npz","contract.json","manifest.json")},
                reference_directory=str(args.references.resolve()),comparison_directory=None if args.compare is None else str(args.compare.resolve()),
                code_sha256=sha256(Path(__file__)),results=results,elapsed_s=time.perf_counter()-started)
    (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    (args.output/"audit_snapshot.py").write_bytes(Path(__file__).read_bytes())


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--bundle",type=Path,required=True);p.add_argument("--references",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--compare",type=Path)
    p.add_argument("--clips",type=lambda value:value.split(','),default=["pico","walk002","walk003","walk008"])
    run(p.parse_args())
