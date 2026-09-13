"""Paired original-pose derivative and derivative+floor reference ablations.

No controller/plant edits. Original source archives remain byte-preserved.
"""
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('E:/codex-artifacts/sonic23_teleop_six_hour_20260910')
METADATA = BASE/'mjbatch_intent_inputs_v1'
OUTPUTS = {'A':BASE/'mjbatch_original_canonical_inputs_v1',
           'B':BASE/'mjbatch_original_canonical_floor_inputs_v1'}
AUTHORITY=ROOT/'gear_sonic/scripts/evaluate_g1_true23_bfmzero.py'
AUTHORITY_BYTES=AUTHORITY.read_bytes()
sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from gear_sonic.scripts.evaluate_g1_true23_bfmzero import MODEL, PHYSICS, load_case_motion, load_motion

DT=.02
OMEGA, BUFFER, EPS = 15., .020, .000001


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value): path.write_text(json.dumps(value,indent=2,allow_nan=False))


def copy(source, destination):
    assert not destination.exists()
    shutil.copy2(source,destination)
    assert sha(source)==sha(destination)


def wsl(path):
    s=str(path).replace('\\','/')
    return '/mnt/'+s[0].lower()+'/'+s[3:]


def scalar_stats(x):
    return dict(min=float(np.min(x)),max=float(np.max(x)),p95=float(np.percentile(x,95)))


def canonical(motion):
    out={k:v.copy() for k,v in motion.items()}
    out['joint_vel']=np.gradient(motion['joint_pos'],DT,axis=0)
    out['body_lin_vel_w']=np.gradient(motion['body_pos_w'],DT,axis=0)
    n=len(out['joint_pos']); earlier=np.maximum(np.arange(n)-1,0); later=np.minimum(np.arange(n)+1,n-1)
    for body in range(24):
        r=Rotation.from_quat(motion['body_quat_w'][:,body,[1,2,3,0]])
        out['body_ang_vel_w'][:,body]=(r[later]*r[earlier].inv()).as_rotvec()/((later-earlier)*DT)[:,None]
    return out


def profile(raw):
    smooth=np.empty_like(raw); smooth[0],velocity=raw[0],0.
    decay=np.exp(-OMEGA*DT)
    for i in range(1,len(raw)):
        offset=smooth[i-1]-raw[i]; combo=velocity+OMEGA*offset
        smooth[i]=raw[i]+(offset+combo*DT)*decay
        velocity=(velocity-OMEGA*combo*DT)*decay
    guard=raw+EPS>smooth+BUFFER
    return np.maximum(raw+EPS,smooth+BUFFER),smooth,guard


native_path=ROOT.parent/'GR00T-WholeBodyControl'/MODEL
native=mujoco.MjModel.from_xml_path(str(native_path)); data=mujoco.MjData(native)
floor=native.geom('floor').id
feet=[native.body(s+'_ankle_roll_link').id for s in ('left','right')]
spheres=[[i for i in range(native.ngeom) if native.geom_bodyid[i]==b and native.geom_type[i]==mujoco.mjtGeom.mjGEOM_SPHERE and native.geom_contype[i]!=0] for b in feet]
assert spheres==[[15,16,17,18],[30,31,32,33]]
physics=json.loads((ROOT/PHYSICS).read_text())
speed=np.asarray(physics['physics']['velocity_limit_hardware_radps'])


def clearance(m):
    result=np.empty((len(m['joint_pos']),2))
    for f in range(len(result)):
        data.qpos[:]=np.r_[m['body_pos_w'][f,0],m['body_quat_w'][f,0],m['joint_pos'][f]]
        mujoco.mj_kinematics(native,data)
        normal=data.geom_xmat[floor].reshape(3,3)[:,2]
        np.testing.assert_array_equal(normal,[0,0,1])
        for side,ids in enumerate(spheres):
            result[f,side]=np.min((data.geom_xpos[ids]-data.geom_xpos[floor])@normal-native.geom_size[ids,0])
    return result


for folder in OUTPUTS.values():
    folder.mkdir(exist_ok=False)
    (folder/'producer_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (folder/'validation_authority_snapshot.py').write_bytes(AUTHORITY_BYTES)
entries=[]
for clip in ('pico','walk002'):
    source,timeline,source_path=load_motion(clip)
    meta=json.loads((METADATA/clip/'portable_receipt.json').read_text())
    assert sha(source_path)==meta['original_native_reference_sha256']
    original29=Path(meta['original29_reference_path'])
    assert sha(original29)==meta['original29_reference_sha256']
    with np.load(original29) as z: original_root=z['source_qpos29'][:,:3].copy()
    A=canonical(source)
    for key in ('fps','joint_pos','body_pos_w','body_quat_w'):
        np.testing.assert_array_equal(A[key],source[key])
    old_clearance=clearance(A)
    raw=np.maximum(0,-old_clearance.min(axis=1))
    lift,smooth,guard=profile(raw)
    for end in (11,361,min(1000,len(lift)),len(lift)-1):
        np.testing.assert_array_equal(profile(raw[:end])[0],lift[:end])
    B={k:v.copy() for k,v in A.items()}
    B['body_pos_w'][:,:,2]+=lift[:,None]
    B['body_lin_vel_w'][:,:,2]=np.gradient(B['body_pos_w'][:,:,2],DT,axis=0)
    new_clearance=clearance(B)
    np.testing.assert_allclose(new_clearance,old_clearance+lift[:,None],atol=2e-10,rtol=0)
    assert new_clearance.min()>=-1e-10
    for variant,motion in (('A',A),('B',B)):
        out=OUTPUTS[variant]/clip; out.mkdir()
        copy(source_path,out/'original_native_reference.npz')
        copy(METADATA/clip/'original_timeline.json',out/'original_timeline.json')
        copy(ROOT/PHYSICS,out/'physics_contract_snapshot.json')
        np.savez_compressed(out/'reference.npz',**motion)
        source_phase=meta['source_phase']; span=slice(source_phase['frame_start'],source_phase['frame_stop'])
        derivative_transform=dict(kind='explicit_central_derivative_adaptation_of_original_native_poses',
            original_reference_file='original_native_reference.npz',original_reference_sha256=sha(source_path),
            poses_unchanged_from_original=variant=='A', joint_positions_unchanged_from_original=True,
            body_quaternions_unchanged_from_original=True,fps_unchanged=True,source_timing_changed=False,
            joint_velocity_convention='np.gradient(joint_pos,0.02,axis=0)',
            body_linear_velocity_convention='np.gradient(body_pos_w,0.02,axis=0)',
            body_angular_velocity_convention='world-axis log(R[i+1]*R[i-1]^-1)/.04; one-sided at archive endpoints',
            original_angular_velocity_convention='world backward difference; source channels preserved through lifecycle splices',
            future_pose_support_frames=1,future_pose_support_seconds=.02,
            canonical_before_floor_reference_sha256=sha(OUTPUTS['A']/clip/'reference.npz'),
            derivative_component_abs_source_p50_p95_max={k:np.percentile(np.abs(A[k][span]-source[k][span]),[50,95,100]).tolist() for k in ('joint_vel','body_lin_vel_w','body_ang_vel_w')})
        write(out/'derivative_transform_receipt.json',derivative_transform)
        derivative_meta=dict(kind=derivative_transform['kind'],transform_receipt_file='derivative_transform_receipt.json',
                             transform_receipt_sha256=sha(out/'derivative_transform_receipt.json'),future_pose_support_frames=1,future_pose_support_seconds=.02)
        floor_meta=None
        if variant=='B':
            copy(OUTPUTS['A']/clip/'reference.npz',out/'before_floor_reference.npz')
            copy(OUTPUTS['A']/clip/'report.json',out/'before_floor_report.json')
            copy(OUTPUTS['A']/clip/'portable_receipt.json',out/'before_floor_portable_receipt.json')
            np.savez_compressed(out/'frame_lift.npz',frame_lift_m=lift,raw_required_lift_m=raw,
                before_foot_clearance_m=old_clearance,after_foot_clearance_m=new_clearance,
                filtered_required_lift_m=smooth,clearance_guard_active=guard)
            t=dict(schema_version=1,kind='causal_upward_whole_pose_translation',clip=clip,
                input_reference_file='before_floor_reference.npz',input_reference_sha256=sha(out/'before_floor_reference.npz'),
                input_report_sha256=sha(out/'before_floor_report.json'),output_reference_sha256=sha(out/'reference.npz'),
                frame_lift_file='frame_lift.npz',frame_lift_sha256=sha(out/'frame_lift.npz'),
                filter=dict(kind='critically_damped_exact_discrete',omega_per_second=OMEGA,fixed_buffer_m=BUFFER,
                    clearance_guard_epsilon_m=EPS,dt_seconds=DT,initial_filtered_lift='first_required_lift',
                    initial_filter_velocity_mps=0.,pose_preview_frames=0,guard_active_frames=int(guard.sum()),
                    parameters_unchanged_from_v4=True,clearance_recomputed_from_original_native_pose=True,v3_lifts_reused=False),
                pose_prefix_causality_tests_passed=True,exported_linear_velocity_future_pose_support_frames=1,
                exported_linear_velocity_future_pose_support_seconds=.02,native_model_sha256=sha(native_path),physics_contract_sha256=sha(ROOT/PHYSICS),
                floor_geom_id=floor,foot_geom_ids=spheres,foot_sphere_radii_m=native.geom_size[np.array(spheres),0].tolist(),
                lift_min_m=float(lift.min()),lift_max_m=float(lift.max()),lift_p95_m=float(np.percentile(lift,95)),
                added_vertical_velocity_max_mps=float(np.max(np.abs(np.diff(lift)/DT))),
                added_vertical_acceleration_max_mps2=float(np.max(np.abs(np.diff(lift,2)/DT**2))),
                initial_foot_clearance_before_m=old_clearance[0].tolist(),initial_foot_clearance_after_m=new_clearance[0].tolist(),
                foot_clearance_after_min_m=float(new_clearance.min()),median_min_foot_clearance_after_m=float(np.median(new_clearance.min(axis=1))),
                near_floor_2mm_source_transitions_before=[int(np.count_nonzero(np.diff(old_clearance[span,i]<=.002))) for i in (0,1)],
                near_floor_2mm_source_transitions_after=[int(np.count_nonzero(np.diff(new_clearance[span,i]<=.002))) for i in (0,1)],
                original29_world_root_error_max_m=float(np.max(np.linalg.norm(motion['body_pos_w'][:,0]-original_root,axis=-1))),
                root_relative_intent_preserved=True,joints_unchanged=True,derivatives_unchanged_except_body_z_linear_velocity_relative_to_canonical_input=True,
                source_timing_changed=False,physical_floor_changed=False,self_collision_geometry_changed=False,
                foot_floor_clear_only=True,dynamics_qualified=False,hardware_authorized=False)
            assert t['original29_world_root_error_max_m']<.20
            write(out/'floor_transform_receipt.json',t)
            floor_meta={k:t[k] for k in ('kind','lift_min_m','lift_max_m','physical_floor_changed','root_relative_intent_preserved','joints_unchanged','source_timing_changed')}
            floor_meta.update(transform_receipt_file='floor_transform_receipt.json',transform_receipt_sha256=sha(out/'floor_transform_receipt.json'),
                              frame_lift_file='frame_lift.npz',frame_lift_sha256=sha(out/'frame_lift.npz'))
        report=dict(kind='original_native_pose_canonical_derivative_ablation'+('_with_floor_lift' if variant=='B' else ''),
            variant=variant,clip=clip,frames=len(lift),full_original_timeline=True,reference_sha256=sha(out/'reference.npz'),
            reference_derivative_transform=derivative_meta,source_timing_scale=1.,source_timing_changed=False,
            original_sources_unchanged=True,joint_pose_retargeted=False,waist_pose_retargeted=False,arm_pose_retargeted=False,
            all_original_pose_arrays_bitexact=variant=='A',only_pose_change_common_world_z=variant=='B',
            foot_floor_clear=bool((new_clearance if variant=='B' else old_clearance).min()>=0),
            world_root_error_from_original29_max_m=float(np.max(np.linalg.norm(motion['body_pos_w'][:,0]-original_root,axis=-1))),
            dynamics_qualified=False,full_body_tracking_qualified=False,hardware_authorized=False)
        if floor_meta: report['reference_floor_transform']=floor_meta
        write(out/'report.json',report)
        load_case_motion(clip,out/'reference.npz')
        receipt={k:meta[k] for k in ('schema_version','clip','original_native_reference_sha256','original29_reference_sha256',
            'original_native_reference_path','original_native_reference_wsl_path','original29_reference_path','original29_reference_wsl_path',
            'base_native_bundle_reference_sha256','base_native_bundle_reference_wsl_path','native_model_sha256','original29_model_sha256',
            'physics_contract_sha256','fps','frame_count','original29_archive_has_fps','source_clock_authority','prehistory_frames',
            'total_requested_controls','source_requested_controls','source_phase','phase_counts','timeline_file','timeline_sha256','fields')}
        receipt.update(kind=report['kind'],reference_file='reference.npz',reference_path=str(out/'reference.npz'),reference_wsl_path=wsl(out/'reference.npz'),
            reference_sha256=sha(out/'reference.npz'),source_artifact_report_sha256=sha(out/'report.json'),source_artifact_directory=str(out),
            source_artifact_reference_sha256=sha(out/'reference.npz'),reference_derivative_transform=derivative_meta,
            validation={k:True for k in ('all_frames_checked','finite_fields','same_fields_shapes_as_original','fps_50','full_original_timeline',
                'source_phase_counts_unchanged','unit_body_quaternions','native_joint_bounds','adjacent_joint_speed_limits',
                'joint_vel_matches_timed_positions','body_lin_vel_matches_timed_positions','body_ang_vel_matches_world_rotation_intervals',
                'native_body_fk_positions','native_body_fk_rotations','copied_bytes_match_source','source_provenance_matches')},
            validation_authority=dict(function='load_case_motion',module_path=str(AUTHORITY),module_sha256=hashlib.sha256(AUTHORITY_BYTES).hexdigest(),
                snapshot_file='../validation_authority_snapshot.py',mujoco=mujoco.__version__),
            validation_numbers=dict(adjacent_joint_velocity_ratio_max=float(np.max(np.abs(np.diff(motion['joint_pos'],axis=0))/DT/speed))),
            physical_initialization_motion='declared_reference_frame_10',recorded_target_seed_validation_motion='original_native_reference',
            seed_initial_state_equality_to_retarget_required=False,seed_physical_states_may_be_copied=False,
            reference_geometry_changed=variant=='B',reference_velocity_convention_changed=True,source_timing_changed=False,source_timing_scale=1.,
            reference_is_kinematic_only=True,full_body_tracking_qualified=False,dynamics_qualified=False,hardware_authorized=False)
        if floor_meta: receipt['reference_floor_transform']=floor_meta
        receipt['copied_files']={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in out.iterdir() if p.is_file()}
        write(out/'portable_receipt.json',receipt)
        entry=dict(clip=clip,variant=variant,frames=len(lift),source_controls=meta['source_requested_controls'],
            reference_path=str(out/'reference.npz'),reference_wsl_path=wsl(out/'reference.npz'),reference_sha256=sha(out/'reference.npz'),
            portable_receipt_sha256=sha(out/'portable_receipt.json'),raw_required_lift_max_m=float(raw.max()),
            lift_max_m=float(lift.max()) if variant=='B' else 0.,guard_frames=int(guard.sum()) if variant=='B' else 0,
            foot_clearance_min_m=float((new_clearance if variant=='B' else old_clearance).min()),all_frames_root_validated=True)
        entries.append(entry);print(json.dumps(entry),flush=True)
assert AUTHORITY.read_bytes()==AUTHORITY_BYTES
for variant,folder in OUTPUTS.items():
    write(folder/'manifest.json',dict(kind='original_native_pose_paired_ablation',variant=variant,
        entries=[e for e in entries if e['variant']==variant],producer_sha256=sha(__file__),
        original_sources_unchanged=True,source_timing_changed=False,dynamics_qualified=False,hardware_authorized=False))
