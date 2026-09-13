"""Exercise the optional body extension against the installed pinned Newton FK."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from gear_sonic.utils.g1_23dof_xr24_soma_stream import _SomaLocalFrameBuilder,_True23BodyTermBuilder
from gear_sonic.utils.g1_23dof_pico_retargeted_producer import SOMA_MJ29_JOINT_NAMES
from soma_retargeter.pipelines.newton_pipeline import NewtonPipeline


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    skeleton=_SomaLocalFrameBuilder(Path('/root/.cache/g1_true23_soma/source')).skeleton
    builder=_True23BodyTermBuilder(NewtonPipeline(skeleton))
    q=np.zeros(36);q[2]=.8;q[6]=1.
    timing=dict(source_frame_index=20,reference_monotonic_ns=1_000_000_000,capture_monotonic_ns=1_000_000_000)
    original=builder.build_full_body(q,timing)
    q[7+SOMA_MJ29_JOINT_NAMES.index('waist_pitch_joint')]=.15
    q[7+SOMA_MJ29_JOINT_NAMES.index('left_wrist_yaw_joint')]=.3
    q[7+SOMA_MJ29_JOINT_NAMES.index('right_wrist_pitch_joint')]=-.3
    changed=builder.build_full_body(q,timing)
    delta=np.linalg.norm(np.asarray(changed['task_position_w'])-original['task_position_w'],axis=1)
    assert np.all(delta>.01),delta
    # The legacy builder continues to zero those absent physical axes.
    zero=q.copy()
    for name in ('waist_pitch_joint','left_wrist_yaw_joint','right_wrist_pitch_joint'):zero[7+SOMA_MJ29_JOINT_NAMES.index(name)]=0.
    legacy_changed=builder.build(q,timing);legacy_zero=builder.build(zero,timing)
    for key in ('vr_3point_local_target','vr_3point_local_orn_target','reference_anchor_quaternion_xyzw'):
        np.testing.assert_array_equal(legacy_changed[key],legacy_zero[key])
    durations=[]
    for _ in range(100):
        started=time.perf_counter();builder.build_full_body(q,timing);durations.append((time.perf_counter()-started)*1000)
    wire=dict(control_source_frame_index=20,control_monotonic_ns=1_000_000_000,native23_body_pose=changed)
    (a.output/'wire_fixture.json').write_text(json.dumps(wire,indent=2)+'\n')
    report=dict(passed=True,original_missing_axes_task_displacement_m=delta.tolist(),legacy_payload_unchanged=True,
        extra_fk_ms_p50_p95_max=np.percentile(durations,[50,95,100]).tolist(),live_session=False,simulation_ready=False)
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
