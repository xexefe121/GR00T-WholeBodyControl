import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from gear_sonic.utils.g1_23dof_contract import HARDWARE_23_JOINT_NAMES
from gear_sonic.utils.g1_23dof_pico_retargeted_producer import SOMA_MJ29_JOINT_NAMES
from gear_sonic.utils.g1_true23_pico_body_adapter import Native23PicoBodyAdapter,SOURCE_BODY_ALIASES


def fixture(frame=20):
    path=Path(__file__).resolve().parents[2]/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1/contract.json'
    contract=json.loads(path.read_text())
    names=list(reversed(contract['body_names']))
    stamp=1_000_000_000+frame*20_000_000
    quat=Rotation.from_euler('z',frame*.01).as_quat()
    body=dict(schema_version=1,kind='native23_current_original29_body_pose',original29_axes_preserved=True,
        source_frame_index=frame,reference_monotonic_ns=stamp,capture_monotonic_ns=stamp,
        joint_names=list(SOMA_MJ29_JOINT_NAMES),joint_position=(np.arange(29)*.01+frame*.001).tolist(),
        body_names=names,body_position_w=[[i*.01+frame*.002,0.,.8] for i in range(24)],
        body_quaternion_xyzw=np.tile(quat,(24,1)).tolist(),task_names=['left_hand','right_hand','head'],
        task_position_w=[[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]],task_quaternion_xyzw=np.tile(quat,(3,1)).tolist())
    return contract,dict(native23_body_pose=body,control_source_frame_index=frame,control_monotonic_ns=stamp,
                         q_ref23_native=[999.]*23)


def test_current_named_pose_backward_derivatives_and_owned_objectives():
    c,w=fixture();a=Native23PicoBodyAdapter(c)
    first=a.adapt(w,received_monotonic_ns=w['control_monotonic_ns']+30_000_000)
    ids=[SOMA_MJ29_JOINT_NAMES.index(n) for n in HARDWARE_23_JOINT_NAMES]
    np.testing.assert_array_equal(first.packet.fields['joint_pos'],np.asarray(w['native23_body_pose']['joint_position'])[ids])
    assert first.packet.fields['body_pos_w'][0,0]==23*.01+20*.002
    np.testing.assert_array_equal(first.packet.fields['joint_vel'],0.)
    np.testing.assert_array_equal(first.task_position,w['native23_body_pose']['task_position_w'])
    w['native23_body_pose']['task_position_w'][0][0]=1000.
    assert first.task_position[0,0]==1.
    _,w=fixture(21);second=a.adapt(w,received_monotonic_ns=w['control_monotonic_ns']+30_000_000)
    np.testing.assert_allclose(second.packet.fields['joint_vel'],.05,atol=1e-12)
    np.testing.assert_allclose(second.packet.fields['body_lin_vel_w'],np.tile([.1,0.,0.],(24,1)),atol=1e-12)
    np.testing.assert_allclose(second.packet.fields['body_ang_vel_w'],np.tile([0.,0.,.5],(24,1)),atol=1e-12)
    assert second.packet.sequence==1 and second.packet.source_time==.02
    assert second.source['source_frame_index']==21 and second.source['source_age_ns']==30_000_000


@pytest.mark.parametrize('failure',['old_schema','stale','reorder','wrong_current','missing_head','invalid_quaternion'])
def test_rejections_latch_and_require_explicit_rearm(failure):
    c,w=fixture();a=Native23PicoBodyAdapter(c);now=w['control_monotonic_ns']+30_000_000
    if failure=='old_schema':del w['native23_body_pose']
    if failure=='stale':now+=100_000_000
    if failure=='reorder':a.adapt(copy.deepcopy(w),received_monotonic_ns=now)
    if failure=='wrong_current':w['native23_body_pose']['source_frame_index']-=1
    if failure=='missing_head':w['native23_body_pose']['task_position_w'].pop()
    if failure=='invalid_quaternion':w['native23_body_pose']['body_quaternion_xyzw'][0]=[0.,0.,0.,0.]
    with pytest.raises(ValueError):a.adapt(w,received_monotonic_ns=now)
    _,valid=fixture()
    with pytest.raises(ValueError,match='latched'):a.adapt(valid,received_monotonic_ns=valid['control_monotonic_ns'])
    a.rearm(1)
    assert a.adapt(valid,received_monotonic_ns=valid['control_monotonic_ns']).packet.epoch==1


def test_receiver_epoch_preserves_initial_packet_age():
    c,w=fixture();a=Native23PicoBodyAdapter(c)
    gate=SimpleNamespace(epoch=0,last_sequence=-1,epoch_started_at=0.,fault=None)
    calls=[]
    controller=SimpleNamespace(receiver=SimpleNamespace(gate=gate),receive=lambda *args:calls.append(args) or True)
    a.receive(controller,w,received_monotonic_ns=w['control_monotonic_ns']+30_000_000,now=10.)
    assert gate.epoch_started_at==9.97
    assert calls[0][0].source_time==0.


def test_soma_wrist_roll_names_preserve_body_frames_and_original_hand_goals():
    c,w=fixture();body=w['native23_body_pose']
    body['body_names']=[SOURCE_BODY_ALIASES.get(name,name) for name in body['body_names']]
    result=Native23PicoBodyAdapter(c).adapt(w,received_monotonic_ns=w['control_monotonic_ns'])
    for native,source in SOURCE_BODY_ALIASES.items():
        source_index=body['body_names'].index(source);target_index=c['body_names'].index(native)
        np.testing.assert_array_equal(result.packet.fields['body_pos_w'][target_index],body['body_position_w'][source_index])
        assert result.source['body_name_mapping'][native]==source
    np.testing.assert_array_equal(result.task_position,body['task_position_w'])
