from pathlib import Path
import numpy as np
from gear_sonic.utils.g1_true23_received_features import prepare_reference,features_numpy,MeasuredHistory
from gear_sonic.utils.g1_true23_causal_receiver import CausalReceiver
from gear_sonic.scripts.evaluate_g1_true23_causal_dynamics import load_case
from gear_sonic.utils.g1_true23_bfmzero_stream import Packet


def test_streaming_and_offline_causal_features_agree():
    model,c,motion,original,timeline=load_case('walk003')
    receiver=CausalReceiver(c,now=0)
    ref=prepare_reference(motion,original,c)
    q=np.r_[motion['body_pos_w'][10,0],motion['body_quat_w'][10,0],motion['joint_pos'][10]]
    v=np.linspace(-.02,.02,29)
    for f in range(450):
        fields={k:x[f] for k,x in motion.items() if k!='fps'}
        assert receiver.receive(Packet(0,f,f*.02,fields),original['source_task_position_w'][f],original['source_task_quaternion_wxyz'][f],f*.02)
        want=features_numpy(q,v,ref,f,c['default_q'],receiver.history.prior,receiver.history.vector())
        np.testing.assert_allclose(receiver.features(q,v,f*.02),want,rtol=1e-6,atol=1e-6)
        receiver.commit(q,v,c['default_q'])


def test_future_mutation_cannot_change_actor_features():
    _,c,motion,original,_=load_case('walk003')
    frame=400
    ref=prepare_reference(motion,original,c)
    q=np.r_[motion['body_pos_w'][frame,0],motion['body_quat_w'][frame,0],motion['joint_pos'][frame]]
    h=MeasuredHistory(c)
    first=features_numpy(q,np.zeros(29),ref,frame,c['default_q'],h.prior,h.vector())
    for value in ref.values():value[frame+1:]=float('nan')
    np.testing.assert_array_equal(first,features_numpy(q,np.zeros(29),ref,frame,c['default_q'],h.prior,h.vector()))


def test_fault_is_latched_and_rearm_is_explicit():
    _,c,motion,original,_=load_case('walk003')
    receiver=CausalReceiver(c)
    fields={k:x[0] for k,x in motion.items() if k!='fps'}
    assert receiver.receive(Packet(0,0,0.,fields),original['source_task_position_w'][0],original['source_task_quaternion_wxyz'][0],0.)
    receiver.gate.check_freshness(.12)
    assert receiver.gate.fault['reason']=='packet_timeout'
    assert not receiver.receive(Packet(0,1,.02,fields),original['source_task_position_w'][0],original['source_task_quaternion_wxyz'][0],.12)
    assert receiver.gate.fault is not None


def test_rearm_applies_same_yaw_and_translation_to_body_and_tasks():
    from scipy.spatial.transform import Rotation
    _,c,motion,original,_=load_case('walk003')
    receiver=CausalReceiver(c)
    fields={k:x[400].copy() for k,x in motion.items() if k!='fps'}
    q=np.r_[fields['body_pos_w'][0],fields['body_quat_w'][0],fields['joint_pos']]
    turn=Rotation.from_euler('z',.4)
    q[:2]+=[1.2,-.7]
    q[3:7]=(turn*Rotation.from_quat(q[[4,5,6,3]])).as_quat()[[3,0,1,2]]
    receiver.commit(q,np.zeros(29),c['default_q'])
    history=receiver.history.vector().copy()
    receiver.rearm(1.,q)
    assert receiver.receive(Packet(1,0,0.,fields),original['source_task_position_w'][400],original['source_task_quaternion_wxyz'][400],1.)
    actual=receiver.samples[-1]
    np.testing.assert_allclose(actual['body_pos_w'][0,:2],q[:2],atol=1e-7)
    expected=turn.apply(original['source_task_position_w'][400]-fields['body_pos_w'][0])+q[:3]
    expected[:,2]=original['source_task_position_w'][400,:,2]
    np.testing.assert_allclose(receiver.tasks[-1][0],expected,atol=1e-6)
    np.testing.assert_array_equal(receiver.history.vector(),history)
    assert receiver.gate.epoch==1 and receiver.gate.fault is None


def test_input_loss_generates_full_body_stand_and_remains_latched():
    import json
    from gear_sonic.scripts.evaluate_g1_true23_causal_dynamics import NEW
    model,c,motion,original,timeline=load_case('walk003')
    tasks=json.loads((NEW/'causal_dynamics_v1/bank/bank.json').read_text())['tasks']
    receiver=CausalReceiver(c,model=model,standing_qpos=timeline['configured_standing_qpos'],tasks=tasks)
    fields={k:x[450].copy() for k,x in motion.items() if k!='fps'}
    q=np.r_[fields['body_pos_w'][0],fields['body_quat_w'][0],fields['joint_pos']]
    assert receiver.receive(Packet(0,0,0.,fields),original['source_task_position_w'][450],original['source_task_quaternion_wxyz'][450],0.)
    for i in range(110):
        features=receiver.features(q,np.zeros(29),.12+i*.02)
        assert features.shape==(1323,) and np.isfinite(features).all()
    assert receiver.mode=='latched_standing' and receiver.gate.fault is not None
    np.testing.assert_allclose(receiver.samples[-1]['joint_pos'],np.asarray(timeline['configured_standing_qpos'])[7:],atol=1e-7)
    np.testing.assert_allclose(receiver.samples[-1]['body_pos_w'][0,:2],fields['body_pos_w'][0,:2],atol=1e-7)
    assert not receiver.receive(Packet(0,1,2.32,fields),original['source_task_position_w'][450],original['source_task_quaternion_wxyz'][450],2.32)
