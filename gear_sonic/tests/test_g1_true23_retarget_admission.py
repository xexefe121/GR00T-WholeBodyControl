"""Admit only source-timed, physically bounded, kinematically coherent references."""
import hashlib
import json

import mujoco
import numpy as np
import pytest

import gear_sonic.scripts.evaluate_g1_true23_bfmzero as evaluator


@pytest.fixture
def candidate(tmp_path,monkeypatch):
    module,model,physics=evaluator.prepare_true23_model(
        evaluator.ROOT.parent/"GR00T-WholeBodyControl"/evaluator.MODEL,evaluator.ROOT/evaluator.PHYSICS)
    data=mujoco.MjData(model)
    mujoco.mj_kinematics(model,data)
    motion=dict(fps=np.array([50.]),joint_pos=np.tile(data.qpos[7:],(3,1)),joint_vel=np.zeros((3,23)),
                body_pos_w=np.tile(data.xpos[1:],(3,1,1)),body_quat_w=np.tile(data.xquat[1:],(3,1,1)),
                body_lin_vel_w=np.zeros((3,24,3)),body_ang_vel_w=np.zeros((3,24,3)))
    original={key:value.copy() for key,value in motion.items()}
    monkeypatch.setattr(evaluator,"load_motion",lambda name:(original,{},tmp_path/"original.npz"))
    monkeypatch.setattr(evaluator,"prepare_true23_model",lambda *args:(module,model,physics))
    path=tmp_path/"reference.npz"
    def save():
        np.savez(path,**motion)
        receipt=dict(clip="test",full_original_timeline=True,
                     reference_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        (tmp_path/"report.json").write_text(json.dumps(receipt))
    return motion,path,save


def test_valid_native_reference_is_admitted(candidate):
    motion,path,save=candidate
    save()
    loaded,_,_=evaluator.load_case_motion("test",path)
    for key,value in motion.items():
        np.testing.assert_array_equal(loaded[key],value)


@pytest.mark.parametrize("corruption,reason",[
    ("joint_velocity","derivative"),("angular_velocity","angular velocity"),
    ("joint_jump","exceeds native speed"),("joint_range","position limits"),
    ("body_geometry","inconsistent with native FK")])
def test_receipted_but_impossible_motion_is_rejected(candidate,corruption,reason):
    motion,path,save=candidate
    if corruption=="joint_velocity":
        motion["joint_vel"][:]=1e6
    elif corruption=="angular_velocity":
        motion["body_ang_vel_w"][:]=1e6
    elif corruption=="joint_jump":
        motion["joint_pos"][1:,3]=.8
        motion["joint_vel"][:]=np.gradient(motion["joint_pos"],.02,axis=0)
    elif corruption=="joint_range":
        motion["joint_pos"][:,0]=10.
    else:
        motion["body_pos_w"][:,6,0]+=.2
    save()
    with pytest.raises(ValueError,match=reason):
        evaluator.load_case_motion("test",path)


def test_changed_reference_bytes_are_rejected(candidate):
    motion,path,save=candidate
    save()
    motion["body_pos_w"][:,0,0]+=.1
    np.savez(path,**motion)
    with pytest.raises(ValueError,match="bytes differ"):
        evaluator.load_case_motion("test",path)
